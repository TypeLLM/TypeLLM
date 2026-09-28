import json
import unittest
from unittest.mock import patch
from typellm import TypeLLMClient, SGLangClient, SGLangError, SchemaError, compile_json_schema, run_schema
from tests.test_typellm import FakeSGLang, FakeChatTokenizer


class TextTests(unittest.TestCase):
    def test_optional_length_and_zero(self):
        for field, limit in [({'type': 'string'}, None), ({'type':'string','maxLength':0},0)]:
            d=compile_json_schema({'type':'object','properties':{'t':field}})[0]
            self.assertTrue(d.text_type)
            self.assertEqual(d.max_length,limit)
        for limit in (-1, True, 1.5, None):
            with self.assertRaises(SchemaError):
                compile_json_schema({'type':'object','properties':{'t':{'type':'string','maxLength':limit}}})

    def test_transport_validation_and_unicode(self):
        client=SGLangClient(text_max_tokens=17)
        value='你好😀\n"\\'
        response=[{'text':json.dumps(value),'meta_info':{'finish_reason':{'type':'stop'}}}]
        with patch.object(client,'_request',return_value=response) as request:
            self.assertEqual(client.generate_texts(['p'],[len(value)]),[value])
            params=request.call_args.args[1]['sampling_params'][0]
            self.assertEqual(json.loads(params['json_schema'])['maxLength'],len(value))
            self.assertEqual(params['max_new_tokens'],17)
        for raw, finish, limit in [('"abc"','length',None),('"abc','stop',None),('123','stop',None),('"ab"','stop',1),('"\\ud800"','stop',None)]:
            with self.subTest(raw=raw,finish=finish), patch.object(client,'_request',return_value=[{'text':raw,'meta_info':{'finish_reason':{'type':finish}}}]):
                with self.assertRaises(SGLangError):
                    client.generate_texts(['p'],[limit])
        with patch.object(client,'_request',return_value=[{'text':'""','meta_info':{'finish_reason':{'type':'stop'}}}]):
            self.assertEqual(client.generate_texts(['p'],[0]),[''])

    def test_mixed_fields_run_together(self):
        client=TypeLLMClient("http://127.0.0.1:30000")
        fake=FakeSGLang([ord('7'),3,ord('A')])
        calls=[]
        def generate(prefixes, limits, **kwargs):
            calls.append(prefixes)
            return ['alpha' if p.rfind('Field: "a"') > p.rfind('Field: "b"') else 'beta' for p in prefixes]
        fake.generate_texts=generate
        client.sglang=fake
        result=client.generate(state='context',questions={
            'a':{'type':'string'},'n':{'type':'integer'},
            'b':{'type':'string','maxLength':10},'ok':{'type':'boolean','return_probabilities':True}})
        self.assertEqual(list(result),['a','n','b','ok'])
        self.assertEqual([result['a'],result['n'],result['b'],result['ok']['value']],['alpha',7,'beta',True])
        self.assertIn(True,result['ok']['probabilities'])
        # Both strings go out in one request, and no field sees another's answer.
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0]),2)
        self.assertNotIn('alpha',calls[0][1])
        self.assertNotIn('alpha',fake.batch_prompts[0][0])

    def test_text_only_and_wrapper_budget(self):
        fake=FakeSGLang()
        fake.generate_texts=lambda prefixes,limits,**kwargs:['hello']*len(prefixes)
        with patch('typellm.runtime.SGLangClient',return_value=fake) as constructor:
            self.assertEqual(run_schema(state='x',questions={'t':{'type':'string'}},text_max_tokens=24),{'t':'hello'})
            self.assertEqual(constructor.call_args.kwargs['text_max_tokens'],24)
        for budget in (0,-1,True):
            with self.assertRaises(ValueError):
                TypeLLMClient("http://127.0.0.1:30000", text_max_tokens=budget)

    def test_thinking_then_constrained_text(self):
        client=TypeLLMClient("http://127.0.0.1:30000")
        client.sglang._context_length_cache=8192
        client.sglang._chat_tokenizer=FakeChatTokenizer()
        client.sglang._chat_tokenizer.apply_chat_template=lambda *args,**kw: "assistant\n<think>\n" if kw["enable_thinking"] else "completed"
        responses=[{'text':'brief</think>'},[{'text':' "done"}','meta_info':{'finish_reason':{'type':'stop'}}}]]
        with patch.object(client.sglang,'_request',side_effect=responses) as request:
            self.assertEqual(client.generate(state='x',questions={'t':{'type':'string','thinking':True}}),{'t':'done'})
            self.assertIn('</think>',request.call_args.args[1]['text'][0])
            self.assertTrue(request.call_args.args[1]['text'][0].endswith('{"t":'))
            self.assertIn('regex',request.call_args.args[1]['sampling_params'][0])

if __name__=='__main__':
    unittest.main()
