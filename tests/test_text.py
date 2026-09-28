import json
import unittest
from unittest.mock import patch
from typellm import TypeLLMClient, SGLangClient, SGLangError, SchemaError, compile_json_schema, run_schema
from tests.test_typellm import FakeSGLang, FakeChatTokenizer


class TextTests(unittest.TestCase):
    def test_max_length_is_not_supported(self):
        self.assertTrue(compile_json_schema({'type':'object','properties':{'t':{'type':'string'}}})[0].text_type)
        for field in ({'type':'string','maxLength':10}, {'type':'string','enum':['a'],'maxLength':1}):
            with self.subTest(field=field), self.assertRaisesRegex(SchemaError, 'maxLength is not supported'):
                compile_json_schema({'type':'object','properties':{'t':field}})

    def test_transport_validation_and_unicode(self):
        client=SGLangClient(text_max_tokens=17)
        value='你好😀\n"\\'
        response=[{'text':json.dumps(value),'meta_info':{'finish_reason':{'type':'stop'}}}]
        with patch.object(client,'_request',return_value=response) as request:
            self.assertEqual(client.generate_texts(['p']),[value])
            params=request.call_args.args[1]['sampling_params'][0]
            self.assertEqual(json.loads(params['json_schema']),{'type':'string'})
            self.assertEqual(params['max_new_tokens'],17)
        for raw, finish in [('"abc"','length'),('"abc','stop'),('123','stop'),('"\\ud800"','stop')]:
            with self.subTest(raw=raw,finish=finish), patch.object(client,'_request',return_value=[{'text':raw,'meta_info':{'finish_reason':{'type':finish}}}]):
                with self.assertRaises(SGLangError):
                    client.generate_texts(['p'])
        with patch.object(client,'_request',return_value=[{'text':'""','meta_info':{'finish_reason':{'type':'stop'}}}]):
            self.assertEqual(client.generate_texts(['p']),[''])

    def test_mixed_fields_run_together(self):
        client=TypeLLMClient("http://127.0.0.1:30000")
        fake=FakeSGLang([ord('7'),3,ord('A')])
        calls=[]
        def generate(prefixes, **kwargs):
            calls.append(prefixes)
            return ['alpha' if p.rfind('Field: "a"') > p.rfind('Field: "b"') else 'beta' for p in prefixes]
        fake.generate_texts=generate
        client.sglang=fake
        result=client.generate(state='context',questions={
            'a':{'type':'string'},'n':{'type':'integer'},
            'b':{'type':'string'},'ok':{'type':'boolean','return_probabilities':True}}).result
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
        fake.generate_texts=lambda prefixes,**kwargs:['hello']*len(prefixes)
        with patch('typellm.runtime.SGLangClient',return_value=fake) as constructor:
            self.assertEqual(run_schema(state='x',questions={'t':{'type':'string'}},text_max_tokens=24).result,{'t':'hello'})
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
            self.assertEqual(client.generate(state='x',questions={'t':{'type':'string','thinking':True}}).result,{'t':'done'})
            self.assertIn('</think>',request.call_args.args[1]['text'][0])
            self.assertTrue(request.call_args.args[1]['text'][0].endswith('{"t":'))
            self.assertIn('regex',request.call_args.args[1]['sampling_params'][0])

if __name__=='__main__':
    unittest.main()
