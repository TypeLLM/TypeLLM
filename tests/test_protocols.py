import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from typellm import SGLangClient, SGLangError
from typellm.protocol import detect_protocol
from tests.test_typellm import FakeChatTokenizer


class ProtocolTokenizer(FakeChatTokenizer):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.chat_template = {
            'gemma': '<|turn>model <|channel>thought',
            'minicpm5': '<|im_start|>assistant <|im_end|>',
            'ling': '<role>ASSISTANT</role><|role_end|>',
            'ring': '<role>ASSISTANT</role><think>',
        }[family]
        self.eos_token, self.eos_token_id = '</s>', 1
        self.special = {'<turn|>': 2, '<|im_end|>': 3, '<|role_end|>': 4}

    def encode(self, text, *, add_special_tokens=False):
        return [self.special[text]] if text in self.special else super().encode(text)

    def decode(self, ids, **kwargs):
        return {v: k for k, v in self.special.items()}.get(ids[0], '')

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        close = {'minicpm5': '<|im_end|>\n',
                 'ling': '<|role_end|>', 'ring': ''}[self.family]
        text = ''.join(f"<{m['role']}>{m['content']}{close}" for m in messages)
        if kwargs['add_generation_prompt']:
            text += '<assistant>'
            if self.family == 'ring' or (self.family == 'minicpm5' and kwargs['enable_thinking']):
                text += '<think>\n'
        return text


class ProtocolTests(unittest.TestCase):
    def client(self, family, response=None):
        client = SGLangClient(thinking_budget=64)
        client._chat_tokenizer = ProtocolTokenizer(family)
        client._context_length_cache = 8192
        end = '</think>'
        client._request = Mock(return_value=response or {'text': 'REASONING' + end + 'discard'})
        return client

    def test_tokenizer_loading_disables_custom_model_code(self):
        tokenizer = ProtocolTokenizer('minicpm5')
        auto = Mock()
        auto.from_pretrained.return_value = tokenizer
        client = SGLangClient(tokenizer='/tokenizer')
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoTokenizer=auto)}):
            self.assertIs(client._get_chat_tokenizer(), tokenizer)
            self.assertIs(client._get_chat_tokenizer(), tokenizer)
        auto.from_pretrained.assert_called_once_with('/tokenizer', trust_remote_code=False)

    def test_a_server_path_missing_here_falls_back_to_the_served_name(self):
        tokenizer = ProtocolTokenizer('minicpm5')
        auto = Mock()
        auto.from_pretrained.side_effect = lambda source, **kwargs: (
            tokenizer if source == 'org/model' else (_ for _ in ()).throw(OSError(source)))
        client = SGLangClient()
        client._model_info = Mock(return_value={'tokenizer_path': '/models/qwen', 'model_path': '/models/qwen',
                                                'served_model_name': 'org/model'})
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoTokenizer=auto)}):
            self.assertIs(client._get_chat_tokenizer(), tokenizer)
        self.assertEqual([call.args[0] for call in auto.from_pretrained.call_args_list], ['/models/qwen', 'org/model'])

    def test_an_explicit_tokenizer_is_the_only_source_tried(self):
        auto = Mock()
        auto.from_pretrained.side_effect = OSError('missing')
        client = SGLangClient(tokenizer='/nowhere')
        client._model_info = Mock(return_value={'served_model_name': 'org/model'})
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoTokenizer=auto)}), \
                self.assertRaisesRegex(SGLangError, "'/nowhere'"):
            client._get_chat_tokenizer()
        auto.from_pretrained.assert_called_once()

    def test_gemma_is_rejected_before_inference(self):
        for thinking in (False, True):
            client = self.client('gemma')
            with self.assertRaisesRegex(SGLangError, 'Gemma 4.*not supported'):
                client.render_chat([], add_generation_prompt=True, thinking=thinking)
            client._request.assert_not_called()

    def test_native_turn_stop_recovers_preserved_and_filtered_terminators(self):
        for family, ending, token in [('minicpm5', '<|im_end|>', 3)]:
            for matched in (ending, token):
                for suffix in ('', ending + '\n'):
                    with self.subTest(family=family, matched=matched, suffix=suffix):
                        client = self.client(family, {
                            'text': 'Keep reasoning' + suffix,
                            'meta_info': {'finish_reason': {'type': 'stop', 'matched': matched}},
                        })
                        prefix = detect_protocol(client._chat_tokenizer).thinking_open + '\n'
                        with self.assertLogs('typellm', level='INFO') as logs:
                            prompt = client._finish_thinking(prefix)
                        self.assertTrue(prompt.startswith(prefix + 'Keep reasoning'))
                        self.assertNotIn(ending, prompt)
                        self.assertTrue(prompt.rstrip().endswith(detect_protocol(client._chat_tokenizer).thinking_close))
                        self.assertIn('native turn terminator', logs.output[0])

    def test_unknown_empty_or_aborted_thinking_is_not_recovered(self):
        for text, finish in [
            ('Partial', {'type': 'stop', 'matched': 'unknown'}),
            ('Partial', {'type': 'stop'}),
            ('Partial', {}),
            ('<|im_end|>', {'type': 'stop', 'matched': 3}),
            ('Partial<|im_end|>', {'type': 'abort', 'matched': 3}),
            ('Partial<|im_end|>', {'type': 'error', 'matched': 3}),
        ]:
            with self.subTest(text=text, finish=finish):
                client = self.client('minicpm5', {'text': text, 'meta_info': {'finish_reason': finish}})
                with self.assertRaises(SGLangError):
                    client._finish_thinking('<think>\n')

    def test_minicpm5_switch_and_ring_always_thinking(self):
        for family in ('minicpm5', 'ring'):
            for thinking in (False, True):
                client = self.client(family)
                prompt = client.render_chat([], add_generation_prompt=True, thinking=thinking)
                expected = thinking or family == 'ring'
                self.assertEqual(client._request.called, expected)
                parent = client.complete_chat_prefix(prompt, 'A')
                self.assertEqual('REASONING' in parent, expected)
                self.assertTrue(client.extend_chat_prefix(parent, 'next').startswith(parent))

    def test_non_thinking_templates_reject_requested_thinking(self):
        for family in ('ling',):
            client = self.client(family)
            with self.assertRaisesRegex(SGLangError, 'may not support thinking'):
                client.render_chat([], add_generation_prompt=True, thinking=True)
            client._request.assert_not_called()



if __name__ == '__main__':
    unittest.main()
