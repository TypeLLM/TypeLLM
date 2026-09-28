import unittest
from unittest.mock import patch

from typellm import TypeLLMClient, SGLangClient, SchemaError, compile_json_schema
from tests.test_typellm import FakeSGLang


class DependencyFake(FakeSGLang):
    thinking = False
    _continuation_parts = SGLangClient._continuation_parts

    def _prepare_answer_prefix(self, prompt):
        return self._finish_thinking(prompt) if self.thinking else prompt
    complete_chat_prefix = SGLangClient.complete_chat_prefix
    extend_chat_prefix = SGLangClient.extend_chat_prefix

    def render_chat(self, messages, *, add_generation_prompt, finish_thinking=True):
        prompt = super().render_chat(messages, add_generation_prompt=add_generation_prompt)
        if self.thinking and add_generation_prompt and finish_thinking:
            return self._finish_thinking(prompt)
        return prompt

    def _finish_thinking(self, prompt):
        return prompt + '<think>retained reasoning</think>'


class DependencyTests(unittest.TestCase):
    def client(self, selected=None):
        client = TypeLLMClient('http://unused')
        client.sglang = DependencyFake(selected or [65] * 20)
        return client

    def test_diamond_forward_reference_and_branch_isolation(self):
        client = self.client()
        questions = {
            'final': {'type': 'boolean', 'depends_on': ['left', 'right']},
            'left': {'type': 'boolean', 'depends_on': ['root']},
            'unrelated': {'type': 'boolean'},
            'right': {'type': 'boolean', 'depends_on': ['root']},
            'root': {'type': 'boolean', 'return_probabilities': True},
        }
        result = client.generate(context='shared', questions=questions).result
        self.assertEqual(list(result), list(questions))
        self.assertEqual([len(b) for b in client.sglang.batch_prompts], [2, 2, 1])
        middle = client.sglang.batch_prompts[1]
        for prompt in middle:
            self.assertIn('"root": true', prompt)
            self.assertNotIn('unrelated', prompt)
            self.assertNotIn('probabilities', prompt)
        final = client.sglang.batch_prompts[2][0]
        for name in ('root', 'left', 'right'):
            self.assertIn(f'"{name}": true', final)
        self.assertNotIn('unrelated', final)
        self.assertIn('Field: "final"', client._last_prompts.get()[0])
        self.assertTrue(result['root']['value'])

    def test_incremental_prefixes_and_unique_warmups(self):
        client = self.client()
        client.generate(context='long context', questions={
            'a': {'type': 'boolean', 'depends_on': []},
            'b': {'type': 'boolean', 'depends_on': ['a']},
            'c': {'type': 'boolean', 'depends_on': ['a']},
            'd': {'type': 'boolean', 'depends_on': ['b', 'c']},
        }).result
        a, b, c, d = client._last_prompts.get()
        self.assertTrue(b.startswith(a))
        self.assertTrue(c.startswith(a))
        self.assertTrue(d.startswith(b))
        self.assertNotIn('name="c"', b)
        self.assertNotIn('name="b"', c)
        self.assertEqual(client.sglang.cached_prefixes, [
            '<user>long context</user>', a, b,
        ])
        # Completed prefixes retain the exact prompt sent for scoring.
        for batch, completed in zip(client.sglang.batch_prompts, [[a], [b, c], [d]]):
            for prompt, final in zip(batch, completed):
                self.assertTrue(final.startswith(prompt))

    def test_thinking_prefix_retained_across_dependency(self):
        client = self.client()
        client.sglang.thinking = True
        client.generate(context='', questions={
            'a': {'type': 'boolean'},
            'b': {'type': 'boolean', 'depends_on': ['a']},
        }).result
        a, b = client._last_prompts.get()
        self.assertTrue(b.startswith(a))
        self.assertEqual(a.count('<think>retained reasoning</think>'), 1)
        self.assertEqual(b.count('<think>retained reasoning</think>'), 2)

    def test_twenty_four_candidates_in_dependency_graph(self):
        client = self.client([ord('X'), ord('A')])
        result = client.generate(context='', questions={
            'pick': {'type': 'integer', 'enum': list(range(24)),
                     'return_probabilities': True},
            'check': {'type': 'boolean', 'depends_on': ['pick']},
        }).result
        self.assertEqual(result['pick']['value'], 23)
        self.assertEqual(len(result['pick']['probabilities']), 24)
        self.assertEqual(list(client.label_token_map)[:24], list('ABCDEFGHIJKLMNOPQRSTUVWX'))
        self.assertIn('"pick": 23', client._last_prompts.get()[1])
        self.assertTrue(client._last_prompts.get()[1].startswith(client._last_prompts.get()[0]))

    def test_always_thinking_template_continuation_runs_once(self):
        client = SGLangClient()

        class Tokenizer:
            chat_template = 'test'
            def apply_chat_template(self, messages, **kwargs):
                history = ''.join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
                return history + ('<assistant><think>' if kwargs['add_generation_prompt'] else '')

        client._chat_tokenizer = Tokenizer()
        with patch.object(client, '_finish_thinking', side_effect=lambda p: p + 'reasoning</think>') as finish:
            prompt = client.render_chat([{'role': 'user', 'content': 'root'}], add_generation_prompt=True)
            parent = client.complete_chat_prefix(prompt, 'A')
            child = client.extend_chat_prefix(parent, 'child')
            self.assertTrue(child.startswith(parent))
            self.assertEqual(finish.call_count, 2)
            self.assertTrue(finish.call_args.args[0].startswith(parent))

    def test_cli_has_no_execution_mode(self):
        from typellm.cli import main
        with patch('sys.argv', ['typellm']), patch('typellm.cli.TypeLLMClient') as factory, patch('builtins.print'), patch('typellm.cli.logging.basicConfig'):
            factory.return_value.generate.return_value.result = {}
            main()
            self.assertNotIn('execution', factory.call_args.kwargs)
        with patch('sys.argv', ['typellm', '--execution', 'dag']), patch('sys.stderr'), self.assertRaises(SystemExit):
            main()

    def test_invalid_graph_before_tokenizer_or_inference(self):
        cases = [
            {'a': {'type': 'boolean', 'depends_on': value}}
            for value in (None, 'b', [1], [''], ['a'], ['missing'], ['b', 'b'])
        ]
        cases.append({'a': {'type': 'boolean', 'depends_on': ['b']},
                      'b': {'type': 'boolean', 'depends_on': ['a']}})
        for questions in cases:
            with self.subTest(questions=questions), self.assertRaises(SchemaError):
                compile_json_schema({'type': 'object', 'properties': questions})

    def test_empty_dependencies(self):
        client = self.client()
        client.generate(context='', questions={'a': {'type': 'boolean', 'depends_on': []}, 'b': {'type': 'boolean'}}).result
        self.assertEqual(len(client.sglang.batch_prompts[0]), 2)

    def test_numeric_dependency_uses_semantic_value(self):
        client = self.client([ord('7'), 3, 65])
        result = client.generate(context='', questions={
            'number': {'type': 'integer', 'depends_on': []},
            'check': {'type': 'boolean', 'depends_on': ['number']},
        }).result
        self.assertEqual(result, {'number': 7, 'check': True})
        self.assertTrue(client._last_prompts.get()[1].startswith(client._last_prompts.get()[0]))
        self.assertIn('"number": 7', client.sglang.batch_prompts[0][0])

    def test_text_dependency_and_schema_interface(self):
        client = self.client()
        client.sglang.generate_texts = lambda prompts, **kwargs: ['hello "世界"'] * len(prompts)
        result = client.generate(context='', schema={'type': 'object', 'properties': {
            'text': {'type': 'string'},
            'check': {'type': 'boolean', 'depends_on': ['text']},
        }}).result
        self.assertEqual(result['text'], 'hello "世界"')
        self.assertTrue(client._last_prompts.get()[1].startswith(client._last_prompts.get()[0]))
        self.assertIn('"text": "hello \\"世界\\""', client.sglang.batch_prompts[0][0])

    def test_fields_without_dependencies_run_together(self):
        client = self.client()
        client.generate(context='', questions={'a': {'type': 'boolean'}, 'b': {'type': 'boolean'}}).result
        self.assertEqual([len(batch) for batch in client.sglang.batch_prompts], [2])
        self.assertFalse(client.sglang.prompts)

    def test_execution_is_no_longer_an_option(self):
        with self.assertRaises(TypeError):
            TypeLLMClient(execution='sequential')
        with self.assertRaises(TypeError):
            self.client().generate(context='', questions={'a': {'type': 'boolean'}}, execution='dag').result


if __name__ == '__main__':
    unittest.main()
