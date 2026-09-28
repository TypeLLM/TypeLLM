import unittest
from unittest.mock import patch

from typellm import TypeLLMClient, SchemaError, compile_json_schema, run_schema
from tests.test_typellm import FakeSGLang


class FieldProbabilityTests(unittest.TestCase):
    def test_selected_fields(self):
        for field_type, candidates in (
            ('string', ['meal', 'travel']),
            ('integer', [1, 2]),
            ('number', [0.1, 0.5]),
        ):
            with self.subTest(field_type=field_type):
                fake = FakeSGLang([ord('B'), ord('A'), ord('B')])
                with patch('typellm.runtime.SGLangClient', return_value=fake):
                    result = run_schema(context='receipt', questions={
                        'choice': {'type': field_type, 'enum': candidates, 'return_probabilities': True},
                        'flag': {'type': 'boolean', 'return_probabilities': True},
                        'plain': {'type': 'boolean', 'return_probabilities': False},
                    }).result
                self.assertEqual(result['choice']['value'], candidates[1])
                self.assertEqual(set(result['choice']['probabilities']), set(candidates))
                self.assertAlmostEqual(sum(result['choice']['probabilities'].values()), 1)
                self.assertIs(result['flag']['value'], True)
                self.assertEqual(set(result['flag']['probabilities']), {True, False})
                self.assertIs(result['plain'], False)

    def test_invalid_options_fail_before_inference(self):
        fields = [
            {'type': kind, 'return_probabilities': value}
            for kind in ('string', 'integer', 'number')
            for value in (True, False)
        ] + [
            {'type': 'boolean', 'return_probabilities': value}
            for value in (None, 0, 1, 'true')
        ]
        client = TypeLLMClient("http://127.0.0.1:30000")
        with patch.object(client.sglang, 'single_token') as tokens:
            for field in fields:
                with self.subTest(field=field), self.assertRaises(SchemaError):
                    client.generate(context='x', questions={'invalid': field}).result
            tokens.assert_not_called()

    def test_default_is_plain(self):
        decisions = compile_json_schema({'type': 'object', 'properties': {
            'choice': {'type': 'string', 'enum': ['a']},
            'flag': {'type': 'boolean'},
        }})
        self.assertTrue(all(not d.return_probabilities for d in decisions))


if __name__ == '__main__':
    unittest.main()
