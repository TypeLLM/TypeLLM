import math
import random
import unittest
from unittest.mock import patch

from typellm import TypeLLMClient, SchemaError, compile_json_schema
from collections import Counter

from typellm.runtime import Choice, _balanced_orders, _choice_orderings
from tests.test_dependencies import DependencyFake


class PermutationTests(unittest.TestCase):
    def run_case(self, dependent, values, **options):
        client = TypeLLMClient("http://127.0.0.1:30000", seed=42, **options)
        client.sglang = DependencyFake([65] * 100)
        fields = {
            'roll': {'type': 'string' if isinstance(values[0], str) else 'integer',
                     'enum': values, 'permutations': 'all', 'return_probabilities': True},
            'flag': {'type': 'boolean'},
        }
        if dependent:
            fields['flag']['depends_on'] = ['roll']
        return client, client.generate(context='test', questions=fields).result

    def test_position_bias_cancels_and_non_enum_runs_once(self):
        for dependent in (False, True):
            for values in (['one', 'two', 'three'], [1, 2, 3]):
                with self.subTest(dependent=dependent, values=values):
                    client, result = self.run_case(dependent, values)
                    for probability in result['roll']['probabilities'].values():
                        self.assertAlmostEqual(probability, 1 / 3)
                    self.assertTrue(result['flag'])
                    count = sum(map(len, client.sglang.batch_prompts)) + len(client.sglang.prompts)
                    self.assertEqual(count, 7)
                    if dependent:
                        self.assertIn('"roll":', client.sglang.batch_prompts[-1][0])
                        self.assertNotIn('probabilities', client.sglang.batch_prompts[-1][0])

    def test_sampling_uses_averaged_distribution(self):
        with patch('typellm.runtime._sample', return_value='B') as sample:
            _client, result = self.run_case(False, ['one', 'two', 'three'], temperature=.5)
        self.assertEqual(result['roll']['value'], 'two')
        self.assertEqual(sample.call_count, 2)  # one final choice plus the boolean
        for p in sample.call_args_list[0].args[0].values():
            self.assertAlmostEqual(p, 1 / 3)

    def test_sampling_unique_reproducible_without_factorial_allocation(self):
        d = Choice('q', dict(zip('ABCDEFGHIJKLMNOPQRSTUVWX', range(24))), permutations=8)
        first = _choice_orderings(d, random.Random(42))
        second = _choice_orderings(d, random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(len({order for _, order in first}), 8)
        self.assertNotEqual(first, _choice_orderings(d, random.Random(43)))

    def test_validation_before_tokenizer_or_inference(self):
        invalid = [
            {'type': 'boolean', 'permutations': 1},
            {'type': 'string', 'permutations': 2},
            {'type': 'number', 'permutations': 2},
        ] + [{'type': 'string', 'enum': ['a', 'b'], 'permutations': v}
             for v in (0, -1, True, False, 2.0, None, '8', 'AUTO', {}, [])]
        invalid.append({'type': 'integer', 'enum': list(range(7)), 'permutations': 'all'})
        client = TypeLLMClient("http://127.0.0.1:30000")
        with patch.object(client.sglang, 'single_token') as tokens:
            for field in invalid:
                with self.subTest(field=field), self.assertRaises(SchemaError):
                    client.generate(context='x', questions={'x': field}).result
            tokens.assert_not_called()

    def test_plain_output_boolean_enum_and_clamped_budget(self):
        client = TypeLLMClient("http://127.0.0.1:30000")
        client.sglang = DependencyFake([65] * 3)
        result = client.generate(context='x', questions={
            'x': {'type': 'boolean', 'enum': [False, True], 'permutations': 100},
            'y': {'type': 'string', 'enum': ['a'], 'permutations': 'all'},
        }).result
        self.assertIsInstance(result['x'], bool)
        self.assertEqual(result['y'], 'a')
        self.assertEqual(len(client.sglang.batch_prompts[0]), 3)

    def test_average_probabilities_not_logits(self):
        client = TypeLLMClient("http://127.0.0.1:30000")
        fake = DependencyFake()
        client.sglang = fake
        with patch.object(fake, 'score_candidates_batch', return_value=([
            ({65: math.log(.9), 66: math.log(.1)}, {}),
            ({65: math.log(.7), 66: math.log(.3)}, {}),
        ], 0.0)):
            result = client.generate(context='x', questions={'x': {
                'type': 'string', 'enum': ['a', 'b'], 'permutations': 'all',
                'return_probabilities': True}}).result
        self.assertAlmostEqual(result['x']['probabilities']['a'], .6)
        self.assertAlmostEqual(result['x']['probabilities']['b'], .4)
        self.assertEqual(result['x']['value'], 'a')

    def test_auto_balances_positions_and_neighbours(self):
        for n in range(2, 10):
            with self.subTest(n=n):
                orders = _balanced_orders(n)
                self.assertEqual(len(orders), n if n % 2 == 0 else 2 * n)
                self.assertTrue(all(sorted(order) == list(range(n)) for order in orders))
                positions = Counter((pos, item) for order in orders for pos, item in enumerate(order))
                self.assertEqual(set(positions.values()), {len(orders) // n})
                pairs = Counter(pair for order in orders for pair in zip(order, order[1:]))
                self.assertEqual(len(pairs), n * (n - 1))
                self.assertEqual(len(set(pairs.values())), 1)

    def test_auto_does_not_depend_on_the_enum_order(self):
        def value_orders(values):
            d = Choice('q', dict(zip('ABCDEF', values)), permutations='auto')
            return sorted(tuple(variant.choices.values()) for variant, _ in _choice_orderings(d, random.Random(0)))
        self.assertEqual(value_orders(['a', 'b', 'c', 'd', 'e', 'f']), value_orders(['d', 'f', 'a', 'c', 'e', 'b']))
        self.assertEqual(len(value_orders(['a', 'b', 'c', 'd', 'e', 'f'])), 6)

    def test_auto_cancels_a_pure_position_bias(self):
        client = TypeLLMClient("http://127.0.0.1:30000")
        client.sglang = DependencyFake()
        biased = {65: math.log(.5), 66: math.log(.2), 67: math.log(.15), 68: math.log(.15)}
        with patch.object(client.sglang, 'score_candidates_batch',
                          side_effect=lambda prompts, ids: ([(biased, {}) for _ in prompts], 0.0)) as score:
            result = client.generate(context='x', questions={'x': {
                'type': 'string', 'enum': ['w', 'x', 'y', 'z'], 'permutations': 'auto',
                'return_probabilities': True}}).result
        self.assertEqual(len(score.call_args.args[0]), 4)
        for probability in result['x']['probabilities'].values():
            self.assertAlmostEqual(probability, 1 / 4)

    def test_one_matches_default(self):
        outputs = []
        for setting in ({}, {'permutations': 1}):
            client = TypeLLMClient("http://127.0.0.1:30000")
            client.sglang = DependencyFake([66])
            outputs.append(client.generate(context='x', questions={'x': {
                'type': 'string', 'enum': ['a', 'b'], 'return_probabilities': True, **setting}}).result)
        self.assertEqual(*outputs)


if __name__ == '__main__':
    unittest.main()
