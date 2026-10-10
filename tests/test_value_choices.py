import unittest

from typellm import SchemaError, TypeLLMClient

from tests.test_dependencies import DependencyFake

VALUES = [f"category_{i}" for i in range(40)]


class FakeScorer:
    """Puts most of the probability on one value, wherever it is listed."""

    max_choices = 512

    def __init__(self, favourite="category_17", share=0.9):
        self.favourite, self.share = favourite, share
        self.calls = []

    def score(self, sglang, prompts, continuations):
        self.calls.append((list(prompts), [list(c) for c in continuations]))
        out = []
        for options in continuations:
            rest = (1 - self.share) / (len(options) - 1)
            out.append([self.share if option == f' "{self.favourite}"}}' else rest for option in options])
        return out


def client_with(scorer=None):
    client = TypeLLMClient("http://unused")
    client.sglang = DependencyFake([65] * 20)
    if scorer is not None:
        client.use_choice_scorer(scorer)
    return client


class ValueChoiceTests(unittest.TestCase):
    def test_more_choices_than_labels_need_a_scorer(self):
        with self.assertRaisesRegex(SchemaError, "the maximum is 26"):
            client_with().generate(context="x", questions={"c": {"type": "string", "enum": VALUES}})

    def test_a_scorer_answers_by_value(self):
        scorer = FakeScorer()
        client = client_with(scorer)
        answer = client.generate(context="x", questions={"c": {
            "type": "string", "enum": VALUES, "return_probabilities": True, "permutations": 1}}).result["c"]
        self.assertEqual(answer["value"], "category_17")
        self.assertAlmostEqual(answer["probabilities"]["category_17"], 0.9)
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
        self.assertAlmostEqual(answer["confidence"], (0.9 - 1 / 40) / (1 - 1 / 40))
        [(prompts, continuations)] = scorer.calls
        self.assertTrue(prompts[0].endswith('{"choice":'))
        self.assertIn('"category_39"', prompts[0])
        self.assertEqual(continuations[0][0], ' "category_0"}')
        self.assertIn('{"choice": "category_17"}', client._last_prompts.get()[0])

    def test_orders_are_averaged_and_capped(self):
        scorer = FakeScorer()
        client_with(scorer).generate(context="x", questions={"c": {
            "type": "string", "enum": VALUES, "return_probabilities": True, "permutations": 5}})
        [(prompts, continuations)] = scorer.calls
        self.assertEqual(len(prompts), 3)
        self.assertEqual(len({tuple(c) for c in continuations}), 3)

    def test_auto_scores_one_order(self):
        scorer = FakeScorer()
        client_with(scorer).generate(context="x", questions={"c": {
            "type": "string", "enum": VALUES, "return_probabilities": True}})
        [(prompts, _)] = scorer.calls
        self.assertEqual(len(prompts), 1)

    def test_a_short_enum_keeps_its_labels(self):
        scorer = FakeScorer()
        client_with(scorer).generate(context="x", questions={"c": {"type": "string", "enum": ["a", "b"]}})
        self.assertEqual(scorer.calls, [])

    def test_when_tests_a_choice_answered_by_value(self):
        client = client_with(FakeScorer())
        result = client.generate(context="x", questions={
            "c": {"type": "string", "enum": VALUES},
            "hit": {"type": "boolean", "when": {"c": "category_17"}},
            "miss": {"type": "boolean", "when": {"c": "category_3"}},
        })
        self.assertEqual(result.result["c"], "category_17")
        self.assertIn("hit", result.result)
        self.assertEqual(result.skipped, ["miss"])

    def test_array_items_keep_the_label_limit(self):
        with self.assertRaisesRegex(SchemaError, "the maximum is 26"):
            client_with(FakeScorer()).generate(context="x", questions={"cs": {
                "type": "array", "items": {"type": "string", "enum": VALUES}}})

    def test_the_scorer_sets_the_limit(self):
        with self.assertRaisesRegex(SchemaError, "the maximum is 512"):
            client_with(FakeScorer()).generate(context="x", questions={"c": {
                "type": "string", "enum": [str(i) for i in range(513)]}})


if __name__ == "__main__":
    unittest.main()
