import unittest

from typellm import TypeLLMClient

from tests.test_batching import FakeServer


class PrefillTests(unittest.TestCase):
    QUESTIONS = {
        "total": {"type": "number"},
        "item": {"type": "string"},
        "paid": {"type": "boolean"},
    }

    def run_generate(self, questions, **kwargs):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions=questions, **kwargs)
        return client, result

    def test_open_fields_continue_from_a_prefilled_key(self):
        client, result = self.run_generate(self.QUESTIONS)
        self.assertEqual(result, {"total": 7, "item": "blue", "paid": True})
        grammar = [p for p in client.sglang.payloads if not isinstance(p["sampling_params"], dict)
                   and "regex" in p["sampling_params"][0]]
        # Numbers and strings continue from '{"name":'; the model writes the space, digits or quote.
        prompts = [t for p in grammar for t in p["text"]]
        self.assertTrue(any(t.endswith('{"total":') for t in prompts))
        self.assertTrue(any(t.endswith('{"item":') for t in prompts))
        # Choices are scored at {"paid": " — the next token is the label.
        labels = [t for p in client.sglang.requests("score") for t in
                  ([p["text"]] if isinstance(p["text"], str) else p["text"]) if t.endswith('{"paid": "')]
        self.assertEqual(len(labels), 1)

    def test_history_holds_the_closed_object(self):
        client, _ = self.run_generate(self.QUESTIONS)
        self.assertIn('{"total": 7}', client.last_prompts[0])
        self.assertIn('{"item": "blue"}', client.last_prompts[1])

    def test_dag_children_extend_the_closed_parent(self):
        client, _ = self.run_generate({
            "total": {"type": "number"},
            "big": {"type": "boolean", "depends_on": ["total"]},
        })
        child = client.last_prompts[1]
        self.assertIn('{"total": 7', child)
        self.assertTrue(child.startswith(client.last_prompts[0].rsplit('{"total": ', 1)[0]))


def text_reply(text, finish="stop"):
    return lambda path, payload=None, **kw: [
        {"text": text, "meta_info": {"finish_reason": {"type": finish}}} for _ in payload["text"]]


class StringConstraintTests(unittest.TestCase):
    def test_max_length_caps_tokens_not_the_grammar(self):
        client = FakeServer()
        client.generate_texts(['{"a":', '{"b":'], [12, None], after_key=True)
        [payload] = [p for p in client.payloads if "regex" in p["sampling_params"][0]]
        first, second = payload["sampling_params"]
        self.assertEqual(first["regex"], second["regex"])
        self.assertTrue(first["regex"].startswith(' ?"') and first["regex"].endswith('*"\\}'))
        self.assertEqual((first["max_new_tokens"], second["max_new_tokens"]), (15, client.text_max_tokens))

    def test_the_model_writes_the_opening_quote(self):
        # A merged start such as ' "$' keeps its first character; no space works too.
        client = FakeServer()
        for text, value in [(' "$12.50"}', "$12.50"), ('"(555) 123"}', "(555) 123"), (' ""}', "")]:
            client._request = text_reply(text)
            self.assertEqual(client.generate_texts(['{"a":'], [None], after_key=True), [value])

    def test_escaped_quotes_decode_as_json(self):
        client = FakeServer()
        client._request = text_reply(' "say \\"hi\\""}')
        self.assertEqual(client.generate_texts(['{"a":'], [None], after_key=True), ['say "hi"'])

    def test_a_long_finished_string_is_cut_to_max_length(self):
        client = FakeServer()
        client._request = text_reply('"Edamame and more"}')
        self.assertEqual(client.generate_texts(['{"a":'], [7], after_key=True), ["Edamame"])

    def test_running_out_of_tokens_truncates_and_closes(self):
        client = FakeServer()
        client._request = text_reply(' "Edamame and', finish="length")
        self.assertEqual(client.generate_texts(['{"a":'], [7], after_key=True), ["Edamame"])
        client._request = text_reply(' "tab\\', finish="length")  # cut inside an escape
        self.assertEqual(client.generate_texts(['{"a":'], [7], after_key=True), ["tab"])
        client._request = text_reply(' "\u5bff\u53f8\u5bff', finish="length")
        self.assertEqual(client.generate_texts(['{"a":'], [2], after_key=True), ["寿司"])

    def test_without_max_length_an_unfinished_string_still_fails(self):
        from typellm import SGLangError
        client = FakeServer()
        client._request = text_reply("Edamame and", finish="length")
        with self.assertRaisesRegex(SGLangError, "did not complete normally"):
            client.generate_texts(['{"a":'], [None], after_key=True)


if __name__ == "__main__":
    unittest.main()
