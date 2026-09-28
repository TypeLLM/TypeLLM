import unittest

from typellm import SchemaError, TypeLLMClient, compile_json_schema

from tests.test_batching import FakeServer, is_number_pattern, width
from tests.test_typellm import FakeSGLang

def compile_field(field):
    return compile_json_schema({"type": "object", "properties": {"x": field}})[0]


class NullServer(FakeServer):
    """FakeServer whose model writes null wherever the grammar allows it."""

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and any("regex" in p for p in _params(payload)):
            self.payloads.append(payload)
            out = [{"text": " null}" if "null" in p["regex"] else " 7}" if is_number_pattern(p["regex"]) else ' "blue"}',
                    "meta_info": {"finish_reason": {"type": "stop"}}} for p in _params(payload)]
            return out[0] if isinstance(payload["text"], str) else out
        return super()._request(path, payload, allow_text=allow_text)


def _params(payload):
    params = payload.get("sampling_params", {})
    return params if isinstance(params, list) else [params]


class CompileTests(unittest.TestCase):
    def test_type_lists_with_null(self):
        self.assertTrue(compile_field({"type": ["string", "null"]}).nullable)
        self.assertTrue(compile_field({"type": ["null", "integer"]}).nullable)
        self.assertFalse(compile_field({"type": "number"}).nullable)
        self.assertEqual(compile_field({"type": ["boolean", "null"]}).choices, (True, False, None))

    def test_enums_allow_null_only_when_listed(self):
        self.assertEqual(compile_field({"type": ["string", "null"], "enum": ["a", None]}).choices, ("a", None))
        self.assertEqual(compile_field({"type": ["string", "null"], "enum": ["a", "b"]}).choices, ("a", "b"))
        with self.assertRaisesRegex(SchemaError, "do not match type"):
            compile_field({"type": "string", "enum": ["a", None]})

    def test_other_type_lists_are_rejected(self):
        for kinds in (["string", "integer"], ["null"], ["string", "null", "integer"], ["null", "null"]):
            with self.subTest(kinds=kinds), self.assertRaisesRegex(SchemaError, r'\[type, "null"\]'):
                compile_field({"type": kinds})


class RuntimeTests(unittest.TestCase):
    def test_nullable_number_can_answer_null(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = NullServer()
        result = client.generate(context="Receipt", questions={
            "tip": {"type": ["number", "null"]}, "count": {"type": "integer"},
        }).result
        self.assertEqual(result, {"tip": None, "count": 7})
        self.assertIn("Return null only if there is no value.", client._last_prompts.get()[0])

    def test_nullable_string_writes_null_in_its_text_request(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = NullServer()
        result = client.generate(context="Receipt", questions={
            "note": {"type": ["string", "null"]}, "name": {"type": "string"},
        }).result
        self.assertEqual(result, {"note": None, "name": "blue"})
        self.assertEqual(client.sglang.requests("score"), [])  # no separate null step
        # One request for both strings, from '{"note":' and '{"name":'; only the nullable one may be null.
        [text] = [p for p in client.sglang.payloads
                  if not isinstance(p["sampling_params"], dict) and "regex" in p["sampling_params"][0]]
        self.assertEqual(width(text), 2)
        self.assertTrue(text["text"][0].endswith('{"note":'))
        self.assertTrue(text["text"][1].endswith('{"name":'))
        self.assertIn("null", text["sampling_params"][0]["regex"])
        self.assertNotIn("null", text["sampling_params"][1]["regex"])
        self.assertIn('{"note": null}', client._last_prompts.get()[0])

    def test_nullable_boolean_scores_null_as_a_choice(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeSGLang(selected_ids=[ord("C")])
        result = client.generate(context="Receipt", questions={
            "paid": {"type": ["boolean", "null"], "return_probabilities": True},
        }).result
        self.assertIsNone(result["paid"]["value"])
        self.assertEqual(set(result["paid"]["probabilities"]), {True, False, None})

    def test_dependents_see_null(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = NullServer()
        client.generate(context="Receipt", questions={
            "tip": {"type": ["number", "null"]},
            "tipped": {"type": "boolean", "depends_on": ["tip"]},
        }).result
        self.assertIn('{"tip": null}', client._last_prompts.get()[1])


if __name__ == "__main__":
    unittest.main()
