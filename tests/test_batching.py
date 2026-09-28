import json
import unittest

from typellm import SGLangClient, TypeLLMClient
from typellm.images import encode_image

from tests.test_images import PNG, VisionTokenizer, fake_detokenize, fake_tokenize

THINK_STOP = "</think>"


class ThinkingTokenizer(VisionTokenizer):
    def apply_chat_template(self, messages, *, add_generation_prompt, enable_thinking=False, **kwargs):
        rendered = super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        return rendered + ("<think>\n" if add_generation_prompt and enable_thinking else "")


def is_number_pattern(pattern):
    """Whether a regex is typellm's numeric one rather than the string one."""
    return pattern.startswith((" ?-?", " ?(?:-?"))


class FakeServer(SGLangClient):
    """A real SGLangClient whose HTTP layer answers like SGLang.

    Numeric fields decode to 7: the server prefers the end token, then "7".
    """

    def __init__(self):
        super().__init__(model="fake")
        self._chat_tokenizer = ThinkingTokenizer()
        self._context_length_cache = 100_000
        self.payloads = []

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/v1/tokenize":
            return {"tokens": fake_tokenize(payload["prompt"])}
        if path == "/v1/detokenize":
            return {"text": fake_detokenize(payload["tokens"])}
        assert path == "/generate", path
        self.payloads.append(payload)
        texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
        params = payload["sampling_params"]
        params = params if isinstance(params, list) else [params] * len(texts)
        if "token_ids_logprob" in payload:
            rows = payload["token_ids_logprob"]
            rows = [rows] if isinstance(rows[0], int) else rows
            out = []
            for ids in rows:
                # Like the real model: a number starts with the lone space token.
                pick = ord(" ") if ord(" ") in ids else 1 if 1 in ids else ord("7") if ord("7") in ids else ids[0]
                out.append({"meta_info": {"output_token_ids_logprobs": [
                    [[0.0 if t == pick else -9.0, t, "?"] for t in ids]]}})
        elif params[0].get("stop") == [THINK_STOP]:
            out = [{"text": "Reasoned." + THINK_STOP, "meta_info": {}} for _ in texts]
        elif "regex" in params[0]:
            # The prompt ends with '{"name":'; a number's digits, or a string's quote,
            # characters and '"}' follow.
            out = [{"text": " 7}" if is_number_pattern(p["regex"]) else ' "blue"}',
                    "meta_info": {"finish_reason": {"type": "stop"}}} for p in params]
        elif "json_schema" in params[0]:
            out = []
            for p in params:
                schema = json.loads(p["json_schema"])
                # An object schema answers {"key": "blue"}; a plain string schema "blue".
                value = {next(iter(schema["properties"])): "blue"} if schema["type"] == "object" else "blue"
                out.append({"text": json.dumps(value), "meta_info": {"finish_reason": {"type": "stop"}}})
        else:
            out = [{"meta_info": {"prompt_tokens": 900}} for _ in texts]
        return out[0] if isinstance(payload["text"], str) else out

    def requests(self, kind):
        def matches(p):
            params = p["sampling_params"]
            first = params[0] if isinstance(params, list) else params
            return {"score": "token_ids_logprob" in p,
                    "think": first.get("stop") == [THINK_STOP],
                    "count": first.get("max_new_tokens") == 0}[kind]
        return [p for p in self.payloads if matches(p)]


def width(payload):
    return 1 if isinstance(payload["text"], str) else len(payload["text"])


def thinks(questions):
    return {name: {**field, "thinking": True} for name, field in questions.items()}


class PrefixWarmupTests(unittest.TestCase):
    MIXED = {"a": {"type": "integer"}, "b": {"type": "number"}, "n": {"type": "string"},
             "k": {"type": "string", "enum": ["x", "y"]}}

    def test_the_context_is_warmed_before_any_batch_that_shares_it(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
                client.sglang = FakeServer()
                client.generate(context="Receipt", questions=thinks(self.MIXED) if thinking else self.MIXED).result
                warm = client.sglang.requests("count")
                self.assertEqual(len(warm), 1)
                # First, before the thinking, number, text and scoring batches.
                self.assertIs(client.sglang.payloads[0], warm[0])
                self.assertIn("Receipt", warm[0]["text"])
                self.assertNotIn('{"a":', warm[0]["text"])

    def test_open_fields_alone_are_not_warmed(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions={"a": {"type": "integer"}, "n": {"type": "string"}}).result
        self.assertEqual(client.sglang.requests("count"), [])


class PerFieldThinkingTests(unittest.TestCase):
    def thinking_prompts(self, server):
        return [t for p in server.requests("think")
                for t in ([p["text"]] if isinstance(p["text"], str) else p["text"])]

    def test_only_fields_that_ask_for_it_think(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions={
            "hard": {"type": "boolean", "instructions": "Hard?", "thinking": True},
            "easy": {"type": "boolean", "instructions": "Easy?"},
            "count": {"type": "integer"},
        }).result
        self.assertEqual(result, {"hard": True, "easy": True, "count": 7})
        [prompt] = self.thinking_prompts(client.sglang)  # one batch, one prompt
        self.assertIn("Hard?", prompt)
        self.assertTrue(prompt.endswith("<think>\n"))
        answers = [t for p in client.sglang.payloads if p not in client.sglang.requests("think")
                   for t in ([p["text"]] if isinstance(p["text"], str) else p["text"])]
        self.assertTrue(any("Easy?" in t and "<think>" not in t for t in answers))

    def test_thinking_false_is_the_same_as_leaving_it_out(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions={
            "a": {"type": "boolean", "instructions": "A?", "thinking": True},
            "b": {"type": "boolean", "instructions": "B?", "thinking": True},
            "skip": {"type": "boolean", "instructions": "Skip?", "thinking": False},
        }).result
        prompts = self.thinking_prompts(client.sglang)
        self.assertEqual(len(client.sglang.requests("think")), 1)
        self.assertEqual(len(prompts), 2)
        self.assertFalse(any("Skip?" in t for t in prompts))

    def test_each_field_gets_its_own_budget(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        client.sglang.thinking_budget = 500  # the client's budget, for fields without their own
        client.generate(context="Receipt", questions={
            "short": {"type": "boolean", "instructions": "Short?", "thinking": True, "thinking_budget": 100},
            "long": {"type": "boolean", "instructions": "Long?", "thinking": True, "thinking_budget": 300},
            "default": {"type": "boolean", "instructions": "Default?", "thinking": True},
        }).result
        [request] = client.sglang.requests("think")
        budgets = {next(q for q in ("Short?", "Long?", "Default?") if q in t): params["max_new_tokens"]
                   for t, params in zip(request["text"], request["sampling_params"])}
        self.assertEqual(budgets, {"Short?": 100, "Long?": 300, "Default?": 500})

    def test_reasoning_and_its_tokens_are_reported_per_field(self):
        class Metered(FakeServer):
            def _request(self, path, payload=None, *, allow_text=False):
                response = super()._request(path, payload, allow_text=allow_text)
                if path == "/generate":
                    params = payload["sampling_params"]
                    thinking = (params[0] if isinstance(params, list) else params).get("stop") == [THINK_STOP]
                    for item in response if isinstance(response, list) else [response]:
                        item["meta_info"]["completion_tokens"] = 40 if thinking else 1
                return response

        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = Metered()
        done = client.generate(context="Receipt", questions={
            "hard": {"type": "boolean", "instructions": "Hard?", "thinking": True},
            "easy": {"type": "boolean", "instructions": "Easy?"},
        })
        self.assertEqual(done.thinking, {"hard": "Reasoned."})
        self.assertEqual(done.usage.thinking_tokens, 40)
        self.assertGreater(done.usage.completion_tokens, 40)  # answers count too
        self.assertEqual(client.generate(context="Receipt", questions={"x": {"type": "boolean"}}).thinking, {})

    def test_invalid_settings_are_schema_errors(self):
        from typellm import SchemaError
        for field in ({"type": "boolean", "thinking": "yes"}, {"type": "boolean", "thinking_budget": 0},
                      {"type": "boolean", "thinking_budget": 2.5}):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                TypeLLMClient("http://127.0.0.1:30000", model="fake").compile_schema(
                    {"type": "object", "properties": {"x": field}})


class BatchedThinkingTests(unittest.TestCase):
    QUESTIONS = {
        "flag": {"type": "boolean"},
        "count": {"type": "integer"},
        "name": {"type": "string"},
        "pick": {"type": "string", "enum": ["x", "y"], "permutations": "all"},
    }

    def test_one_thinking_request_covers_every_prompt_in_a_layer(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions=thinks(self.QUESTIONS)).result
        self.assertEqual(result, {"flag": True, "count": 7, "name": "blue", "pick": "x"})
        [think] = client.sglang.requests("think")
        # Four fields plus the reversed ordering of "pick".
        self.assertEqual(width(think), 5)
        self.assertEqual(len(think["sampling_params"]), 5)
        for payload in client.sglang.requests("score"):
            texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
            self.assertTrue(all("Reasoned." + THINK_STOP in t for t in texts))

    def test_dag_thinks_once_per_layer(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions=thinks({
            "a": {"type": "boolean"}, "b": {"type": "integer"},
            "c": {"type": "boolean", "depends_on": ["a", "b"]},
            "d": {"type": "string", "depends_on": ["a"]},
        })).result
        self.assertEqual([width(p) for p in client.sglang.requests("think")], [2, 2])

    def test_images_count_prompt_tokens_in_one_batch(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        client.generate(context="Receipt", images=[PNG],
                        questions=thinks({"a": {"type": "boolean"}, "b": {"type": "boolean"}})).result
        counts = [p for p in client.sglang.requests("count") if width(p) > 1]
        self.assertEqual([width(p) for p in counts], [2])
        [think] = client.sglang.requests("think")
        self.assertEqual(think["image_data"], encode_image(PNG))


if __name__ == "__main__":
    unittest.main()
