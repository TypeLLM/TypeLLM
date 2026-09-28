import itertools
import re
import unittest

from typellm import SGLangError, TypeLLMClient
from typellm.runtime import _numeric_text_is_complete, numeric_pattern

from tests.test_batching import FakeServer, is_number_pattern, width
from tests.test_nullable import NullServer


def regex_requests(server):
    return [p for p in server.payloads
            if any("regex" in q for q in (p["sampling_params"] if isinstance(p["sampling_params"], list)
                                          else [p["sampling_params"]]))]


class ReplyServer(FakeServer):
    """Answers every grammar request with one fixed text and finish reason."""

    def __init__(self, text, finish="stop"):
        super().__init__()
        self.reply, self.finish = text, finish

    def _request(self, path, payload=None, *, allow_text=False):
        params = payload.get("sampling_params") if payload else None
        if path == "/generate" and isinstance(params, list) and "regex" in params[0]:
            self.payloads.append(payload)
            return [{"text": self.reply, "meta_info": {"finish_reason": {"type": self.finish}}} for _ in params]
        return super()._request(path, payload, allow_text=allow_text)


class PatternTests(unittest.TestCase):
    """The regex admits exactly the numbers the parser accepts."""

    def samples(self, max_digits):
        short = ("".join(chars) for n in range(1, 6) for chars in itertools.product("-0.19", repeat=n))
        long = ["1" * max_digits, "1" * (max_digits + 1), "1" * (max_digits - 1) + ".5",
                "1" * max_digits + ".5", "0." + "5" * (max_digits - 1), "0." + "5" * max_digits,
                "-" + "9" * max_digits]
        return itertools.chain(short, long)

    def test_the_pattern_matches_the_parser(self):
        for numeric_type in ("number", "integer"):
            for max_digits in (32, 4):
                pattern = re.compile(numeric_pattern(numeric_type, max_digits))
                for text in self.samples(max_digits):
                    expected = (_numeric_text_is_complete(text, numeric_type)
                                and sum(c.isdigit() for c in text) <= max_digits)
                    # The value may start with the space, and end at '}' or the end of the message.
                    for written in (" " + text + "}", text + "}", " " + text, text):
                        with self.subTest(type=numeric_type, digits=max_digits, written=written):
                            self.assertEqual(bool(pattern.fullmatch(written)), expected)

    def test_null_only_where_nullable(self):
        self.assertTrue(re.fullmatch(numeric_pattern("number", 32, nullable=True), " null}"))
        self.assertFalse(re.fullmatch(numeric_pattern("number", 32), " null}"))


class GrammarNumberTests(unittest.TestCase):
    QUESTIONS = {"a": {"type": "integer"}, "b": {"type": "number"}, "c": {"type": "integer"}}

    def test_every_number_of_a_layer_takes_one_request(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions=self.QUESTIONS).result
        self.assertEqual(result, {"a": 7, "b": 7, "c": 7})
        self.assertEqual(client.sglang.requests("score"), [])
        [numbers] = regex_requests(client.sglang)
        self.assertEqual(width(numbers), 3)
        self.assertTrue(all(t.endswith('":') for t in numbers["text"]))
        self.assertEqual(numbers["sampling_params"][0]["temperature"], 0)
        self.assertIn('{"a": 7}', client._last_prompts.get()[0])

    def test_numbers_and_strings_of_a_layer_share_one_request(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions={
            "a": {"type": "integer"}, "n": {"type": "string"},
            "b": {"type": "number"}, "m": {"type": ["string", "null"]}}).result
        self.assertEqual(result, {"a": 7, "n": "blue", "b": 7, "m": "blue"})
        [request] = regex_requests(client.sglang)
        self.assertEqual(width(request), 4)
        patterns = [p["regex"] for p in request["sampling_params"]]
        self.assertEqual([is_number_pattern(p) for p in patterns], [True, True, False, False])
        self.assertTrue(all(t.endswith(k) for t, k in zip(request["text"], ('{"a":', '{"b":', '{"n":', '{"m":'))))
        self.assertIn("null", patterns[3])

    def test_sampling_passes_the_temperature_and_no_truncation(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake", temperature=0.7, seed=1)
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions={"a": {"type": "integer"}}).result
        [params] = regex_requests(client.sglang)[0]["sampling_params"]
        self.assertEqual((params["temperature"], params["top_p"], params["top_k"]), (0.7, 1.0, -1))

    def test_nullable_numbers_can_be_null_in_the_same_request(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = NullServer()
        result = client.generate(context="Receipt", questions={"tip": {"type": ["number", "null"]}}).result
        self.assertEqual(result, {"tip": None})
        self.assertEqual(client.sglang.requests("score"), [])
        self.assertIn('{"tip": null}', client._last_prompts.get()[0])

    def test_an_incomplete_or_cut_off_number_is_an_error(self):
        for reply, finish, error in ((" 3.}", "stop", ValueError), (" 12", "length", SGLangError)):
            with self.subTest(reply=reply):
                client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
                client.sglang = ReplyServer(reply, finish)
                with self.assertRaises(error):
                    client.generate(context="Receipt", questions={"a": {"type": "number"}}).result

    def test_negative_and_spaceless_numbers_parse(self):
        for reply, value in ((" -3.5}", -3.5), ("42}", 42.0), (" 0", 0.0)):
            with self.subTest(reply=reply):
                client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
                client.sglang = ReplyServer(reply)
                self.assertEqual(client.generate(context="R", questions={"a": {"type": "number"}}).result,
                                 {"a": value})


if __name__ == "__main__":
    unittest.main()
