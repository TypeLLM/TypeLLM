"""The hosted client's result and error contract, including older service responses."""

import unittest
from dataclasses import asdict
from unittest.mock import patch

import httpx

from typellm import GenerationTimeout, SGLangError, SchemaError, TypeLLMClient, Usage


class HostedContractTests(unittest.TestCase):
    questions = {"ok": {"type": "boolean"}}

    def client(self, response):
        client = TypeLLMClient(api_key="test")
        client._transport = httpx.MockTransport(lambda request: response)
        return client

    def success(self, usage, **extra):
        return httpx.Response(200, json={
            "result": {"ok": True}, "usage": usage, **extra,
        })

    def test_unreported_counts_are_unknown_and_reported_zero_is_zero(self):
        response = self.success({"input_tokens": 12, "thinking_tokens": 0})
        done = self.client(response).generate(context="x", questions=self.questions)
        self.assertEqual(done.result, {"ok": True})
        self.assertEqual(done.thinking, {})
        self.assertEqual(asdict(done.usage), {
            "requests": None, "prompt_tokens": None, "cached_tokens": None,
            "completion_tokens": None, "input_tokens": 12, "thinking_tokens": 0,
        })

    def test_reported_internal_counts_are_preserved(self):
        counts = asdict(Usage(requests=3, prompt_tokens=100, cached_tokens=20,
                              completion_tokens=9, input_tokens=40, thinking_tokens=5))
        counts["future_counter"] = 99
        done = self.client(self.success(counts)).generate(context="x", questions=self.questions)
        self.assertEqual(asdict(done.usage), {k: v for k, v in counts.items() if k != "future_counter"})

    def test_invalid_success_usage_is_rejected(self):
        for counts in (None, [], {}, {"input_tokens": 1},
                       {"input_tokens": True, "thinking_tokens": 0},
                       {"input_tokens": 1, "thinking_tokens": -1},
                       {"input_tokens": 1, "thinking_tokens": 0, "requests": "2"}):
            with self.subTest(counts=counts), self.assertRaises(SGLangError) as caught:
                self.client(self.success(counts)).generate(context="x", questions=self.questions)
            self.assertEqual(caught.exception.status, 200)

    def test_failure_preserves_partial_usage(self):
        for status, expected in ((400, SGLangError), (502, SGLangError), (504, GenerationTimeout)):
            response = httpx.Response(status, json={
                "error": {"type": "timeout" if status == 504 else "upstream_error", "message": "failed"},
                "usage": {"input_tokens": 30, "thinking_tokens": 7},
            })
            with self.subTest(status=status), self.assertRaises(expected) as caught:
                self.client(response).generate(context="x", questions=self.questions)
            self.assertEqual(caught.exception.status, status)
            self.assertEqual(caught.exception.usage.input_tokens, 30)
            self.assertEqual(caught.exception.usage.thinking_tokens, 7)
            self.assertIsNone(caught.exception.usage.requests)

    def test_failure_without_usage_does_not_claim_zero_cost(self):
        responses = [httpx.Response(429, text="busy"), httpx.Response(429, json=[])]
        for usage in (None, "unknown", {"input_tokens": -1}, {"input_tokens": True}):
            responses.append(httpx.Response(429, json={"error": {"message": "busy"}, "usage": usage}))
        for response in responses:
            with self.subTest(body=response.text), self.assertRaises(SGLangError) as caught:
                self.client(response).generate(context="x", questions=self.questions)
            self.assertEqual(caught.exception.status, 429)
            self.assertIsNone(caught.exception.usage)

    def test_transport_failure_has_unknown_usage(self):
        def fail(request):
            raise httpx.ReadError("connection lost", request=request)
        client = TypeLLMClient(api_key="test")
        client._transport = httpx.MockTransport(fail)
        with self.assertRaises(SGLangError) as caught:
            client.generate(context="x", questions=self.questions)
        self.assertIsNone(caught.exception.usage)

    def test_accepted_remote_schema_is_not_recompiled(self):
        # A newer service can accept schema keywords/types that this client does not know.
        questions = {"ok": {"type": "a_future_type"}}
        done = self.client(self.success({"input_tokens": 1, "thinking_tokens": 0})).generate(
            context="x", questions=questions)
        self.assertTrue(done.result["ok"])

    def test_rejected_schema_uses_the_same_exception_as_local(self):
        for field in ({"type": "integer", "enum": ["wrong"]},
                      {"type": "string", "enum": ["duplicate", "duplicate"]},
                      {"type": "an_unsupported_type"}):
            questions = {"q": field}
            with self.subTest(field=field):
                try:
                    TypeLLMClient().generate(context="x", questions=questions)
                except (SchemaError, NotImplementedError) as local_error:
                    expected_type, expected_message = type(local_error), str(local_error)
                else:
                    self.fail("fixture should be rejected before tokenization")
                response = httpx.Response(400, json={
                    "error": {"type": "invalid_request", "message": expected_message},
                    "usage": {"input_tokens": 0, "thinking_tokens": 0},
                })
                with self.assertRaises(expected_type) as caught:
                    self.client(response).generate(context="x", questions=questions)
                self.assertEqual(str(caught.exception), expected_message)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.usage.input_tokens, 0)
                self.assertIsInstance(caught.exception.__cause__, SGLangError)

    def test_service_errors_are_not_misclassified_as_schema_errors(self):
        invalid = {"q": {"type": "integer", "enum": ["wrong"]}}
        for status, kind, questions in ((400, "limit_exceeded", invalid),
                                        (429, "invalid_request", invalid),
                                        (400, "invalid_request", self.questions),
                                        (400, None, invalid)):
            response = httpx.Response(status, json={"error": {"type": kind, "message": "refused"}})
            with self.subTest(status=status, kind=kind), self.assertRaises(SGLangError) as caught:
                self.client(response).generate(context="x", questions=questions)
            self.assertEqual(caught.exception.status, status)
            self.assertIsNone(caught.exception.usage)

    def test_schema_diagnostic_failure_preserves_the_api_error(self):
        response = httpx.Response(400, json={"error": {"type": "invalid_request", "message": "refused"}})
        with patch("typellm.runtime.compile_json_schema", side_effect=TypeError("unsupported shape")):
            with self.assertRaises(SGLangError) as caught:
                self.client(response).generate(context="x", questions=self.questions)
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("refused", str(caught.exception))

    def test_success_does_not_compile_schema_locally(self):
        response = self.success({"input_tokens": 1, "thinking_tokens": 0})
        with patch("typellm.runtime.compile_json_schema", side_effect=AssertionError("must not compile")):
            done = self.client(response).generate(context="x", questions=self.questions)
        self.assertTrue(done.result["ok"])

    def test_malformed_success_keeps_reported_usage(self):
        response = self.success({"input_tokens": 10, "thinking_tokens": 2},
                                result={"ok": {"value": True, "probabilities": []}})
        with self.assertRaises(SGLangError) as caught:
            self.client(response).generate(context="x", questions=self.questions)
        self.assertEqual(caught.exception.usage.input_tokens, 10)
        self.assertEqual(caught.exception.status, 200)


if __name__ == "__main__":
    unittest.main()
