import json
import pickle
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx

from typellm import (
    GenerationCancelled,
    GenerationTimeout,
    SGLangClient,
    SGLangError,
    TypeLLMClient,
    Usage,
)
from typellm.sglang import call_scope

from tests.test_batching import FakeServer

PICK = {"pick": {"type": "string", "enum": list("abcdef"), "permutations": 3}}


class SlowServer(FakeServer):
    """FakeServer that yields between requests so concurrent calls interleave."""

    def _request(self, path, payload=None, *, allow_text=False):
        time.sleep(0.001)
        return super()._request(path, payload, allow_text=allow_text)


class SharedClientTests(unittest.TestCase):
    def test_concurrent_calls_keep_their_own_prompts(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = SlowServer()
        finished = threading.Barrier(8)

        def run(n):
            client.generate(context=f"Receipt {n}", questions={
                "a": {"type": "integer"}, "b": {"type": "boolean"},
            }).result
            finished.wait()  # Every call has finished before any reads its prompts.
            return n, client._last_prompts.get()

        with ThreadPoolExecutor(8) as pool:
            for n, prompts in pool.map(run, range(16)):
                self.assertEqual(len(prompts), 2)
                for prompt in prompts:
                    self.assertIn(f"Receipt {n}", prompt)

    def test_a_seeded_call_is_reproducible_and_leaves_the_shared_stream(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake", seed=1)
        client.sglang = FakeServer()
        state = client.rng.getstate()
        client.generate(context="Roll", questions=PICK, seed=5).result
        first = client._last_prompts.get()
        client.generate(context="Roll", questions=PICK, seed=5).result
        self.assertEqual(client._last_prompts.get(), first)
        self.assertEqual(client.rng.getstate(), state)
        client.generate(context="Roll", questions=PICK).result
        self.assertNotEqual(client.rng.getstate(), state)

    def test_clients_pickle_after_a_call(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions={"b": {"type": "boolean"}}).result
        copy = pickle.loads(pickle.dumps(client))
        self.assertEqual(copy._last_prompts.get(), [])
        copy.generate(context="Receipt", questions={"b": {"type": "boolean"}}).result
        self.assertEqual(len(copy._last_prompts.get()), 1)


class MeteredServer(SlowServer):
    """Reports 10 prompt, 4 cached and 1 completion token per prompt."""

    def __init__(self, fail_after=None):
        super().__init__()
        self.fail_after = fail_after

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and self.fail_after is not None and len(self.payloads) >= self.fail_after:
            raise SGLangError("SGLang went away")
        response = super()._request(path, payload, allow_text=allow_text)
        if path == "/generate":
            for item in response if isinstance(response, list) else [response]:
                item.setdefault("meta_info", {}).update(
                    prompt_tokens=10, cached_tokens=4, completion_tokens=1)
        return response


def prompts_sent(server):
    return sum(1 if isinstance(p["text"], str) else len(p["text"]) for p in server.payloads)


class UsageTests(unittest.TestCase):
    QUESTIONS = {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": "string"}}

    def test_usage_sums_every_generate_request_of_the_call(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = MeteredServer()
        usage = client.generate(context="Receipt", questions=self.QUESTIONS).usage
        n = prompts_sent(client.sglang)
        # The input counts once, though every prompt carries it.
        sent = client.sglang.count_tokens("Receipt") + client.sglang.count_tokens(json.dumps(self.QUESTIONS))
        self.assertEqual(usage, Usage(
            requests=len(client.sglang.payloads),
            prompt_tokens=10 * n, cached_tokens=4 * n, completion_tokens=n, input_tokens=sent,
        ))
        self.assertGreater(usage.requests, 1)
        client.sglang.payloads.clear()
        usage = client.generate(context="Receipt", questions={"b": {"type": "boolean"}}).usage
        self.assertEqual(usage.requests, len(client.sglang.payloads))

    def test_usage_shows_billed_tokens_first_and_only_counts_it_has(self):
        self.assertEqual(repr(Usage(input_tokens=66)), "Usage(input_tokens=66, thinking_tokens=0)")
        self.assertEqual(repr(Usage(input_tokens=5, requests=2, prompt_tokens=30)),
                         "Usage(input_tokens=5, thinking_tokens=0, requests=2, prompt_tokens=30)")

    def test_concurrent_calls_count_only_their_own_requests(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = MeteredServer()
        finished = threading.Barrier(4)

        def run(questions):
            usage = client.generate(context="Receipt", questions=questions).usage
            finished.wait()
            return usage.requests

        one = {"b": {"type": "boolean"}}
        with ThreadPoolExecutor(4) as pool:
            counts = list(pool.map(run, [one, self.QUESTIONS, one, self.QUESTIONS]))
        self.assertEqual(counts[0], counts[2])
        self.assertEqual(counts[1], counts[3])
        self.assertGreater(counts[1], counts[0])
        self.assertEqual(sum(counts), len(client.sglang.payloads))

    def test_a_failed_call_still_reports_the_requests_it_made(self):
        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = MeteredServer(fail_after=2)
        with self.assertRaisesRegex(SGLangError, "went away") as caught:
            client.generate(context="Receipt", questions=self.QUESTIONS)
        self.assertEqual(caught.exception.usage.requests, 2)
        self.assertGreater(caught.exception.usage.input_tokens, 0)
        with self.assertRaises(ValueError) as caught:
            client.generate(questions=self.QUESTIONS)
        self.assertFalse(hasattr(caught.exception, "usage"))  # refused before any work


class DeadlineTests(unittest.TestCase):
    QUESTIONS = UsageTests.QUESTIONS

    def test_a_call_stops_once_its_time_runs_out(self):
        class Slower(MeteredServer):
            def _request(self, path, payload=None, *, allow_text=False):
                if path == "/generate":
                    time.sleep(0.03)
                return super()._request(path, payload, allow_text=allow_text)

        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = Slower()
        client.generate(context="Receipt", questions=self.QUESTIONS).result
        needed = len(client.sglang.payloads)
        client.sglang.payloads.clear()
        with self.assertRaises(GenerationTimeout) as caught:
            client.generate(context="Receipt", questions=self.QUESTIONS, timeout=0.05)
        self.assertLess(len(client.sglang.payloads), needed)
        self.assertEqual(caught.exception.usage.requests, len(client.sglang.payloads))

    def test_a_cancelled_call_sends_no_further_requests(self):
        cancel = threading.Event()

        class Cancelling(MeteredServer):
            def _request(self, path, payload=None, *, allow_text=False):
                response = super()._request(path, payload, allow_text=allow_text)
                if path == "/generate" and len(self.payloads) == 2:
                    cancel.set()
                return response

        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = Cancelling()
        with self.assertRaises(GenerationCancelled):
            client.generate(context="Receipt", questions=self.QUESTIONS, cancel=cancel).result
        self.assertEqual(len(client.sglang.payloads), 2)

    def test_socket_timeouts_never_outlast_the_call(self):
        seen = []

        def handler(request):
            seen.append(request.extensions["timeout"]["read"])
            return httpx.Response(200, json={"meta_info": {}})

        client = SGLangClient(timeout=120)
        client._http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client._request("/generate", {"text": "x"})
        with call_scope(timeout=5):
            client._generate({"text": "x"})
        self.assertEqual(seen[0], 120)
        self.assertLessEqual(seen[1], 5)

    def test_a_request_cut_short_by_the_deadline_is_a_timeout(self):
        def handler(request):
            time.sleep(0.05)
            raise httpx.ReadTimeout("timed out", request=request)

        client = SGLangClient()
        client._http_client = httpx.Client(transport=httpx.MockTransport(handler))
        with self.assertRaises(GenerationTimeout) as caught, call_scope(timeout=0.02):
            client._generate({"text": "x"})
        self.assertIsInstance(caught.exception.__cause__, SGLangError)
        # Without a deadline the same failure stays a plain SGLangError.
        with self.assertRaises(SGLangError) as caught:
            client._generate({"text": "x"})
        self.assertNotIsInstance(caught.exception, GenerationTimeout)

    def dropping(self, error, times):
        """A client whose server drops the first `times` requests with `error`."""
        seen = []

        def handler(request):
            seen.append(request)
            if len(seen) <= times:
                raise error("connection dropped", request=request)
            return httpx.Response(200, json={"text": "ok"})

        client = SGLangClient()
        client._http_client = httpx.Client(transport=httpx.MockTransport(handler))
        return client, seen

    def test_a_dropped_connection_is_retried_once(self):
        for error in (httpx.ReadError, httpx.RemoteProtocolError, httpx.WriteError):
            client, seen = self.dropping(error, 1)
            with self.subTest(error=error.__name__):
                self.assertEqual(client._request("/generate", {"text": "x"}), {"text": "ok"})
                self.assertEqual(len(seen), 2)
        client, seen = self.dropping(httpx.ReadError, 2)
        with self.assertRaisesRegex(SGLangError, "Could not reach"):
            client._request("/generate", {"text": "x"})
        self.assertEqual(len(seen), 2)

    def test_a_refused_connection_is_not_retried(self):
        for error in (httpx.ConnectError, httpx.ReadTimeout):
            client, seen = self.dropping(error, 1)
            with self.subTest(error=error.__name__), self.assertRaises(SGLangError):
                client._request("/generate", {"text": "x"})
            self.assertEqual(len(seen), 1)

    def test_invalid_budgets_are_rejected(self):
        class Recording(FakeServer):
            def _request(self, path, payload=None, *, allow_text=False):
                self.paths.append(path)
                return super()._request(path, payload, allow_text=allow_text)

        client = TypeLLMClient("http://127.0.0.1:30000", model="fake")
        client.sglang = Recording()
        client.sglang.paths = []
        for kwargs in ({"timeout": 0}, {"timeout": -1}, {"timeout": True}, {"timeout": "5"},
                       {"cancel": True}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                client.generate(context="Receipt", questions={"b": {"type": "boolean"}}, **kwargs).result
        self.assertEqual(client.sglang.paths, [])  # Rejected before even compiling.


class LazyLoadTests(unittest.TestCase):
    def test_the_chat_tokenizer_loads_once_across_threads(self):
        calls = []
        barrier = threading.Barrier(4)

        def from_pretrained(source, **kwargs):
            calls.append(source)
            time.sleep(0.01)
            return object()

        client = SGLangClient(model="fake", tokenizer="fake-tokenizer")

        def load(_):
            barrier.wait()
            return client._get_chat_tokenizer()

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)
        with patch.dict(sys.modules, {"transformers": transformers}):
            with ThreadPoolExecutor(4) as pool:
                loaded = set(map(id, pool.map(load, range(4))))
        self.assertEqual(calls, ["fake-tokenizer"])
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
