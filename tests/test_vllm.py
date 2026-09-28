"""Unit tests for the vLLM transport adapter."""

from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from typellm import (
    GenerationCancelled,
    TypeLLMClient,
    VLLMClient,
    VLLMError,
)

from tests.test_batching import ThinkingTokenizer, is_number_pattern
from tests.test_images import PNG, VISION, fake_tokenize, fake_detokenize


THINK_STOP = "</think>"
IMAGE_PAD_ID = 900001


class VLLMVisionTokenizer(ThinkingTokenizer):
    """Encodes the vision placeholder and think close as unique token ids."""

    def encode(self, text, *, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text.startswith(VISION, i):
                ids.append(IMAGE_PAD_ID)
                i += len(VISION)
            elif text.startswith(THINK_STOP, i):
                ids.append(900002)
                i += len(THINK_STOP)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, **kwargs):
        out = []
        for token_id in ids:
            if token_id == IMAGE_PAD_ID:
                out.append(VISION)
            elif token_id == 900002:
                out.append(THINK_STOP)
            elif 0 <= token_id < 128:
                out.append(chr(token_id))
            else:
                out.append("?")
        return "".join(out)


class FakeVLLM(VLLMClient):
    """A real VLLMClient whose HTTP layer answers like vLLM 0.28."""

    def __init__(self, prefix_cache_block_tokens: int | None = None):
        super().__init__(
            model="fake-vllm",
            prefix_cache_block_tokens=prefix_cache_block_tokens,
        )
        self._chat_tokenizer = VLLMVisionTokenizer()
        self._context_length_cache = 100_000
        self._capabilities_checked = True  # skip live probes in unit tests
        self._list_prompt_scoring = True
        self._list_prompt_scoring_checked = True
        self.payloads = []
        self.paths = []
        self.cache_prefix_calls = []

    def cache_prefix(self, prefix: str):
        self.cache_prefix_calls.append(prefix)
        return super().cache_prefix(prefix)

    def _request(self, path, payload=None, *, allow_text=False):
        self.paths.append(path)
        self.payloads.append({"path": path, "payload": payload})
        if path == "/v1/models":
            return {
                "data": [{
                    "id": "fake-vllm",
                    "root": "fake-vllm",
                    "max_model_len": 100_000,
                }]
            }
        if path == "/tokenize":
            return {"tokens": fake_tokenize(payload["prompt"]), "count": 1}
        if path == "/detokenize":
            return {"text": fake_detokenize(payload["tokens"])}
        if path == "/reset_prefix_cache":
            raise VLLMError("not found", status=404)
        if path == "/v1/chat/completions/render":
            return self._fake_render(payload)
        if path == "/inference/v1/generate":
            return self._fake_generate(payload)
        if path == "/v1/completions":
            return self._fake_completion(payload)
        raise AssertionError(f"unexpected path {path}")

    def _fake_render(self, payload):
        messages = payload["messages"]
        content = messages[0]["content"]
        n_images = sum(1 for part in content if part.get("type") == "image_url")
        token_ids = []
        spans = []
        for _ in range(n_images):
            spans.append({"offset": len(token_ids), "length": 4})
            token_ids.extend([IMAGE_PAD_ID] * 4)
        token_ids.extend(list(b"TYPELLM_IMAGE_PROBE"))
        return {
            "token_ids": token_ids,
            "features": {
                "mm_hashes": {"image": [f"hash{i}" for i in range(n_images)]},
                "mm_placeholders": {"image": spans},
                "kwargs_data": {"image": [{} for _ in range(n_images)]},
            },
        }

    def _fake_generate(self, payload):
        sampling = payload["sampling_params"]
        token_ids = payload["token_ids"]
        pad_runs = 0
        i = 0
        while i < len(token_ids):
            if token_ids[i] == IMAGE_PAD_ID:
                run = 0
                while i < len(token_ids) and token_ids[i] == IMAGE_PAD_ID:
                    run += 1
                    i += 1
                assert run == 4, run
                pad_runs += 1
            else:
                i += 1
        assert pad_runs >= 1
        if sampling.get("logprob_token_ids"):
            ids = list(sampling["logprob_token_ids"])
            pick = ord(" ") if ord(" ") in ids else (
                1 if 1 in ids else (ord("7") if ord("7") in ids else ids[0])
            )
            tops = [
                {"token": f"token_id:{tid}", "logprob": 0.0 if tid == pick else -9.0}
                for tid in ids
            ]
            return {
                "choices": [{
                    "token_ids": [pick],
                    "finish_reason": "length",
                    "logprobs": {
                        "content": [{
                            "token": f"token_id:{pick}",
                            "logprob": 0.0,
                            "top_logprobs": tops,
                        }]
                    },
                }],
                "usage": {
                    "prompt_tokens": len(token_ids) + 64,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        stop_ids = sampling.get("stop_token_ids") or []
        if stop_ids:
            close = stop_ids[0]
            body = list(b"Reasoned.") + [close]
            return {
                "choices": [{
                    "token_ids": body,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(token_ids) + 64,
                    "completion_tokens": len(body),
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        regex = (sampling.get("structured_outputs") or {}).get("regex", "")
        if is_number_pattern(regex):
            out = list(b" 7}")
        elif "null" in regex:
            out = list(b" null}")
        elif regex.startswith(" ?\""):
            out = list(b' "blue"}')
        else:
            out = list(b"ok")
        return {
            "choices": [{
                "token_ids": out,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(token_ids) + 64,
                "completion_tokens": len(out),
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    def _fake_completion(self, payload):
        prompt = payload["prompt"]
        # List-prompt scoring returns one choice per prompt.
        if isinstance(prompt, list):
            choices = []
            total_prompt = 0
            for item in prompt:
                one = self._fake_completion({**payload, "prompt": item})
                choices.append(one["choices"][0])
                total_prompt += one["usage"]["prompt_tokens"]
            return {
                "choices": choices,
                "usage": {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": len(choices),
                    "prompt_tokens_details": {
                        "cached_tokens": (
                            784 if any(len(p) > 3000 for p in prompt) else 0
                        ),
                    },
                },
            }
        if payload.get("logprob_token_ids"):
            ids = list(payload["logprob_token_ids"])
            pick = ord(" ") if ord(" ") in ids else (
                1 if 1 in ids else (ord("7") if ord("7") in ids else ids[0])
            )
            top = {f"token_id:{tid}": (0.0 if tid == pick else -9.0) for tid in ids}
            return {
                "choices": [{
                    "text": chr(pick) if pick < 128 else "?",
                    "finish_reason": "length",
                    "stop_reason": None,
                    "logprobs": {
                        "text_offset": [0],
                        "token_logprobs": [0.0],
                        "tokens": [f"token_id:{pick}"],
                        "top_logprobs": [top],
                    },
                }],
                "usage": {
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        stop = payload.get("stop")
        if stop == [THINK_STOP] or (isinstance(stop, list) and THINK_STOP in stop):
            return {
                "choices": [{
                    "text": "Reasoned." + THINK_STOP,
                    "finish_reason": "stop",
                    "stop_reason": THINK_STOP,
                }],
                "usage": {
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        structured = payload.get("structured_outputs") or {}
        regex = structured.get("regex", "")
        if regex == "[xyz]{3}":
            text = "xyz"
        elif is_number_pattern(regex):
            text = " 7}"
        elif "|null)" in regex or "null|" in regex:
            text = " null}"
        elif regex.startswith(" ?\""):
            text = ' "blue"}'
        elif structured.get("json"):
            schema = structured["json"]
            if schema.get("type") == "object":
                key = next(iter(schema["properties"]))
                text = json.dumps({key: "blue"})
            else:
                text = json.dumps("blue")
        elif payload.get("max_tokens") == 1 and not regex and not structured:
            text = "x"
        else:
            text = "ok"
        return {
            "choices": [{
                "text": text,
                "finish_reason": "stop",
                "stop_reason": None,
            }],
            "usage": {
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": max(1, len(text) // 2),
                "prompt_tokens_details": {
                    "cached_tokens": 784 if len(prompt) > 3000 else 0,
                },
            },
        }


def client_with_fake(*, prefix_cache_block_tokens: int | None = None):
    client = TypeLLMClient(
        backend="vllm",
        model="fake-vllm",
        prefix_cache_block_tokens=prefix_cache_block_tokens,
    )
    client.sglang = FakeVLLM(prefix_cache_block_tokens=prefix_cache_block_tokens)
    return client


class BackendWiringTests(unittest.TestCase):
    def test_backend_defaults_to_sglang(self):
        client = TypeLLMClient(model="x")
        self.assertEqual(client.backend, "sglang")
        self.assertEqual(client.sglang._server_name, "SGLang")

    def test_backend_vllm_builds_vllm_client(self):
        client = TypeLLMClient(backend="vllm", model="x")
        self.assertIsInstance(client.sglang, VLLMClient)
        self.assertEqual(client.sglang._server_name, "vLLM")
        self.assertEqual(client.sglang.base_url, "http://127.0.0.1:8000")

    def test_vllm_url_strips_v1_suffix(self):
        client = TypeLLMClient(
            "http://example:8000/v1", backend="vllm", model="x"
        )
        self.assertEqual(client.sglang.base_url, "http://example:8000")

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            TypeLLMClient(backend="llama")

    def test_api_key_rejects_vllm_backend(self):
        with self.assertRaises(ValueError):
            TypeLLMClient(api_key="tl_sk_x", backend="vllm")

    def test_prefix_cache_block_tokens_override(self):
        client = TypeLLMClient(
            backend="vllm", model="x", prefix_cache_block_tokens=784
        )
        self.assertEqual(client.sglang.prefix_cache_block_tokens, 784)


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeVLLM()

    def test_sampling_param_mapping(self):
        body = self.server._to_completion_body(
            "p",
            {
                "max_new_tokens": 8,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": -1,
                "min_p": 0.0,
                "sampling_seed": 7,
                "stop": [THINK_STOP],
                "no_stop_trim": True,
                "regex": " ?[0-9]+\\}?",
            },
            None,
        )
        self.assertEqual(body["max_tokens"], 8)
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["top_k"], 0)
        self.assertEqual(body["seed"], 7)
        self.assertTrue(body["include_stop_str_in_output"])
        self.assertEqual(body["structured_outputs"]["regex"], " ?[0-9]+\\}?")

    def test_max_new_tokens_zero_becomes_discard(self):
        body = self.server._to_completion_body("p", {"max_new_tokens": 0}, None)
        self.assertEqual(body["max_tokens"], 1)
        self.assertTrue(body["_typellm_discard_output"])

    def test_logprob_extraction(self):
        choice = {
            "logprobs": {
                "top_logprobs": [{
                    "token_id:32": -0.1,
                    "token_id:33": -2.0,
                    "token_id:99": -5.0,
                }]
            }
        }
        rows = self.server._extract_top_logprobs(choice, [32, 33])
        self.assertEqual(rows, [[-0.1, 32, None], [-2.0, 33, None]])

    def test_missing_candidate_logprob_raises(self):
        choice = {"logprobs": {"top_logprobs": [{"token_id:32": -0.1}]}}
        with self.assertRaises(VLLMError):
            self.server._extract_top_logprobs(choice, [32, 33])

    def test_finish_reason_and_usage_mapping(self):
        response = {
            "choices": [{
                "text": "A",
                "finish_reason": "stop",
                "stop_reason": None,
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 784},
            },
        }
        mapped = self.server._completion_to_sglang(response, None)
        self.assertEqual(mapped["text"], "A")
        self.assertEqual(mapped["meta_info"]["cached_tokens"], 784)
        self.assertEqual(mapped["meta_info"]["finish_reason"]["type"], "stop")

    def test_cache_prefix_discards_output_token(self):
        meta = self.server.cache_prefix("hello world")
        self.assertEqual(meta["completion_tokens"], 0)
        self.assertIn("/v1/completions", self.server.paths)

    def test_flush_cache_explains_dev_mode(self):
        with self.assertRaises(VLLMError) as ctx:
            self.server.flush_cache()
        self.assertIn("VLLM_SERVER_DEV_MODE", str(ctx.exception))

    def test_batch_preserves_prompt_order(self):
        prompts = [f"prompt-{i} " + ("x" * i) for i in range(8)]
        ids = [[ord("A"), ord("B")] for _ in prompts]
        scored, _ = self.server.score_candidates_batch(prompts, ids)
        self.assertEqual(len(scored), 8)
        for scores, _meta in scored:
            self.assertEqual(set(scores), {ord("A"), ord("B")})


class EndToEndVLLMTests(unittest.TestCase):
    RECEIPT = {
        "merchant": {"type": "string", "instructions": "Merchant name."},
        "total": {"type": "number", "instructions": "Total."},
        "expense_type": {
            "type": "string",
            "enum": ["meal", "travel", "equipment"],
            "instructions": "Expense type.",
        },
        "reimbursable": {"type": "boolean", "instructions": "Reimburse?"},
        "confidence": {
            "type": "number",
            "enum": [0.0, 0.25, 0.5, 0.75, 1.0],
            "instructions": "Confidence.",
        },
    }

    def test_all_output_types(self):
        client = client_with_fake()
        result = client.generate(
            context="Receipt from Hilton. Total 324.50. Travel.",
            questions=self.RECEIPT,
        ).result
        self.assertIsInstance(result["merchant"], str)
        self.assertIsInstance(result["total"], float)
        self.assertIn(result["expense_type"], {"meal", "travel", "equipment"})
        self.assertIsInstance(result["reimbursable"], bool)
        self.assertIn(result["confidence"], {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_return_probabilities(self):
        client = client_with_fake()
        result = client.generate(
            context="Travel expense.",
            questions={
                "expense_type": {
                    "type": "string",
                    "enum": ["meal", "travel", "equipment"],
                    "return_probabilities": True,
                }
            },
        ).result
        probs = result["expense_type"]["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertEqual(set(probs), {"meal", "travel", "equipment"})

    def test_nullable_fields(self):
        client = client_with_fake()
        result = client.generate(
            context="No tip on this receipt.",
            questions={
                "tip": {"type": ["number", "null"], "instructions": "Tip."},
                "paid_in_cash": {"type": ["boolean", "null"], "instructions": "Cash?"},
            },
        ).result
        self.assertIn("tip", result)
        self.assertIn("paid_in_cash", result)

    def test_depends_on_layers(self):
        client = client_with_fake()
        result = client.generate(
            context="Payments service errors after a deployment.",
            questions={
                "system": {
                    "type": "string",
                    "enum": ["payments", "accounts", "search"],
                    "instructions": "Which system?",
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "depends_on": ["system"],
                    "instructions": "Severity?",
                },
                "rollback": {
                    "type": "boolean",
                    "depends_on": ["severity"],
                    "instructions": "Rollback?",
                },
            },
        ).result
        self.assertIn(result["system"], {"payments", "accounts", "search"})
        self.assertIn(result["severity"], {"low", "medium", "high"})
        self.assertIsInstance(result["rollback"], bool)

    def test_thinking_returns_reasoning(self):
        client = client_with_fake()
        response = client.generate(
            context="Policy check.",
            questions={
                "ok": {
                    "type": "boolean",
                    "instructions": "Policy ok?",
                    "thinking": True,
                    "thinking_budget": 64,
                }
            },
        )
        self.assertIsInstance(response.result["ok"], bool)
        self.assertIn("ok", response.thinking)
        self.assertTrue(response.thinking["ok"])

    def test_permutations_auto_runs_multiple_orderings(self):
        client = client_with_fake()
        server = client.sglang
        result = client.generate(
            context="Fair die.",
            questions={
                "roll": {
                    "type": "string",
                    "enum": ["one", "two", "three", "four", "five", "six"],
                    "permutations": "auto",
                    "return_probabilities": True,
                    "instructions": "Roll?",
                }
            },
        ).result
        score_calls = [
            p for p in server.payloads
            if p["path"] == "/v1/completions"
            and (p["payload"] or {}).get("logprob_token_ids")
        ]
        # Merged list-prompt scoring issues one call with six prompts; the
        # fan-out path issues six single-prompt calls.
        prompt_count = 0
        for call in score_calls:
            prompt = (call["payload"] or {}).get("prompt")
            prompt_count += len(prompt) if isinstance(prompt, list) else 1
        self.assertEqual(prompt_count, 6)
        self.assertAlmostEqual(sum(result["roll"]["probabilities"].values()), 1.0, places=5)

    def test_free_text_max_length(self):
        client = client_with_fake()
        result = client.generate(
            context="Summarize.",
            questions={
                "summary": {
                    "type": "string",
                    "maxLength": 20,
                    "instructions": "One short sentence.",
                }
            },
        ).result
        self.assertIsInstance(result["summary"], str)
        self.assertLessEqual(len(result["summary"]), 20)

    def test_images_end_to_end(self):
        client = client_with_fake()
        result = client.generate(
            context="Read the receipt photo.",
            images=[PNG],
            questions={
                "total": {"type": "number", "instructions": "Total."},
                "paid": {"type": "boolean", "instructions": "Paid?"},
            },
        ).result
        self.assertIsInstance(result["total"], float)
        self.assertIsInstance(result["paid"], bool)
        self.assertIn("/v1/chat/completions/render", client.sglang.paths)
        self.assertIn("/inference/v1/generate", client.sglang.paths)
        gen = next(
            p for p in client.sglang.payloads if p["path"] == "/inference/v1/generate"
        )
        spans = gen["payload"]["features"]["mm_placeholders"]["image"]
        self.assertEqual(spans[0]["length"], 4)
        self.assertEqual(gen["payload"]["token_ids"].count(IMAGE_PAD_ID), 4)

    def test_shared_client_concurrency(self):
        client = client_with_fake()

        def once(_):
            return client.generate(
                context="Travel.",
                questions={"ok": {"type": "boolean"}},
            ).result["ok"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            outs = list(pool.map(once, range(8)))
        self.assertEqual(len(outs), 8)
        self.assertTrue(all(isinstance(v, bool) for v in outs))

    def test_cancel_raises(self):
        client = client_with_fake()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(GenerationCancelled):
            client.generate(
                context="x",
                questions={"ok": {"type": "boolean"}},
                cancel=cancel,
            )

    def test_usage_populated(self):
        client = client_with_fake()
        response = client.generate(
            context="Receipt.",
            questions={"ok": {"type": "boolean"}},
        )
        self.assertGreaterEqual(response.usage.requests, 1)
        self.assertGreaterEqual(response.usage.prompt_tokens, 1)


class CapabilityProbeTests(unittest.TestCase):
    def test_probe_logprobs_failure(self):
        server = FakeVLLM()
        server._capabilities_checked = False

        def bad_completion(payload):
            return {
                "choices": [{
                    "text": "A",
                    "finish_reason": "length",
                    "logprobs": {
                        "top_logprobs": [{"token_id:32": -0.1}],
                    },
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }

        server._fake_completion = bad_completion  # type: ignore[method-assign]
        with self.assertRaises(VLLMError):
            server._probe_logprob_token_ids()


class LatencyOptimizationTests(unittest.TestCase):
    RECEIPT = {
        "merchant": {"type": "string", "instructions": "Merchant name."},
        "total": {"type": "number", "instructions": "Total."},
        "expense_type": {
            "type": "string",
            "enum": ["meal", "travel", "equipment"],
            "instructions": "Expense type.",
        },
        "reimbursable": {"type": "boolean", "instructions": "Reimburse?"},
    }

    def test_warmup_skipped_below_block_size(self):
        client = client_with_fake(prefix_cache_block_tokens=784)
        client.generate(
            context="Short receipt.",
            questions={"ok": {"type": "boolean", "instructions": "Ok?"}},
        )
        self.assertEqual(client.sglang.cache_prefix_calls, [])

    def test_warmup_runs_above_block_size(self):
        client = client_with_fake(prefix_cache_block_tokens=8)
        client.generate(
            context="A short receipt with enough tokens for a cache block.",
            questions={"ok": {"type": "boolean", "instructions": "Ok?"}},
        )
        self.assertGreaterEqual(len(client.sglang.cache_prefix_calls), 1)

    def test_merged_scoring_matches_fanout(self):
        prompts = [f"prompt-{i} " for i in range(4)]
        ids = [[ord("A"), ord("B")] for _ in prompts]
        merged = FakeVLLM()
        merged._list_prompt_scoring = True
        fanout = FakeVLLM()
        fanout._list_prompt_scoring = False
        merged_scores, _ = merged.score_candidates_batch(prompts, ids)
        fanout_scores, _ = fanout.score_candidates_batch(prompts, ids)
        self.assertEqual(len(merged_scores), len(fanout_scores))
        for (a, _), (b, _) in zip(merged_scores, fanout_scores):
            self.assertEqual(a, b)
        # Merged path should issue one list-prompt completions call.
        list_calls = [
            p for p in merged.payloads
            if p["path"] == "/v1/completions"
            and isinstance((p["payload"] or {}).get("prompt"), list)
        ]
        self.assertEqual(len(list_calls), 1)

    def test_open_and_score_overlap(self):
        import time

        client = client_with_fake(prefix_cache_block_tokens=784)
        server = client.sglang
        marks = []

        original_fields = server.generate_fields
        original_score = server.score_candidates_batch

        def slow_fields(*args, **kwargs):
            marks.append(("open_start", time.perf_counter()))
            time.sleep(0.05)
            result = original_fields(*args, **kwargs)
            marks.append(("open_end", time.perf_counter()))
            return result

        def slow_score(*args, **kwargs):
            marks.append(("score_start", time.perf_counter()))
            time.sleep(0.05)
            result = original_score(*args, **kwargs)
            marks.append(("score_end", time.perf_counter()))
            return result

        server.generate_fields = slow_fields  # type: ignore[method-assign]
        server.score_candidates_batch = slow_score  # type: ignore[method-assign]
        client.generate(context="Receipt from Hilton.", questions=self.RECEIPT)
        by_name = dict(marks)
        self.assertIn("open_start", by_name)
        self.assertIn("score_start", by_name)
        # Both started before either finished => overlapped.
        self.assertLess(by_name["score_start"], by_name["open_end"])
        self.assertLess(by_name["open_start"], by_name["score_end"])

    def test_results_identical_with_and_without_list_scoring(self):
        with_list = client_with_fake(prefix_cache_block_tokens=784)
        with_list.sglang._list_prompt_scoring = True
        without = client_with_fake(prefix_cache_block_tokens=784)
        without.sglang._list_prompt_scoring = False
        a = with_list.generate(context="Receipt.", questions=self.RECEIPT).result
        b = without.generate(context="Receipt.", questions=self.RECEIPT).result
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()