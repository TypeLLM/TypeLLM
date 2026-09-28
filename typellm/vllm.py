"""vLLM OpenAI-compatible transport for TypeLLM.

``VLLMClient`` subclasses ``SGLangClient`` and replaces only the HTTP seam:
chat rendering, thinking, scoring and grammar validation stay shared. Text-only
calls use ``/v1/completions``; image calls use ``/v1/chat/completions/render``
plus ``/inference/v1/generate``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any, Mapping, Sequence

from .protocol import detect_protocol
from .sglang import (
    GenerationTimeout,
    SGLangClient,
    SGLangError,
    _call_scope,
    extract_candidate_logprobs,
)

LOG = logging.getLogger("typellm")

_TOKEN_ID_KEY = re.compile(r"^token_id:(\d+)$")


class VLLMError(SGLangError):
    """A vLLM transport failure; subclasses SGLangError for shared handling."""


class VLLMClient(SGLangClient):
    """TypeLLM backend that speaks vLLM's OpenAI-compatible HTTP API."""

    _server_name = "vLLM"
    _POOL_WORKERS = 32

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model: str | None = None,
        timeout: float = 120.0,
        tokenizer: str | None = None,
        *,
        thinking_budget: int | None = None,
        text_max_tokens: int = 128,
        answer_reserve_tokens: int = 64,
        prefix_cache_block_tokens: int | None = None,
    ) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        super().__init__(
            root,
            model,
            timeout,
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            text_max_tokens=text_max_tokens,
            answer_reserve_tokens=answer_reserve_tokens,
        )
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._capabilities_checked = False
        self._capabilities_lock = threading.Lock()
        self._image_render_cache: dict[tuple[str, ...], Mapping[str, Any]] | None = None
        self._models_cache: Mapping[str, Any] | None = None
        # None: probe once; an override skips the network probe.
        if prefix_cache_block_tokens is not None:
            if type(prefix_cache_block_tokens) is not int or prefix_cache_block_tokens <= 0:
                raise ValueError("prefix_cache_block_tokens must be a positive integer")
            self._prefix_cache_block_tokens = prefix_cache_block_tokens
            self._prefix_cache_block_probed = True
        else:
            self._prefix_cache_block_tokens = 1
            self._prefix_cache_block_probed = False
        self._list_prompt_scoring = False
        self._list_prompt_scoring_checked = False

    def close(self) -> None:
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False)
        super().close()

    def _executor(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self._POOL_WORKERS)
            return self._pool

    def warmup(self) -> None:
        super().warmup()
        self._ensure_capabilities()

    # ------------------------------------------------------------------ models

    def _models(self) -> Mapping[str, Any]:
        with self._load_lock:
            if self._models_cache is None:
                response = self._request("/v1/models")
                if not isinstance(response, Mapping):
                    raise VLLMError("/v1/models returned a non-object response")
                self._models_cache = response
            return self._models_cache

    def _served_models(self) -> list[Mapping[str, Any]]:
        data = self._models().get("data", [])
        if not isinstance(data, list):
            raise VLLMError("/v1/models returned no model list")
        return [item for item in data if isinstance(item, Mapping)]

    def _model_info(self) -> Mapping[str, Any]:
        models = self._served_models()
        if not models:
            raise VLLMError("vLLM serves no models; pass model=...")
        if self.model:
            for item in models:
                if item.get("id") == self.model or item.get("root") == self.model:
                    return item
            raise VLLMError(
                f"Model {self.model!r} is not among the served ids "
                f"{[m.get('id') for m in models]}"
            )
        if len(models) == 1:
            return models[0]
        raise VLLMError(
            "vLLM serves multiple models; pass model=... to choose one: "
            f"{[m.get('id') for m in models]}"
        )

    def _tokenizer_model(self) -> str:
        if self.model:
            return self.model
        info = self._model_info()
        for key in ("id", "root"):
            value = info.get(key)
            if isinstance(value, str) and value:
                self.model = value
                return value
        raise VLLMError("Could not discover a served model id; pass model=...")

    def _tokenizer_source(self) -> str:
        if self.tokenizer:
            return self.tokenizer
        info = self._model_info()
        for key in ("root", "id"):
            value = info.get(key)
            if isinstance(value, str) and value:
                return value
        if self.model:
            return self.model
        raise VLLMError("Could not discover the tokenizer used by vLLM; pass tokenizer=...")

    def _context_length(self) -> int:
        if self._context_length_cache is None:
            info = self._model_info()
            limit = info.get("max_model_len")
            if type(limit) is not int or limit <= 0:
                raise VLLMError(
                    "Could not discover the served context length from /v1/models"
                )
            self._context_length_cache = limit
        return self._context_length_cache

    def single_token(self, label: str) -> tuple[int, str]:
        if label in self._label_tokens:
            return self._label_tokens[label]
        model = self._tokenizer_model()
        tokenized = self._request(
            "/tokenize",
            {"model": model, "prompt": label, "add_special_tokens": False},
        )
        token_ids = tokenized.get("tokens") if isinstance(tokenized, Mapping) else None
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            raise ValueError(
                f"Candidate label {label!r} must encode to exactly one token; "
                f"vLLM returned token IDs {token_ids!r}"
            )
        token_id = int(token_ids[0])
        detokenized = self._request(
            "/detokenize", {"model": model, "tokens": [token_id]}
        )
        token_text = None
        if isinstance(detokenized, Mapping):
            token_text = detokenized.get("text", detokenized.get("prompt"))
        if token_text != label:
            raise ValueError(
                f"Candidate {label!r} tokenizes to ID {token_id}, but that token "
                f"decodes as {token_text!r}; exact append would be ambiguous"
            )
        result = (token_id, label)
        self._label_tokens[label] = result
        return result

    def flush_cache(self) -> None:
        try:
            self._request("/reset_prefix_cache", {})
        except SGLangError as exc:
            if exc.status == 404:
                raise VLLMError(
                    "vLLM refused to flush its prefix cache: POST /reset_prefix_cache "
                    "requires VLLM_SERVER_DEV_MODE=1 on the server"
                ) from exc
            raise

    # ---------------------------------------------------------- capabilities

    @property
    def prefix_cache_block_tokens(self) -> int:
        if not self._prefix_cache_block_probed:
            self._ensure_capabilities()
        return self._prefix_cache_block_tokens

    def _ensure_capabilities(self) -> None:
        with self._capabilities_lock:
            if self._capabilities_checked:
                return
            self._probe_logprob_token_ids()
            self._probe_structured_outputs()
            if not self._prefix_cache_block_probed:
                self._probe_prefix_cache_block()
            self._probe_list_prompt_scoring()
            self._capabilities_checked = True

    def _probe_logprob_token_ids(self) -> None:
        ids = [self.single_token("A")[0], self.single_token("B")[0]]
        response = self._request(
            "/v1/completions",
            {
                "model": self._tokenizer_model(),
                "prompt": "Answer A or B: ",
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 1,
                "logprob_token_ids": ids,
                "return_tokens_as_token_ids": True,
            },
        )
        mapped = self._completion_to_sglang(response, ids)
        found = {
            entry[1]
            for entry in mapped["meta_info"]["output_token_ids_logprobs"][0]
        }
        if not set(ids).issubset(found):
            raise VLLMError(
                "vLLM did not return logprobs for every requested logprob_token_ids "
                f"entry; wanted={ids}, found={sorted(found)}. "
                "TypeLLM needs this field for enum/boolean scoring."
            )

    def _probe_structured_outputs(self) -> None:
        # Constraints are held until the reasoning parser sees a closed think
        # block. Render a non-thinking assistant turn so </think> is present.
        prompt = self.render_chat(
            [{"role": "user", "content": "Write exactly three letters."}],
            add_generation_prompt=True,
            thinking=False,
        )
        response = self._request(
            "/v1/completions",
            {
                "model": self._tokenizer_model(),
                "prompt": prompt,
                "max_tokens": 8,
                "temperature": 0,
                "structured_outputs": {"regex": "[xyz]{3}"},
            },
        )
        text = self._completion_text(response)
        if not re.fullmatch(r"[xyz]{3}", text.strip()):
            raise VLLMError(
                "vLLM ignored structured_outputs.regex on a closed-thinking prompt "
                f"(got {text!r}). Check that a structured-output backend is enabled "
                "and that --reasoning-parser is compatible with constrained decoding."
            )

    def _probe_prefix_cache_block(self) -> None:
        """Infer the prefix-cache block size from two cache-hit measurements."""
        model = self._tokenizer_model()
        tokenizer = self._get_chat_tokenizer()
        cached_counts: list[int] = []
        # Build prompts by token count: repeated characters compress heavily
        # under BPE, so a character length is not a token length.
        for target in (900, 1800):
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,;!?"
            # Grow in chunks; BPE compresses repeated characters.
            prompt = ""
            while len(tokenizer.encode(prompt, add_special_tokens=False)) < target:
                start = len(prompt)
                prompt += "".join(
                    alphabet[(start + i) % len(alphabet)] for i in range(256)
                )
            body = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
            }
            self._request("/v1/completions", body)
            second = self._request("/v1/completions", body)
            usage = second.get("usage") if isinstance(second, Mapping) else None
            details = (
                usage.get("prompt_tokens_details")
                if isinstance(usage, Mapping)
                else None
            )
            cached = (
                details.get("cached_tokens")
                if isinstance(details, Mapping)
                else None
            )
            if type(cached) is int and cached > 0:
                cached_counts.append(cached)
        if len(cached_counts) >= 2:
            block = math.gcd(cached_counts[0], cached_counts[1])
            if block > 0:
                self._prefix_cache_block_tokens = block
                self._prefix_cache_block_probed = True
                LOG.info("vllm_prefix_cache_block_tokens=%s", block)
                return
        if cached_counts:
            block = cached_counts[0]
            if block > 0:
                self._prefix_cache_block_tokens = block
                self._prefix_cache_block_probed = True
                LOG.info("vllm_prefix_cache_block_tokens=%s (single probe)", block)
                return
        # Prefix caching may be off; keep the conservative default of 1.
        self._prefix_cache_block_probed = True
        LOG.info("vllm_prefix_cache_block_tokens=1 (no cache hits during probe)")

    def _probe_list_prompt_scoring(self) -> None:
        """Check whether /v1/completions accepts a list of prompts for scoring."""
        ids = [self.single_token("A")[0], self.single_token("B")[0]]
        try:
            response = self._request(
                "/v1/completions",
                {
                    "model": self._tokenizer_model(),
                    "prompt": ["Answer A or B: ", "Pick A or B: "],
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": 1,
                    "logprob_token_ids": ids,
                    "return_tokens_as_token_ids": True,
                },
            )
        except SGLangError as exc:
            LOG.info("vllm_list_prompt_scoring=False (%s)", exc)
            self._list_prompt_scoring = False
            self._list_prompt_scoring_checked = True
            return
        choices = response.get("choices") if isinstance(response, Mapping) else None
        if not isinstance(choices, list) or len(choices) != 2:
            self._list_prompt_scoring = False
            self._list_prompt_scoring_checked = True
            return
        try:
            for choice in choices:
                if not isinstance(choice, Mapping):
                    raise VLLMError("list-prompt choice is not an object")
                self._extract_top_logprobs(choice, ids)
        except VLLMError as exc:
            LOG.info("vllm_list_prompt_scoring=False (%s)", exc)
            self._list_prompt_scoring = False
            self._list_prompt_scoring_checked = True
            return
        self._list_prompt_scoring = True
        self._list_prompt_scoring_checked = True
        LOG.info("vllm_list_prompt_scoring=True")

    # -------------------------------------------------------------- generate

    def _generate(self, payload: Mapping[str, Any]) -> Any:
        self._ensure_capabilities()
        scope = _call_scope.get()
        if scope is not None:
            scope.check()
        try:
            if self._active_images.get():
                response = self._generate_with_images(payload)
            else:
                response = self._generate_completions(payload)
        except SGLangError as exc:
            timed_out = False
            if scope is not None and scope.deadline is not None:
                cause = type(exc.__cause__).__name__ if exc.__cause__ else ""
                timed_out = (
                    scope.expired()
                    or "Timeout" in cause
                    or "timed out" in str(exc).lower()
                )
            if timed_out:
                raise GenerationTimeout("generate() ran out of time") from exc
            raise
        if scope is not None:
            scope.usage._add(response)
            if scope.unmeasured_images:
                # Image tokens are already in the server's prompt_tokens count
                # after render+generate; measure against the local text prompt.
                measure_payload = {
                    "text": payload["text"],
                }
                self._measure_images(scope, measure_payload, response)
        return response

    def score_candidates_batch(
        self,
        prefixes: Sequence[str],
        candidate_ids: Sequence[Sequence[int]],
    ) -> tuple[list[tuple[dict[int, float], Mapping[str, Any]]], float]:
        """Score candidates; use one list-prompt request when the server supports it."""
        if not prefixes:
            return [], 0.0
        if len(prefixes) != len(candidate_ids):
            raise ValueError("prefixes and candidate_ids must have the same length")
        self._ensure_capabilities()
        if len(prefixes) == 1 or not self._list_prompt_scoring:
            return super().score_candidates_batch(prefixes, candidate_ids)
        # Union of all candidates: logprobs come from the full vocabulary
        # log-softmax, so each field still reads only its own IDs.
        union_ids = list(dict.fromkeys(token_id for row in candidate_ids for token_id in row))
        body = {
            "model": self._tokenizer_model(),
            "prompt": list(prefixes),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
            "logprob_token_ids": union_ids,
            "return_tokens_as_token_ids": True,
        }
        start = time.perf_counter()
        response = self._request("/v1/completions", body)
        elapsed = time.perf_counter() - start
        choices = response.get("choices") if isinstance(response, Mapping) else None
        if not isinstance(choices, list) or len(choices) != len(prefixes):
            LOG.info(
                "vllm_list_prompt_scoring fallback: expected %s choices, got %s",
                len(prefixes),
                len(choices) if isinstance(choices, list) else type(choices).__name__,
            )
            return super().score_candidates_batch(prefixes, candidate_ids)
        usage = response.get("usage") if isinstance(response, Mapping) else {}
        details = usage.get("prompt_tokens_details") if isinstance(usage, Mapping) else None
        cached = 0
        if isinstance(details, Mapping) and type(details.get("cached_tokens")) is int:
            cached = details["cached_tokens"]
        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, Mapping) else 0
        completion_tokens = (
            usage.get("completion_tokens", 0) if isinstance(usage, Mapping) else 0
        )
        # Attribute usage once for the whole batch so call-scope accounting
        # matches a single server request.
        scope = _call_scope.get()
        mapped_batch = []
        for choice, ids in zip(choices, candidate_ids):
            if not isinstance(choice, Mapping):
                raise VLLMError("list-prompt choice is not an object")
            text = choice.get("text")
            if not isinstance(text, str):
                raise VLLMError("list-prompt choice has no text")
            finish = choice.get("finish_reason")
            finish_type = "length" if finish == "length" else "stop"
            if finish in {"abort", "error"}:
                finish_type = "abort"
            mapped = {
                "text": text,
                "meta_info": {
                    "finish_reason": {
                        "type": finish_type,
                        "matched": choice.get("stop_reason"),
                    },
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached,
                    "output_token_ids_logprobs": [
                        self._extract_top_logprobs(choice, ids)
                    ],
                },
            }
            mapped_batch.append(mapped)
        if scope is not None:
            # Count as one request with per-choice meta already summed above
            # would double-count; add a single aggregate response instead.
            scope.usage._add(
                {
                    "meta_info": {
                        "prompt_tokens": prompt_tokens if type(prompt_tokens) is int else 0,
                        "completion_tokens": (
                            completion_tokens if type(completion_tokens) is int else 0
                        ),
                        "cached_tokens": cached,
                    }
                }
            )
        results: list[tuple[dict[int, float], Mapping[str, Any]]] = []
        for item, ids in zip(mapped_batch, candidate_ids):
            scores = extract_candidate_logprobs(item, ids)
            meta = item.get("meta_info", {})
            results.append((scores, meta if isinstance(meta, Mapping) else {}))
        return results, elapsed

    def _generate_completions(self, payload: Mapping[str, Any]) -> Any:
        texts = payload["text"]
        prompts = [texts] if isinstance(texts, str) else list(texts)
        params = payload.get("sampling_params", {})
        param_list = params if isinstance(params, list) else [params] * len(prompts)
        if len(param_list) != len(prompts):
            raise VLLMError("prompts and sampling_params must have the same length")
        candidate_rows = self._candidate_rows(payload, len(prompts))
        jobs = [
            (i, prompt, param, candidates)
            for i, (prompt, param, candidates) in enumerate(
                zip(prompts, param_list, candidate_rows)
            )
        ]
        results: list[Any] = [None] * len(jobs)
        if len(jobs) == 1:
            index, prompt, param, candidates = jobs[0]
            results[index] = self._one_completion(prompt, param, candidates)
        else:
            futures = {}
            for index, prompt, param, candidates in jobs:
                ctx = copy_context()
                futures[
                    self._executor().submit(
                        ctx.run, self._one_completion, prompt, param, candidates
                    )
                ] = index
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return results[0] if isinstance(texts, str) else results

    def _one_completion(
        self,
        prompt: str,
        params: Mapping[str, Any],
        candidate_ids: Sequence[int] | None,
    ) -> Mapping[str, Any]:
        body = self._to_completion_body(prompt, params, candidate_ids)
        discard = body.pop("_typellm_discard_output", False)
        response = self._request("/v1/completions", body)
        mapped = self._completion_to_sglang(response, candidate_ids)
        if discard:
            meta = dict(mapped["meta_info"])
            meta["completion_tokens"] = 0
            mapped = {**mapped, "text": "", "meta_info": meta}
        return mapped

    def _candidate_rows(
        self, payload: Mapping[str, Any], count: int
    ) -> list[list[int] | None]:
        if not payload.get("return_logprob"):
            return [None] * count
        rows = payload.get("token_ids_logprob")
        if rows is None:
            return [None] * count
        if rows and isinstance(rows[0], int):
            return [list(rows)] * count
        if len(rows) != count:
            raise VLLMError("token_ids_logprob must match the number of prompts")
        return [list(row) for row in rows]

    def _to_completion_body(
        self,
        prompt: str,
        params: Mapping[str, Any],
        candidate_ids: Sequence[int] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._tokenizer_model(),
            "prompt": prompt,
        }
        max_new = params.get("max_new_tokens", 16)
        discard = False
        if max_new == 0:
            body["max_tokens"] = 1
            discard = True
        else:
            body["max_tokens"] = int(max_new)
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("min_p", "min_p"),
        ):
            if src in params:
                body[dst] = params[src]
        if "top_k" in params:
            top_k = params["top_k"]
            body["top_k"] = 0 if top_k == -1 else top_k
        if "sampling_seed" in params:
            body["seed"] = params["sampling_seed"]
        if "stop" in params:
            body["stop"] = params["stop"]
        if params.get("no_stop_trim"):
            body["include_stop_str_in_output"] = True
        if "regex" in params:
            body["structured_outputs"] = {"regex": params["regex"]}
        elif "json_schema" in params:
            schema = params["json_schema"]
            if isinstance(schema, str):
                schema = json.loads(schema)
            body["structured_outputs"] = {"json": schema}
        if candidate_ids is not None:
            body["logprobs"] = 1
            body["logprob_token_ids"] = list(candidate_ids)
            body["return_tokens_as_token_ids"] = True
        if discard:
            body["_typellm_discard_output"] = True
        return body

    @staticmethod
    def _completion_text(response: Any) -> str:
        if not isinstance(response, Mapping):
            raise VLLMError(f"Unexpected completions response type: {type(response).__name__}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLLMError("completions response has no choices")
        text = choices[0].get("text")
        if not isinstance(text, str):
            raise VLLMError("completions response has no text")
        return text

    def _completion_to_sglang(
        self,
        response: Any,
        candidate_ids: Sequence[int] | None,
    ) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise VLLMError(f"Unexpected completions response type: {type(response).__name__}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLLMError("completions response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise VLLMError("completions choice is not an object")
        text = choice.get("text")
        if not isinstance(text, str):
            raise VLLMError("completions response has no text")
        finish = choice.get("finish_reason")
        stop_reason = choice.get("stop_reason")
        finish_type = "length" if finish == "length" else "stop"
        if finish in {"abort", "error"}:
            finish_type = "abort"
        matched = stop_reason
        if finish_type == "stop" and matched is None:
            eos = getattr(self._get_chat_tokenizer(), "eos_token_id", None)
            if type(eos) is int:
                matched = eos
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        details = usage.get("prompt_tokens_details")
        cached = 0
        if isinstance(details, Mapping) and type(details.get("cached_tokens")) is int:
            cached = details["cached_tokens"]
        meta: dict[str, Any] = {
            "finish_reason": {"type": finish_type, "matched": matched},
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": cached,
        }
        if candidate_ids is not None:
            meta["output_token_ids_logprobs"] = [
                self._extract_top_logprobs(choice, candidate_ids)
            ]
        return {"text": text, "meta_info": meta}

    def _extract_top_logprobs(
        self, choice: Mapping[str, Any], candidate_ids: Sequence[int]
    ) -> list[list[Any]]:
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, Mapping):
            raise VLLMError("completions response is missing logprobs")
        tops = logprobs.get("top_logprobs")
        if not isinstance(tops, list) or not tops:
            raise VLLMError("completions response is missing top_logprobs")
        first = tops[0]
        if not isinstance(first, Mapping):
            raise VLLMError("top_logprobs entry is not an object")
        found: dict[int, float] = {}
        for key, value in first.items():
            match = _TOKEN_ID_KEY.match(str(key))
            if match is None:
                continue
            found[int(match.group(1))] = float(value)
        missing = [token_id for token_id in candidate_ids if token_id not in found]
        if missing:
            raise VLLMError(
                f"Candidate logprobs were not found for every requested token ID: "
                f"missing={missing}, found={sorted(found)}"
            )
        return [[found[token_id], token_id, None] for token_id in candidate_ids]

    # --------------------------------------------------------------- images

    def _generate_with_images(self, payload: Mapping[str, Any]) -> Any:
        images = self._active_images.get()
        if not images:
            return self._generate_completions(payload)
        rendered = self._image_features(images)
        texts = payload["text"]
        prompts = [texts] if isinstance(texts, str) else list(texts)
        params = payload.get("sampling_params", {})
        param_list = params if isinstance(params, list) else [params] * len(prompts)
        candidate_rows = self._candidate_rows(payload, len(prompts))
        placeholder = self.image_placeholder()
        results: list[Any] = [None] * len(prompts)
        # Image requests carry large multimodal payloads; run them sequentially
        # to avoid saturating the server and deadlocking on shared client locks.
        for index, (prompt, param, candidates) in enumerate(
            zip(prompts, param_list, candidate_rows)
        ):
            results[index] = self._one_image_generate(
                prompt, param, candidates, images, placeholder, rendered
            )
        return results[0] if isinstance(texts, str) else results

    def _image_features(self, images: Sequence[str]) -> Mapping[str, Any]:
        key = tuple(images)
        with self._load_lock:
            if self._image_render_cache is None:
                self._image_render_cache = {}
            cached = self._image_render_cache.get(key)
            if cached is not None:
                return cached
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image}} for image in images
        ]
        content.append({"type": "text", "text": "TYPELLM_IMAGE_PROBE"})
        response = self._request(
            "/v1/chat/completions/render",
            {
                "model": self._tokenizer_model(),
                "messages": [{"role": "user", "content": content}],
                "add_generation_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if not isinstance(response, Mapping) or "features" not in response:
            raise VLLMError("chat render did not return multimodal features")
        with self._load_lock:
            assert self._image_render_cache is not None
            self._image_render_cache[key] = response
        return response

    def _one_image_generate(
        self,
        prompt: str,
        params: Mapping[str, Any],
        candidate_ids: Sequence[int] | None,
        images: Sequence[str],
        placeholder: str,
        rendered: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        found = prompt.count(placeholder)
        if found != len(images):
            raise VLLMError(
                f"Prompt contains {found} image placeholders for {len(images)} images; "
                "the context text must not contain the model's image tokens"
            )
        tokenizer = self._get_chat_tokenizer()
        token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        features = rendered.get("features")
        if not isinstance(features, Mapping):
            raise VLLMError("chat render features are missing")
        placeholders = features.get("mm_placeholders", {})
        image_spans = placeholders.get("image") if isinstance(placeholders, Mapping) else None
        if not isinstance(image_spans, list) or len(image_spans) != len(images):
            raise VLLMError(
                f"Expected {len(images)} image placeholders from render, "
                f"got {image_spans!r}"
            )
        # Local encode keeps one pad token per image; expand each to the
        # server's length and rewrite offsets so features stay aligned.
        pad_id = None
        probe_ids = rendered.get("token_ids")
        if isinstance(probe_ids, list) and image_spans:
            first = image_spans[0]
            if isinstance(first, Mapping):
                offset = first.get("offset")
                length = first.get("length")
                if type(offset) is int and type(length) is int and length > 0:
                    pad_id = probe_ids[offset]
        if pad_id is None:
            raise VLLMError("Could not discover the image pad token from chat render")
        expanded: list[int] = []
        new_spans: list[dict[str, int]] = []
        span_index = 0
        i = 0
        while i < len(token_ids):
            if token_ids[i] == pad_id and span_index < len(image_spans):
                length = int(image_spans[span_index]["length"])
                new_spans.append({"offset": len(expanded), "length": length})
                expanded.extend([pad_id] * length)
                span_index += 1
                i += 1
            else:
                expanded.append(token_ids[i])
                i += 1
        if span_index != len(images):
            raise VLLMError(
                f"Expanded {span_index} image pads for {len(images)} images"
            )
        rewritten_features = {
            "mm_hashes": features.get("mm_hashes"),
            "mm_placeholders": {"image": new_spans},
            "kwargs_data": features.get("kwargs_data"),
        }
        sampling = self._to_generate_sampling(params, candidate_ids)
        discard = sampling.pop("_typellm_discard_output", False)
        close_id = sampling.pop("_typellm_thinking_close_id", None)
        body = {
            "model": self._tokenizer_model(),
            "token_ids": expanded,
            "features": rewritten_features,
            "sampling_params": sampling,
        }
        response = self._request("/inference/v1/generate", body)
        mapped = self._generate_to_sglang(response, candidate_ids, close_id)
        if discard:
            meta = dict(mapped["meta_info"])
            meta["completion_tokens"] = 0
            mapped = {**mapped, "text": "", "meta_info": meta}
        return mapped

    def _to_generate_sampling(
        self,
        params: Mapping[str, Any],
        candidate_ids: Sequence[int] | None,
    ) -> dict[str, Any]:
        sampling: dict[str, Any] = {}
        max_new = params.get("max_new_tokens", 16)
        discard = False
        if max_new == 0:
            sampling["max_tokens"] = 1
            discard = True
        else:
            sampling["max_tokens"] = int(max_new)
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("min_p", "min_p"),
        ):
            if src in params:
                sampling[dst] = params[src]
        if "top_k" in params:
            top_k = params["top_k"]
            sampling["top_k"] = 0 if top_k == -1 else top_k
        if "sampling_seed" in params:
            sampling["seed"] = params["sampling_seed"]
        if "regex" in params:
            sampling["structured_outputs"] = {"regex": params["regex"]}
        elif "json_schema" in params:
            schema = params["json_schema"]
            if isinstance(schema, str):
                schema = json.loads(schema)
            sampling["structured_outputs"] = {"json": schema}
        close_id = None
        if "stop" in params:
            stops = params["stop"]
            stop_list = [stops] if isinstance(stops, str) else list(stops)
            # Prefer stop_token_ids on the token generate route.
            token_stops = []
            for stop in stop_list:
                ids = self._get_chat_tokenizer().encode(stop, add_special_tokens=False)
                if len(ids) != 1:
                    raise VLLMError(
                        f"Thinking stop {stop!r} must be a single token for the "
                        f"image generate path; got {ids!r}"
                    )
                token_stops.append(ids[0])
                if close_id is None:
                    close_id = ids[0]
            sampling["stop_token_ids"] = token_stops
            if params.get("no_stop_trim"):
                sampling["include_stop_str_in_output"] = True
        if candidate_ids is not None:
            sampling["logprobs"] = len(candidate_ids)
            sampling["logprob_token_ids"] = list(candidate_ids)
            sampling["allowed_token_ids"] = list(candidate_ids)
        if discard:
            sampling["_typellm_discard_output"] = True
        if close_id is not None and params.get("no_stop_trim"):
            sampling["_typellm_thinking_close_id"] = close_id
        return sampling

    def _generate_to_sglang(
        self,
        response: Any,
        candidate_ids: Sequence[int] | None,
        close_id: int | None,
    ) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise VLLMError(f"Unexpected generate response type: {type(response).__name__}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLLMError("generate response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise VLLMError("generate choice is not an object")
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list):
            raise VLLMError("generate response has no token_ids")
        tokenizer = self._get_chat_tokenizer()
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        protocol = detect_protocol(tokenizer)
        # Drop trailing turn/EOS markers that structured regex does not cover.
        endings = {protocol.turn_end, getattr(tokenizer, "eos_token", None)} - {None, ""}
        stripped = text
        changed = True
        while changed:
            changed = False
            for ending in endings:
                if stripped.endswith(ending):
                    stripped = stripped[: -len(ending)]
                    changed = True
        text = stripped
        if close_id is not None and token_ids and token_ids[-1] != close_id:
            # stop_token_ids trim the close token; restore it so thinking
            # completion sees the same shape as SGLang's no_stop_trim.
            if protocol.thinking_close not in text:
                text = text + protocol.thinking_close
        finish = choice.get("finish_reason")
        finish_type = "length" if finish == "length" else "stop"
        if finish in {"abort", "error"}:
            finish_type = "abort"
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        details = usage.get("prompt_tokens_details")
        cached = 0
        if isinstance(details, Mapping) and type(details.get("cached_tokens")) is int:
            cached = details["cached_tokens"]
        meta: dict[str, Any] = {
            "finish_reason": {
                "type": finish_type,
                "matched": close_id if finish_type == "stop" else None,
            },
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": cached,
        }
        if candidate_ids is not None:
            meta["output_token_ids_logprobs"] = [
                self._extract_generate_logprobs(choice, candidate_ids)
            ]
        return {"text": text, "meta_info": meta}

    def _extract_generate_logprobs(
        self, choice: Mapping[str, Any], candidate_ids: Sequence[int]
    ) -> list[list[Any]]:
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, Mapping):
            raise VLLMError("generate response is missing logprobs")
        content = logprobs.get("content")
        if not isinstance(content, list) or not content:
            raise VLLMError("generate response is missing logprobs content")
        first = content[0]
        if not isinstance(first, Mapping):
            raise VLLMError("generate logprobs content is not an object")
        tops = first.get("top_logprobs")
        if not isinstance(tops, list):
            raise VLLMError("generate response is missing top_logprobs")
        found: dict[int, float] = {}
        for entry in tops:
            if not isinstance(entry, Mapping):
                continue
            token = entry.get("token")
            match = _TOKEN_ID_KEY.match(str(token)) if token is not None else None
            if match is None:
                continue
            found[int(match.group(1))] = float(entry["logprob"])
        # Also accept the sampled token itself.
        sampled = first.get("token")
        match = _TOKEN_ID_KEY.match(str(sampled)) if sampled is not None else None
        if match is not None and "logprob" in first:
            found[int(match.group(1))] = float(first["logprob"])
        missing = [token_id for token_id in candidate_ids if token_id not in found]
        if missing:
            raise VLLMError(
                f"Candidate logprobs were not found for every requested token ID: "
                f"missing={missing}, found={sorted(found)}"
            )
        return [[found[token_id], token_id, None] for token_id in candidate_ids]
