"""Minimal native SGLang HTTP client and response-shape compatibility."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import httpx

from .protocol import detect_protocol


def _overrides(thinking: bool | None, budget: int | None) -> tuple:
    """A prompt's own thinking settings as extra arguments; none when it has none."""
    return () if thinking is None and budget is None else (thinking, budget)


class SGLangError(RuntimeError):
    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status  # The HTTP status SGLang answered with, if any.


class GenerationTimeout(SGLangError):
    """The generate() call ran out of its total time budget."""


class GenerationCancelled(RuntimeError):
    """The generate() call was cancelled through its cancel event."""


@dataclass
class Usage:
    """The tokens of one generate() call.

    input_tokens is what the caller sent, each part counted once: the context,
    the questions or schema as JSON, and the images as the server expands them.
    The other counts are what SGLang reported for the call's /generate requests,
    where the shared context is part of every prompt.
    """

    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    # The part of completion_tokens that is reasoning, not typed answers.
    thinking_tokens: int = 0
    input_tokens: int = 0

    def __repr__(self) -> str:
        # What is billed first, as the HTTP API reports it; SGLang's counts when there are any.
        billed = [("input_tokens", self.input_tokens), ("thinking_tokens", self.thinking_tokens)]
        served = [(name, getattr(self, name)) for name in ("requests", "prompt_tokens", "cached_tokens",
                                                           "completion_tokens") if getattr(self, name)]
        return "Usage(" + ", ".join(f"{name}={value}" for name, value in billed + served) + ")"

    def _add(self, response: Any) -> None:
        self.requests += 1
        for item in response if isinstance(response, list) else [response]:
            meta = item.get("meta_info") if isinstance(item, Mapping) else None
            if not isinstance(meta, Mapping):
                continue
            for name in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                value = meta.get(name)
                if type(value) is int:
                    setattr(self, name, getattr(self, name) + value)


def _count_thinking(response: Any) -> None:
    """Add a thinking request's generated tokens to this call's usage."""
    scope = _call_scope.get()
    if scope is None:
        return
    for item in response if isinstance(response, list) else [response]:
        meta = item.get("meta_info") if isinstance(item, Mapping) else None
        tokens = meta.get("completion_tokens") if isinstance(meta, Mapping) else None
        if type(tokens) is int:
            scope.usage.thinking_tokens += tokens


@dataclass
class _CallScope:
    usage: Usage = field(default_factory=Usage)
    deadline: float | None = None  # time.monotonic() when the call runs out of time
    cancel: threading.Event | None = None
    # Images whose tokens input_tokens still lacks; the first response measures them.
    unmeasured_images: int = 0

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def check(self) -> None:
        """Stop before the next request once the call is cancelled or out of time."""
        if self.cancel is not None and self.cancel.is_set():
            raise GenerationCancelled("generate() was cancelled")
        if self.expired():
            raise GenerationTimeout("generate() ran out of time")

    def socket_timeout(self, timeout: float) -> float:
        if self.deadline is None:
            return timeout
        return max(0.001, min(timeout, self.deadline - time.monotonic()))


# Set for the duration of one generate() call, in its own thread or task.
_call_scope: ContextVar[_CallScope | None] = ContextVar("typellm_call_scope", default=None)


@contextmanager
def call_scope(
    timeout: float | None = None, cancel: threading.Event | None = None,
) -> Iterator[_CallScope]:
    """Track every /generate request made in this block, in this thread or task.

    With a timeout (seconds, for the whole block) or a cancel event, the block
    stops before its next request once either runs out; a request already
    sent still finishes.
    """
    if timeout is not None and (type(timeout) not in (int, float) or not timeout > 0):
        raise ValueError("timeout must be a positive number of seconds or None")
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise ValueError("cancel must be a threading.Event or None")
    deadline = None if timeout is None else time.monotonic() + timeout
    scope = _CallScope(deadline=deadline, cancel=cancel)
    token = _call_scope.set(scope)
    try:
        yield scope
    finally:
        _call_scope.reset(token)


# One character inside a JSON string: anything but a quote, backslash or control
# character, or an escape sequence.
_JSON_STRING_CHAR = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'


def _after_open_quote(text: str) -> str:
    """Drop the ' "' or '"' the model wrote after '{"name":'."""
    if not text.startswith(('"', ' "')):
        raise ValueError(text)
    return text[text.index('"') + 1:]


def _closed_json_string(text: str) -> str:
    """Decode ' "characters"}' written after '{"name":'."""
    text = _after_open_quote(text)
    if not text.endswith('"}'):
        raise ValueError(text)
    return json.loads('"' + text[:-1])



class SGLangClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model: str | None = None,
        timeout: float = 120.0,
        tokenizer: str | None = None,
        *,
        thinking_budget: int | None = None,
        text_max_tokens: int = 128,
        answer_reserve_tokens: int = 64,
    ) -> None:
        if thinking_budget is not None and (type(thinking_budget) is not int or thinking_budget <= 0):
            raise ValueError("thinking_budget must be a positive integer or None")
        if type(text_max_tokens) is not int or text_max_tokens <= 0:
            raise ValueError("text_max_tokens must be a positive integer")
        if type(answer_reserve_tokens) is not int or answer_reserve_tokens <= 0:
            raise ValueError("answer_reserve_tokens must be a positive integer")
        self.answer_reserve_tokens = max(answer_reserve_tokens, text_max_tokens)
        self._context_length_cache: int | None = None
        self.text_max_tokens = text_max_tokens
        # The budget of fields that think without their own.
        self.thinking_budget = thinking_budget
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.tokenizer = tokenizer
        self._label_tokens: dict[str, tuple[int, str]] = {}
        self._model_info_cache: Mapping[str, Any] | None = None
        self._chat_tokenizer: Any | None = None
        self._image_placeholder_cache: str | None = None
        # Serializes the expensive lazy loads when threads share one client.
        self._load_lock = threading.RLock()
        self._http_client: httpx.Client | None = None
        # Per-thread/task, so concurrent generate() calls never share images.
        self._active_images: ContextVar[tuple[str, ...]] = ContextVar(
            f"typellm_images_{id(self)}", default=()
        )

    def __getstate__(self) -> dict[str, Any]:
        # A ContextVar cannot be pickled; images belong to one call anyway.
        state = self.__dict__.copy()
        del state["_active_images"]
        del state["_load_lock"]
        state["_http_client"] = None  # Each copy opens its own connections.
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._active_images = ContextVar(f"typellm_images_{id(self)}", default=())
        self._load_lock = threading.RLock()

    def close(self) -> None:
        """Close the pooled connections to SGLang; a later request reopens them."""
        with self._load_lock:
            http_client, self._http_client = self._http_client, None
        if http_client is not None:
            http_client.close()

    def __enter__(self) -> "SGLangClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def warmup(self) -> None:
        """Load the model info and chat tokenizer before the first request."""
        self._tokenizer_model()
        self._get_chat_tokenizer()

    def _info(self, name: str) -> Any:
        # SGLang 0.5.6 renamed /get_<name> to /<name>; older servers and
        # sglang-router 0.3.2 only know the old name.
        try:
            return self._request(f"/{name}")
        except SGLangError as exc:
            if exc.status != 404:
                raise
            return self._request(f"/get_{name}")

    def _model_info(self) -> Mapping[str, Any]:
        with self._load_lock:
            if self._model_info_cache is None:
                response = self._info("model_info")
                if not isinstance(response, Mapping):
                    raise SGLangError("/model_info returned a non-object response")
                self._model_info_cache = response
            return self._model_info_cache

    def _http(self) -> httpx.Client:
        with self._load_lock:
            if self._http_client is None:
                # Keep-alive connections: one call makes many small requests. Idle
                # ones are dropped before SGLang closes them (5 s), so a request
                # rarely goes out on a connection the server has just closed.
                self._http_client = httpx.Client(limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=64, keepalive_expiry=2.0,
                ))
            return self._http_client

    def _request(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        allow_text: bool = False,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        scope = _call_scope.get()
        timeout = self.timeout if scope is None else scope.socket_timeout(self.timeout)
        try:
            try:
                raw = self._send(path, payload is None, body, timeout)
            except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError):
                # The connection dropped before an answer, most often a pooled one
                # the server had just closed. SGLang is fine: try once on a new one.
                if scope is not None:
                    scope.check()
                    timeout = scope.socket_timeout(self.timeout)
                raw = self._send(path, payload is None, body, timeout)
        except httpx.HTTPError as exc:
            # Connection failures, timeouts, resets and truncated responses.
            raise SGLangError(
                f"Could not reach SGLang at {self.base_url}: {exc!r}"
            ) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            if allow_text:
                return raw
            raise SGLangError(
                f"SGLang {path} returned non-JSON data: {raw[:500]}"
            ) from exc

    def _send(self, path: str, get: bool, body: bytes | None, timeout: float) -> str:
        # Streamed so a failed body read still reports the status it follows.
        with self._http().stream(
            "GET" if get else "POST",
            self.base_url + path,
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        ) as response:
            if response.status_code >= 400:
                try:
                    detail = response.read().decode("utf-8", errors="replace")
                except httpx.HTTPError as read_error:
                    detail = f"Could not read error response: {read_error}"
                raise SGLangError(
                    f"SGLang {path} returned HTTP {response.status_code}: {detail}",
                    status=response.status_code,
                )
            return response.read().decode("utf-8")

    @contextmanager
    def images(self, images: Sequence[str]) -> Iterator[None]:
        """Attach encoded images to every /generate request made in this block."""
        images = tuple(images)
        if images:
            self.image_placeholder()  # Fail before any request if unsupported.
        token = self._active_images.set(images)
        try:
            yield
        finally:
            self._active_images.reset(token)

    def image_placeholder(self) -> str:
        """Return the text the chat template writes for one image."""
        if self._image_placeholder_cache is not None:
            return self._image_placeholder_cache
        # Derive the placeholder from the template, never hardcode a model's
        # vision tokens. The marker is confined to this local template probe.
        marker = "TYPELLM_IMAGE_BOUNDARY_3f1d7a"
        unsupported = (
            "The served chat template does not render image content; "
            "use a vision-language model to pass images"
        )
        text_only = self.render_chat(
            [{"role": "user", "content": marker}], add_generation_prompt=False
        )
        try:
            with_image = self.render_chat(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": marker}]}],
                add_generation_prompt=False,
            )
        except SGLangError as exc:
            raise SGLangError(unsupported) from exc
        if text_only.count(marker) != 1 or with_image.count(marker) != 1:
            raise SGLangError(unsupported)
        before, after = text_only.split(marker)
        image_before, image_after = with_image.split(marker)
        if not image_before.startswith(before) or image_after != after:
            raise SGLangError(unsupported)
        placeholder = image_before[len(before):].strip()
        if not placeholder:
            raise SGLangError(unsupported)
        self._image_placeholder_cache = placeholder
        return placeholder

    def _generate(self, payload: Mapping[str, Any]) -> Any:
        """POST /generate, adding the active images to each prompt."""
        payload = self._with_images(payload)
        scope = _call_scope.get()
        if scope is None:
            return self._request("/generate", payload)
        scope.check()
        try:
            response = self._request("/generate", payload)
        except SGLangError as exc:
            if scope.expired():
                raise GenerationTimeout("generate() ran out of time") from exc
            raise
        scope.usage._add(response)
        if scope.unmeasured_images:
            self._measure_images(scope, payload, response)
        return response

    def count_tokens(self, text: str) -> int:
        return len(self._get_chat_tokenizer().encode(text, add_special_tokens=False))

    def _measure_images(self, scope: _CallScope, payload: Mapping[str, Any], response: Any) -> None:
        """Add the images' tokens to input_tokens: the server's prompt count less the local one."""
        text = payload.get("text")
        prompt = text if isinstance(text, str) else text[0] if isinstance(text, list) and text else None
        item = response[0] if isinstance(response, list) and response else response
        meta = item.get("meta_info") if isinstance(item, Mapping) else None
        served = meta.get("prompt_tokens") if isinstance(meta, Mapping) else None
        if not isinstance(prompt, str) or type(served) is not int:
            return
        # The server replaces each placeholder with the image's tokens.
        placeholders = scope.unmeasured_images * self.count_tokens(self.image_placeholder())
        scope.usage.input_tokens += max(0, served - self.count_tokens(prompt) + placeholders)
        scope.unmeasured_images = 0

    def _with_images(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        images = self._active_images.get()
        if not images:
            return payload
        text = payload["text"]
        prompts = [text] if isinstance(text, str) else list(text)
        placeholder = self.image_placeholder()
        for prompt in prompts:
            found = prompt.count(placeholder)
            if found != len(images):
                raise SGLangError(
                    f"Prompt contains {found} image placeholders for {len(images)} images; "
                    "the context text must not contain the model's image tokens"
                )
        if isinstance(text, str):
            image_data: Any = list(images)
        elif len(images) == 1:
            # SGLang gives a lone item to every prompt, so the image goes once
            # rather than once per prompt (720 copies for a 6-value "all").
            image_data = images[0]
        else:
            # A list is read as one entry per prompt.
            image_data = [list(images) for _ in prompts]
        return {**payload, "image_data": image_data}

    def _tokenizer_model(self) -> str:
        if self.model:
            return self.model
        info = self._model_info()
        for key in ("served_model_name", "model_path", "tokenizer_path"):
            value = info.get(key) if isinstance(info, Mapping) else None
            if isinstance(value, str) and value:
                self.model = value
                return value
        raise SGLangError(
            "Could not discover a tokenizer model from /model_info; "
            "pass model=... or set SGLANG_MODEL"
        )

    def _tokenizer_sources(self) -> list[str]:
        """Where the chat tokenizer may load from, best first.

        SGLang reports its own paths, which may exist only on its machine; its
        served model name, often a Hugging Face ID, is tried after them.
        """
        if self.tokenizer:
            return [self.tokenizer]
        info = self._model_info()
        sources = [info.get(key) if isinstance(info, Mapping) else None
                   for key in ("tokenizer_path", "model_path", "served_model_name")] + [self.model]
        sources = list(dict.fromkeys(value for value in sources if isinstance(value, str) and value))
        if not sources:
            raise SGLangError("Could not discover the tokenizer used by SGLang; pass tokenizer=...")
        return sources

    def _get_chat_tokenizer(self) -> Any:
        with self._load_lock:
            return self._load_chat_tokenizer()

    def _load_chat_tokenizer(self) -> Any:
        if self._chat_tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise SGLangError(
                    "Chat-template rendering requires transformers; install it "
                    "with `pip install transformers`"
                ) from exc
            sources = self._tokenizer_sources()
            failure: Exception | None = None
            for source in sources:
                try:
                    self._chat_tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
                    break
                except Exception as exc:
                    failure = exc
            else:
                raise SGLangError(
                    f"Could not load the tokenizer chat template from {', '.join(map(repr, sources))}. "
                    "Pass the model's local tokenizer path or Hugging Face ID as tokenizer=..."
                ) from failure
        return self._chat_tokenizer

    def render_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
        finish_thinking: bool = True,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        """Render history; optionally finish thinking before constrained decoding.

        thinking asks this prompt to reason; thinking_budget overrides the client's.
        """
        think = bool(thinking)
        tokenizer = self._get_chat_tokenizer()
        if not getattr(tokenizer, "chat_template", None):
            raise SGLangError("The served tokenizer does not define a chat template")
        try:
            detect_protocol(tokenizer)
        except ValueError as exc:
            raise SGLangError(str(exc)) from exc
        try:
            rendered = tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=think and add_generation_prompt,
            )
        except Exception as exc:
            raise SGLangError(
                "Could not render the tokenizer chat template; ensure "
                "transformers and jinja2 are installed"
            ) from exc
        if not isinstance(rendered, str):
            raise SGLangError("Tokenizer chat template returned non-text output")
        # SGLang tokenizes the prompt with special tokens, so a tokenizer that
        # prepends BOS would double the one the template already wrote.
        bos = getattr(tokenizer, "bos_token", None)
        if bos and rendered.startswith(bos) and tokenizer.encode("x")[0] == tokenizer.bos_token_id:
            rendered = rendered[len(bos):]
        if add_generation_prompt and finish_thinking:
            return self._prepare_answer_prefix(rendered, *_overrides(thinking, thinking_budget))
        return rendered

    def _prepare_answer_prefix(self, prefix: str, thinking: bool | None = None,
                               budget: int | None = None) -> str:
        protocol = detect_protocol(self._get_chat_tokenizer())
        # A template that always opens thinking reasons whatever the setting.
        if thinking or protocol.has_open_thinking(prefix):
            return self._finish_thinking(prefix, *(() if budget is None else (budget,)))
        return prefix

    def _continuation_parts(self, question: str | None = None,
                            thinking: bool | None = None) -> tuple[str, str]:
        # Derive turn delimiters from the actual tokenizer, never hardcode a
        # model's chat tokens. The marker is confined to this local template probe.
        marker = "TYPELLM_ASSISTANT_BOUNDARY_8b46c9"
        messages = [{"role": "user", "content": "Context"},
                    {"role": "assistant", "content": marker}]
        closed = self.render_chat(messages, add_generation_prompt=False)
        if closed.count(marker) != 1:
            raise SGLangError("Chat template cannot preserve assistant content for KV continuation")
        closing = closed.split(marker)[1]
        if question is None:
            return closing, ""
        extended = self.render_chat(
            messages + [{"role": "user", "content": question}],
            add_generation_prompt=True, finish_thinking=False,
            **({} if thinking is None else {"thinking": thinking}),
        )
        if extended.count(marker) != 1:
            raise SGLangError("Chat template cannot preserve assistant content for KV continuation")
        tail = extended.split(marker)[1]
        if not tail.startswith(closing):
            raise SGLangError("Chat template does not support append-only KV continuation")
        return closing, tail[len(closing):]

    def complete_chat_prefix(self, prompt: str, answer: str) -> str:
        closing, _ = self._continuation_parts()
        return prompt + answer + closing

    def extend_chat_prefix(self, prefix: str, question: str, *, finish_thinking: bool = True,
                           thinking: bool | None = None, thinking_budget: int | None = None) -> str:
        _, suffix = self._continuation_parts(question, thinking)
        prompt = prefix + suffix
        return (self._prepare_answer_prefix(prompt, *_overrides(thinking, thinking_budget))
                if finish_thinking else prompt)

    def thinking_text(self, prompt: str, finished: str) -> str | None:
        """The reasoning that finishing thinking added to prompt, or None."""
        if finished == prompt or not finished.startswith(prompt):
            return None
        added = finished[len(prompt):]
        close = detect_protocol(self._get_chat_tokenizer()).thinking_close
        if close and close in added:
            added = added[:added.index(close)]
        return added.strip() or None

    def prepare_answer_prefixes(
        self,
        prefixes: Sequence[str],
        thinking: Sequence[bool | None] | None = None,
        budgets: Sequence[int | None] | None = None,
    ) -> list[str]:
        """Finish thinking for many generation prompts in one batched request.

        thinking says which prompts reason and budgets their own budgets (None: the
        client's); only the prompts that think join the batch.
        """
        protocol = detect_protocol(self._get_chat_tokenizer())
        wants = [bool(t) for t in (thinking or [None] * len(prefixes))]
        budgets = list(budgets or [None] * len(prefixes))
        pending = [i for i, p in enumerate(prefixes) if wants[i] or protocol.has_open_thinking(p)]
        finished = list(prefixes)
        # Budgets go only to prompts that have their own, as before for the rest.
        own = [budgets[i] for i in pending]
        if len(pending) == 1:
            finished[pending[0]] = self._finish_thinking(prefixes[pending[0]], *([] if own[0] is None else own))
        elif pending:
            done = self._finish_thinking_batch([prefixes[i] for i in pending],
                                               *(() if all(b is None for b in own) else (own,)))
            for i, value in zip(pending, done):
                finished[i] = value
        return finished

    def _context_length(self) -> int:
        """Read the served context window, including any server override."""
        if self._context_length_cache is None:
            info = self._info("server_info")
            candidates = []
            if isinstance(info, Mapping):
                candidates.append(info.get("context_length"))
                args = info.get("server_args", {})
                if isinstance(args, Mapping):
                    candidates.append(args.get("context_length"))
            limits = [n for n in candidates if type(n) is int and n > 0]
            if not limits:
                models = self._request("/v1/models")
                data = models.get("data", []) if isinstance(models, Mapping) else []
                for item in data:
                    if isinstance(item, Mapping) and (len(data) == 1 or item.get("id") == self.model):
                        limit = item.get("max_model_len")
                        if type(limit) is int and limit > 0:
                            limits.append(limit)
            if not limits:
                raise SGLangError("Could not discover the served context length to reserve final-answer space")
            self._context_length_cache = min(limits)
        return self._context_length_cache

    def _finish_thinking(self, prefix: str, budget: int | None = None) -> str:
        params, image_tokens = self._thinking_params(prefix, budget=budget)
        response = self._generate({"text": prefix, "sampling_params": params})
        _count_thinking(response)
        return self._complete_thinking(prefix, response, image_tokens)

    def _finish_thinking_batch(self, prefixes: Sequence[str],
                               budgets: Sequence[int | None] | None = None) -> list[str]:
        budgets = list(budgets or [None] * len(prefixes))
        served: list[int | None] = [None] * len(prefixes)
        if self._active_images.get():
            response = self._generate({
                "text": list(prefixes),
                "sampling_params": {"max_new_tokens": 0, "temperature": 0},
            })
            if not isinstance(response, list) or len(response) != len(prefixes):
                raise SGLangError("Unexpected prompt-count batch response shape")
            served = [item.get("meta_info", {}).get("prompt_tokens") if isinstance(item, Mapping) else None
                      for item in response]
        planned = [self._thinking_params(prefix, count, budget)
                   for prefix, count, budget in zip(prefixes, served, budgets)]
        response = self._generate({
            "text": list(prefixes),
            "sampling_params": [params for params, _ in planned],
        })
        if not isinstance(response, list) or len(response) != len(prefixes):
            raise SGLangError("Unexpected thinking batch response shape")
        _count_thinking(response)
        return [self._complete_thinking(prefix, item, image_tokens)
                for prefix, item, (_, image_tokens) in zip(prefixes, response, planned)]

    def _thinking_params(self, prefix: str, served_tokens: int | None = None,
                         budget: int | None = None) -> tuple[dict[str, Any], int]:
        """Return sampling params for one thinking request and its image-token count."""
        tokenizer = self._get_chat_tokenizer()
        protocol = detect_protocol(tokenizer)
        if not protocol.has_open_thinking(prefix):
            raise SGLangError(
                f"thinking requires a native chat template ending in an open "
                f"{protocol.thinking_open} block; this template may not support thinking"
            )
        forced_end = protocol.forced_close()
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        image_tokens = 0
        if self._active_images.get():
            # Each image placeholder expands to many tokens on the server, so
            # count the prompt there; the prefill also warms the cache.
            if served_tokens is None:
                served_tokens = self.cache_prefix(prefix).get("prompt_tokens")
            if type(served_tokens) is not int:
                raise SGLangError("SGLang did not report prompt_tokens; cannot budget thinking with images")
            image_tokens = max(0, served_tokens - prefix_tokens)
            prefix_tokens += image_tokens
        closing_tokens = len(tokenizer.encode(forced_end, add_special_tokens=False))
        # This is available context, not an independent default thinking budget.
        available = self._context_length() - prefix_tokens - self.answer_reserve_tokens - closing_tokens - 16
        if available <= 0:
            raise SGLangError("Input leaves no room for thinking and the final constrained answer")
        budget = self.thinking_budget if budget is None else budget
        limit = available if budget is None else min(available, budget)
        return {
            "max_new_tokens": limit,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "stop": [protocol.thinking_close], "no_stop_trim": True,
        }, image_tokens

    def _complete_thinking(self, prefix: str, response: Any, image_tokens: int) -> str:
        tokenizer = self._get_chat_tokenizer()
        protocol = detect_protocol(tokenizer)
        text = response.get("text") if isinstance(response, Mapping) else None
        if not isinstance(text, str):
            raise SGLangError("Thinking returned non-text output; no typed result returned")
        meta = response.get("meta_info", {})
        finish = meta.get("finish_reason", {}) if isinstance(meta, Mapping) else {}
        if isinstance(finish, Mapping) and finish.get("type") in {"abort", "error"}:
            raise SGLangError("Thinking was aborted; no typed result returned")
        if protocol.thinking_close not in text:
            stop_kind = finish.get("type") if isinstance(finish, Mapping) else None
            ended_turn = False
            if stop_kind == "stop":
                matched = finish.get("matched")
                # Only recover a recognized native EOS/turn end, not an
                # arbitrary stop or an unreported/truncated server response.
                endings = {protocol.turn_end, getattr(tokenizer, "eos_token", None)} - {None, ""}
                for ending in endings:
                    ids = tokenizer.encode(ending, add_special_tokens=False)
                    if matched == ending or (type(matched) is int and ids == [matched]):
                        ended_turn = True
                        # Some servers preserve the token, others filter it.
                        # Remove only its trailing occurrence, never user text.
                        trimmed = text.rstrip()
                        if trimmed.endswith(ending):
                            text = trimmed[:-len(ending)]
                        break
            if (stop_kind != "length" and not ended_turn) or not text.strip():
                raise SGLangError(f"Thinking ended without a closing {protocol.thinking_close} marker; no typed result returned")
            completed = protocol.answer_prefix(prefix, text + "\n\nI will now give the final answer.\n")
            # Guard tokenizer/count mismatches before issuing a final request.
            if len(tokenizer.encode(completed, add_special_tokens=False)) + image_tokens + self.answer_reserve_tokens > self._context_length():
                raise SGLangError("Thinking response exceeded the reserved context space")
            logging.getLogger("typellm").info(
                "Thinking %s; closing reasoning before constrained decoding",
                "ended at native turn terminator" if ended_turn else "length limit reached",
            )
            return completed
        reasoning = text.split(protocol.thinking_close, 1)[0]
        if not reasoning.strip():
            raise SGLangError("Thinking returned an empty block; no typed result returned")
        # Discard any unconstrained answer after the marker. The existing
        # runtime records only selected labels/numbers in subsequent history.
        return protocol.answer_prefix(prefix, reasoning)

    def single_token(self, label: str) -> tuple[int, str]:
        """Return (token_id, exact decoded text), rejecting multi-token labels."""
        if label in self._label_tokens:
            return self._label_tokens[label]
        model = self._tokenizer_model()
        # A label continues the prompt, so it must be tokenized without BOS;
        # SGLang adds special tokens by default.
        tokenized = self._request(
            "/v1/tokenize",
            {"model": model, "prompt": label, "add_special_tokens": False},
        )
        token_ids = tokenized.get("tokens") if isinstance(tokenized, Mapping) else None
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            raise ValueError(
                f"Candidate label {label!r} must encode to exactly one token; "
                f"SGLang returned token IDs {token_ids!r}"
            )
        token_id = int(token_ids[0])
        detokenized = self._request(
            "/v1/detokenize", {"model": model, "tokens": [token_id]}
        )
        token_text = detokenized.get("text") if isinstance(detokenized, Mapping) else None
        if token_text != label:
            raise ValueError(
                f"Candidate {label!r} tokenizes to ID {token_id}, but that token "
                f"decodes as {token_text!r}; exact append would be ambiguous"
            )
        result = (token_id, label)
        self._label_tokens[label] = result
        return result

    def score_candidates(
        self, prefix: str, candidate_ids: Sequence[int]
    ) -> tuple[dict[int, float], Mapping[str, Any], float]:
        payload = {
            "text": prefix,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "token_ids_logprob": list(candidate_ids),
            "return_text_in_logprobs": True,
        }
        start = time.perf_counter()
        response = self._generate(payload)
        elapsed = time.perf_counter() - start
        # The unrestricted generated token is deliberately ignored. Decisions
        # use only the explicitly requested candidate-token log probabilities.
        scores = extract_candidate_logprobs(response, candidate_ids)
        meta = response.get("meta_info", {}) if isinstance(response, Mapping) else {}
        return scores, meta, elapsed

    def cache_prefix(self, prefix: str) -> Mapping[str, Any]:
        """Prefill a shared prefix without generating an output token."""
        response = self._generate(
            {
                "text": prefix,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0},
            },
        )
        if isinstance(response, list):
            if len(response) != 1:
                raise SGLangError(
                    f"Expected one prefix-cache response, got {len(response)}"
                )
            response = response[0]
        if not isinstance(response, Mapping):
            raise SGLangError(
                "Unexpected prefix-cache response type: "
                f"{type(response).__name__}"
            )
        meta = response.get("meta_info", {})
        return meta if isinstance(meta, Mapping) else {}

    def score_candidates_batch(
        self,
        prefixes: Sequence[str],
        candidate_ids: Sequence[Sequence[int]],
    ) -> tuple[list[tuple[dict[int, float], Mapping[str, Any]]], float]:
        """Score one candidate set for each prompt in a native SGLang batch."""
        if not prefixes:
            return [], 0.0
        if len(prefixes) != len(candidate_ids):
            raise ValueError("prefixes and candidate_ids must have the same length")
        payload = {
            "text": list(prefixes),
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "token_ids_logprob": [list(ids) for ids in candidate_ids],
            "return_text_in_logprobs": True,
        }
        start = time.perf_counter()
        response = self._generate(payload)
        elapsed = time.perf_counter() - start
        if isinstance(response, Mapping) and len(prefixes) == 1:
            responses = [response]
        elif isinstance(response, list):
            responses = response
        else:
            raise SGLangError(
                f"Expected a batched /generate response, got {type(response).__name__}"
            )
        if len(responses) != len(prefixes):
            raise SGLangError(
                f"Expected {len(prefixes)} batched responses, got {len(responses)}"
            )

        results: list[tuple[dict[int, float], Mapping[str, Any]]] = []
        for item, ids in zip(responses, candidate_ids):
            scores = extract_candidate_logprobs(item, ids)
            meta = item.get("meta_info", {}) if isinstance(item, Mapping) else {}
            results.append((scores, meta if isinstance(meta, Mapping) else {}))
        return results, elapsed

    def _batch(self, prefixes: Sequence[str], params: Sequence[Mapping[str, Any]], what: str) -> list[Any]:
        """One /generate request with one sampling-params dict per prompt."""
        response = self._generate({"text": list(prefixes), "sampling_params": list(params)})
        if isinstance(response, Mapping) and len(prefixes) == 1:
            response = [response]
        if not isinstance(response, list) or len(response) != len(prefixes):
            raise SGLangError(f"Unexpected {what} batch response shape")
        return response

    @staticmethod
    def _number_params(patterns: Sequence[str], max_new_tokens: int, temperature: float, seed: int) -> list[dict]:
        return [{"max_new_tokens": max_new_tokens, "temperature": temperature,
                 "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                 "sampling_seed": seed + i, "regex": pattern}
                for i, pattern in enumerate(patterns)]

    @staticmethod
    def _read_numbers(items: Sequence[Any]) -> list[str]:
        texts = []
        for item in items:
            meta = item.get("meta_info", {}) if isinstance(item, Mapping) else {}
            finish = meta.get("finish_reason", {}) if isinstance(meta, Mapping) else {}
            kind = finish.get("type") if isinstance(finish, Mapping) else finish
            text = item.get("text") if isinstance(item, Mapping) else None
            if kind != "stop" or not isinstance(text, str):
                raise SGLangError(f"Number generation did not complete normally: {finish!r}")
            texts.append(text)
        return texts

    def _text_params(self, nullable: Sequence[bool], after_key: bool, temperature: float,
                     seed: int) -> list[dict]:
        if any(nullable) and not after_key:
            raise ValueError("nullable text needs the prefilled '{\"name\":' prompt")
        params = []
        for can_be_null in nullable:
            if after_key:
                # End with the object's closing brace too: models close {"name": "text"}
                # with the single token '"}', which a bare '"' would rule out.
                value = '"' + _JSON_STRING_CHAR + '*"'
                # A nullable field may write null instead, in the same request.
                value = f"(?:{value}|null)" if can_be_null else value
                constraint = {"regex": " ?" + value + "\\}"}
            else:
                constraint = {"json_schema": json.dumps({"type": "string"})}
            params.append({"max_new_tokens": self.text_max_tokens,
                           "temperature": temperature, "sampling_seed": seed, **constraint})
        return params

    @staticmethod
    def _read_texts(items: Sequence[Any], nullable: Sequence[bool], after_key: bool) -> list[str | None]:
        values: list[str | None] = []
        for item, can_be_null in zip(items, nullable):
            if not isinstance(item, Mapping):
                raise SGLangError("Invalid text response")
            meta = item.get("meta_info", {})
            finish = meta.get("finish_reason", {}) if isinstance(meta, Mapping) else {}
            kind = finish.get("type") if isinstance(finish, Mapping) else finish
            if can_be_null and isinstance(item.get("text"), str) and item["text"].lstrip().startswith("null"):
                if kind != "stop":
                    raise SGLangError(f"Text generation did not complete normally: {finish!r}")
                values.append(None)
                continue
            if kind != "stop":
                raise SGLangError(f"Text generation did not complete normally: {finish!r}")
            try:
                text = item["text"]
                value = _closed_json_string(text) if after_key else json.loads(text)
            except (KeyError, TypeError, ValueError) as exc:
                raise SGLangError("Text generation returned an invalid JSON string") from exc
            if not isinstance(value, str):
                raise SGLangError("Text generation returned a non-string value")
            if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
                raise SGLangError("Text generation returned an unpaired Unicode surrogate")
            values.append(value)
        return values

    @staticmethod
    def _check_nullable(nullable, prefixes):
        nullable = [False] * len(prefixes) if nullable is None else list(nullable)
        if len(nullable) != len(prefixes):
            raise ValueError("prefixes and nullable must have the same length")
        return nullable

    def generate_numbers(
        self,
        prefixes: Sequence[str],
        patterns: Sequence[str],
        max_new_tokens: int,
        *,
        temperature: float = 0,
        seed: int = 0,
    ) -> list[str]:
        """Generate one regex-constrained value per prompt in a native batch.

        SGLang masks every step to the pattern, so a number takes one request
        instead of one per token. Returns the raw generated text.
        """
        if len(prefixes) != len(patterns):
            raise ValueError("prefixes and patterns must have the same length")
        if not prefixes:
            return []
        params = self._number_params(patterns, max_new_tokens, temperature, seed)
        return self._read_numbers(self._batch(prefixes, params, "number"))

    def generate_texts(
        self,
        prefixes: Sequence[str],
        *,
        temperature: float = 0,
        seed: int = 0,
        after_key: bool = False,
        nullable: Sequence[bool] | None = None,
    ) -> list[str | None]:
        """Generate JSON strings in a native batch, then validate every value.

        With after_key, every prompt ends with '{"name":', and the model writes the
        opening quote, the characters and the closing '"}'. Writing the quote
        itself lets it start with a merged token such as ' "$', which keeps the
        first character. A string stops at text_max_tokens.
        """
        nullable = self._check_nullable(nullable, prefixes)
        if not prefixes:
            return []
        params = self._text_params(nullable, after_key, temperature, seed)
        return self._read_texts(self._batch(prefixes, params, "text"), nullable, after_key)

    def generate_fields(
        self,
        number_prefixes: Sequence[str],
        patterns: Sequence[str],
        number_max_tokens: int,
        text_prefixes: Sequence[str],
        *,
        temperature: float = 0,
        number_seed: int = 0,
        text_seed: int = 0,
        after_key: bool = False,
        nullable: Sequence[bool] | None = None,
    ) -> tuple[list[str], list[str | None]]:
        """Numbers and strings of one layer in a single request, decoded side by side.

        Returns what generate_numbers and generate_texts would for each part.
        """
        if len(number_prefixes) != len(patterns):
            raise ValueError("prefixes and patterns must have the same length")
        nullable = self._check_nullable(nullable, text_prefixes)
        params = (self._number_params(patterns, number_max_tokens, temperature, number_seed)
                  + (self._text_params(nullable, after_key, temperature, text_seed)
                     if text_prefixes else []))
        prefixes = list(number_prefixes) + list(text_prefixes)
        if not prefixes:
            return [], []
        items = self._batch(prefixes, params, "field")
        split = len(number_prefixes)
        return (self._read_numbers(items[:split]),
                self._read_texts(items[split:], nullable, after_key))

    def flush_cache(self) -> None:
        reply = self._request("/flush_cache", {}, allow_text=True)
        if isinstance(reply, str) and not reply.startswith("Cache flushed."):
            raise SGLangError(f"SGLang refused to flush its prefix cache: {reply}")


def _score_entry(entry: Any, candidate_ids: set[int]) -> tuple[int, float] | None:
    if isinstance(entry, Mapping):
        token_id = entry.get("token_id", entry.get("id"))
        logprob = entry.get("logprob", entry.get("log_prob"))
        if token_id is not None and logprob is not None:
            return int(token_id), float(logprob)
        if len(entry) == 1:
            key, value = next(iter(entry.items()))
            try:
                return int(key), float(value)
            except (TypeError, ValueError):
                return None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        # SGLang 0.5.19: [logprob, token_id, optional_token_text].
        if isinstance(entry[1], int) and entry[1] in candidate_ids:
            return int(entry[1]), float(entry[0])
        if isinstance(entry[0], int) and entry[0] in candidate_ids:
            return int(entry[0]), float(entry[1])
    return None


def extract_candidate_logprobs(
    response: Any, candidate_ids: Sequence[int]
) -> dict[int, float]:
    """Extract next-token candidate scores across known SGLang response shapes."""
    if isinstance(response, list):
        if len(response) != 1:
            raise SGLangError(f"Expected one /generate response, got {len(response)}")
        response = response[0]
    if not isinstance(response, Mapping):
        raise SGLangError(f"Unexpected /generate response type: {type(response).__name__}")

    meta = response.get("meta_info", {})
    containers = []
    for owner in (meta, response):
        if isinstance(owner, Mapping):
            for key in ("output_token_ids_logprobs", "token_ids_logprobs"):
                if owner.get(key) is not None:
                    containers.append((key, owner[key]))

    wanted = {int(token_id) for token_id in candidate_ids}
    for _, container in containers:
        if isinstance(container, Mapping):
            entries = [{key: value} for key, value in container.items()]
        else:
            positions = container if isinstance(container, list) else [container]
            if positions and isinstance(positions[0], list):
                first = positions[0]
                entries = (
                    first
                    if first and isinstance(first[0], (list, tuple, Mapping))
                    else positions
                )
            else:
                entries = positions
        found: dict[int, float] = {}
        for entry in entries:
            parsed = _score_entry(entry, wanted)
            if parsed is not None and parsed[0] in wanted:
                found[parsed[0]] = parsed[1]
        if wanted.issubset(found):
            return {token_id: found[token_id] for token_id in candidate_ids}

    keys = list(meta.keys()) if isinstance(meta, Mapping) else []
    raise SGLangError(
        "Candidate logprobs were not found for every requested token ID. "
        f"wanted={sorted(wanted)}, meta_info keys={keys}, "
        f"candidate fields={containers!r}"
    )
