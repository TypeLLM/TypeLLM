"""Schema binding and constrained decision execution."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import warnings
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import permutations as all_permutations
from string import ascii_uppercase, digits
from typing import Any, Mapping, Sequence

import httpx

from .schema import (
    MAX_ENUM_CHOICES,
    SchemaError,
    compile_json_schema,
    dependency_layers,
)
from .images import encode_images
from .sglang import GenerationTimeout, SGLangClient, SGLangError, Usage, call_scope


LOG = logging.getLogger("typellm")
HOSTED_URL = "https://api.typellm.ai"
DEFAULT_SOCKET_TIMEOUT = 120.0


def _closed_answer(decision: "Choice", value_json: str) -> str:
    """The full answer an open field leaves in history, e.g. {"total": 174600}."""
    return decision.answer_prefill + " " + value_json + "}" if decision.answer_prefill else value_json


def _closed_label(decision: "Choice", label: str) -> str:
    """A choice answer as history keeps it, e.g. {"category": "A"}."""
    return decision.label_prefill + label + '"}' if decision.label_prefill else label


def _user_content(text: str, image_count: int) -> str | list[dict[str, str]]:
    """Put images ahead of the text in the first user turn."""
    if not image_count:
        return text
    return [{"type": "image"}] * image_count + [{"type": "text", "text": text}]


@dataclass(frozen=True)
class Choice:
    """A runtime decision whose control labels have been bound to values."""

    question: str
    choices: Mapping[str, Any]
    name: str | None = None
    syntax: str = "Choice"
    numeric_type: str | None = None
    text_type: bool = False
    permutations: int | str = 1
    return_probabilities: bool = False
    depends_on: tuple[str, ...] | None = None
    nullable: bool = False
    thinking: bool | None = None
    thinking_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.choices and self.numeric_type is None and not self.text_type:
            raise ValueError("Choice.choices must not be empty")
        if self.text_type and (self.choices or self.numeric_type is not None):
            raise ValueError("Text choices cannot have enum values or a numeric type")
        if self.numeric_type not in {None, "integer", "number"}:
            raise ValueError("numeric_type must be None, 'integer', or 'number'")
        if self.numeric_type is not None and self.choices:
            raise ValueError("Open numeric choices must be empty")
        if len(self.choices) > MAX_ENUM_CHOICES:
            raise ValueError(
                f"Choice has {len(self.choices)} values; "
                f"the maximum is {MAX_ENUM_CHOICES}"
            )
        if any(not label for label in self.choices):
            raise ValueError("Choice labels must be non-empty strings")

    @property
    def answer_prefill(self) -> str:
        """Start of the answer for open fields; the model continues with the value.

        Chat models tend to answer {"name": value}, so the value is decoded right
        where they would write it.
        """
        if self.name is None or not (self.text_type or self.numeric_type is not None):
            return ""
        # No trailing space: in {"name": 12} the space belongs to the value's first token.
        return "{" + json.dumps(self.name, ensure_ascii=False) + ":"

    @property
    def label_prefill(self) -> str:
        """Start of a choice answer, {"name": "; the next token is the label."""
        if self.name is None or self.text_type or self.numeric_type is not None:
            return ""
        return "{" + json.dumps(self.name, ensure_ascii=False) + ': "'

    def opening_text(self) -> str:
        """The field's prompt: the same Field / Type / Instructions / Answer lines for every type."""
        lines = []
        if self.name is not None:
            lines.append(f"Field: {json.dumps(self.name, ensure_ascii=False)}")
        # A nullable field says "or null" in both its type and its answer: a bare
        # <number> reads as "a number is required" and pulls absent values to 0.
        or_null = " or null" if self.nullable else ""
        if self.text_type:
            kind = f"string{or_null}"
        elif self.numeric_type is not None:
            kind = self.numeric_type + or_null
        else:
            kind = "boolean" if self.syntax == "Bool" else "choice"
        lines.append(f"Type: {kind}")
        lines.append(f"Instructions: {self.question}")
        if self.text_type or self.numeric_type is not None:
            placeholder = "<string" + or_null + ">" if self.text_type else f"<{self.numeric_type}{or_null}>"
            answer = (f"Answer as {{{json.dumps(self.name, ensure_ascii=False)}: {placeholder}}}."
                      if self.name is not None else f"Answer with a JSON {placeholder[1:-1]} only.")
            if self.numeric_type == "number":
                answer += " Do not use exponent notation."
            if self.nullable:
                answer += " Return null only if there is no value."
        else:
            lines.append(f"Choices: {json.dumps(dict(self.choices), ensure_ascii=False)}")
            answer = (f'Answer as {{{json.dumps(self.name, ensure_ascii=False)}: "<label>"}}.'
                      if self.name is not None else "Answer with the best label only.")
        lines.append(answer)
        return "\n".join(lines)


@dataclass(frozen=True)
class Generation:
    """What generate() returns, as the HTTP API does: the typed answers, the
    reasoning of each field that thought, and the tokens of the call."""

    result: dict[str, Any]
    thinking: dict[str, str]
    usage: Usage


class TypeLLMClient:
    """Generate typed answers with local SGLang or the hosted API."""

    DEFAULT_LABEL_POOL = tuple(ascii_uppercase + digits)

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        mode: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        timeout: float | None = None,
        label_pool: Sequence[str] | None = None,
        numeric_max_digits: int = 32,
        tokenizer: str | None = None,
        text_max_tokens: int = 128,
    ) -> None:
        """Run on your own SGLang server, by default http://127.0.0.1:30000,
        unless TYPELLM_API_KEY is set (see below).

        With api_key, calls go to the hosted API instead, by default
        https://api.typellm.ai. It compiles and runs the schema itself. Model
        chooses one of its models; temperature, seed and timeout apply, while
        tokenizer, text_max_tokens, numeric_max_digits and label_pool need your
        own server and raise ValueError. A hosted timeout set here is the
        default for generate() calls; without one, the service uses its own
        default.

        Given neither api_key nor base_url, the TYPELLM_API_KEY environment
        variable is used as api_key unless it is blank. Whitespace around either
        key is dropped, and a blank api_key raises ValueError. A base_url alone
        always means your own server.

        temperature 0 (the default) picks the most likely answer; above 0 samples
        at that temperature. mode is deprecated: temperature alone decides.
        """
        mode, temperature = _resolve_decoding(mode, temperature)
        if type(numeric_max_digits) is not int or numeric_max_digits <= 0:
            raise ValueError("numeric_max_digits must be a positive integer")
        if api_key is not None:
            api_key = api_key.strip()
            if not api_key:
                raise ValueError("api_key is empty")
        elif base_url is None:
            api_key = os.environ.get("TYPELLM_API_KEY", "").strip() or None
        self.api_key = api_key
        if api_key is None:
            self.sglang = SGLangClient(
                base_url or "http://127.0.0.1:30000",
                model,
                DEFAULT_SOCKET_TIMEOUT if timeout is None else timeout,
                tokenizer=tokenizer,
                text_max_tokens=text_max_tokens,
                answer_reserve_tokens=numeric_max_digits + 3,
            )
        else:
            local_only = [name for name, value, default in (
                ("tokenizer", tokenizer, None), ("text_max_tokens", text_max_tokens, 128),
                ("numeric_max_digits", numeric_max_digits, 32),
                ("label_pool", label_pool, None)) if value != default]
            if local_only:
                raise ValueError(f"{', '.join(local_only)} can only be set for your own server, "
                                 "a base_url without api_key. The hosted API, chosen by api_key "
                                 "or TYPELLM_API_KEY, sets its own.")
            self.sglang = None
            self.base_url = (base_url or HOSTED_URL).rstrip("/")
            self.model = model
            self.timeout = timeout
            self._transport: httpx.BaseTransport | None = None  # tests put a MockTransport here
        self.mode = mode
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.label_pool = tuple(label_pool or self.DEFAULT_LABEL_POOL)
        self.numeric_max_digits = numeric_max_digits
        self.label_token_map: dict[str, int] = {}
        # The prompts of the last call, for tests; per thread or task, so concurrent
        # generate() calls never see each other's. print_final_prompt shows them.
        self._last_prompts: ContextVar[list[str]] = ContextVar(
            f"typellm_last_prompts_{id(self)}", default=[]
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_last_prompts"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._last_prompts = ContextVar(f"typellm_last_prompts_{id(self)}", default=[])


    def _control_labels(self, count: int) -> list[str]:
        labels: list[str] = []
        token_ids: set[int] = set()
        for label in self.label_pool:
            try:
                token_id, _ = self.sglang.single_token(label)
            except ValueError:
                continue
            if token_id in token_ids:
                continue
            labels.append(label)
            token_ids.add(token_id)
            self.label_token_map[label] = token_id
            if len(labels) == count:
                LOG.info("runtime_label_token_map=%s", self.label_token_map)
                return labels
        raise SchemaError(
            f"Schema needs {count} control labels, but only {len(labels)} unique "
            "single-token labels were found for the current tokenizer"
        )

    def compile_schema(self, schema: Mapping[str, Any]) -> list[Choice]:
        """Compile standard JSON Schema or the original ordered-list format."""
        if self.api_key is not None:
            raise ValueError("compile_schema needs your own server when using api_key")
        if not isinstance(schema, Mapping):
            raise SchemaError("schema must be a mapping")
        if schema.get("type", "object") != "object":
            raise SchemaError("the top-level schema type must be 'object'")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            decisions = compile_json_schema(schema)
            finite_sizes = [len(item.choices) for item in decisions if item.choices]
            labels = self._control_labels(max(finite_sizes)) if finite_sizes else []
            return [
                Choice(
                    question=item.question,
                    choices=dict(zip(labels, item.choices)),
                    name=item.name,
                    syntax=item.syntax,
                    numeric_type=item.numeric_type,
                    text_type=item.text_type,
                    permutations=item.permutations,
                    return_probabilities=item.return_probabilities,
                    depends_on=item.depends_on,
                    nullable=item.nullable,
                    thinking=item.thinking,
                    thinking_budget=item.thinking_budget,
                )
                for item in decisions
            ]
        if not isinstance(properties, list):
            raise SchemaError(
                "schema.properties must be either an ordered JSON Schema object "
                "or a legacy TypeLLM list"
            )
        if not properties:
            raise SchemaError("schema.properties must not be empty")

        normalized: list[tuple[str, str, str, list[Any]]] = []
        names: set[str] = set()
        max_choices = 0
        for index, prop in enumerate(properties):
            if not isinstance(prop, Mapping):
                raise SchemaError(f"properties[{index}] must be a mapping")
            if "depends_on" in prop:
                raise SchemaError("depends_on requires questions or object-form schema.properties")
            name = prop.get("name")
            for old_key in ("question", "x-question"):
                if old_key in prop:
                    raise SchemaError(f"{old_key} is no longer supported; use instructions")
            question = prop.get("instructions")
            kind = prop.get("type")
            if not isinstance(name, str) or not name:
                raise SchemaError(f"properties[{index}].name must be a non-empty string")
            if name in names:
                raise SchemaError(f"duplicate property name: {name!r}")
            names.add(name)
            if not isinstance(question, str) or not question:
                raise SchemaError(
                    f"properties[{index}].instructions must be a non-empty string"
                )
            if kind == "choice":
                values = prop.get("choices")
                if not isinstance(values, list) or not values:
                    raise SchemaError(
                        f"choice property {name!r} requires a non-empty choices list"
                    )
                if any(not isinstance(value, str) or not value for value in values):
                    raise SchemaError(
                        f"all choices for {name!r} must be non-empty strings"
                    )
                if len(set(values)) != len(values):
                    raise SchemaError(f"choices for {name!r} must be unique")
                if len(values) > MAX_ENUM_CHOICES:
                    raise SchemaError(
                        f"choices for {name!r} has {len(values)} values; "
                        f"the maximum is {MAX_ENUM_CHOICES}"
                    )
            elif kind == "bool":
                if "choices" in prop:
                    raise SchemaError(
                        f"bool property {name!r} must not define choices; "
                        "it automatically uses true/false"
                    )
                values = [True, False]
            else:
                raise SchemaError(
                    f"properties[{index}].type must be 'choice' or 'bool', "
                    f"got {kind!r}"
                )
            max_choices = max(max_choices, len(values))
            normalized.append((name, kind, question, values))

        labels = self._control_labels(max_choices)
        return [
            Choice(
                question=question,
                choices=dict(zip(labels, values)),
                name=name,
                syntax=(
                    "Bool"
                    if kind == "bool"
                    else "Choice"
                ),
            )
            for name, kind, question, values in normalized
        ]

    def generate(
        self,
        *,
        context: str | None = None,
        state: str | None = None,
        schema: Mapping[str, Any] | None = None,
        questions: Mapping[str, Any] | None = None,
        images: Sequence[Any] | None = None,
        mode: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        print_final_prompt: bool = False,
    ) -> Generation:
        """Answer every field; fields run in parallel unless they declare depends_on.

        Returns the typed answers in .result, with .thinking and .usage. When a
        local call fails, the exception's .usage holds the tokens it spent.

        A seed fixes this call's own random choices and leaves the client's shared
        stream alone; without one, calls share that stream. The server's numerics
        can still vary with its cache and batching, so a seed does not guarantee
        identical results. timeout caps the call and raises GenerationTimeout.
        In local mode, setting cancel raises GenerationCancelled; timeout and
        cancel stop the call before its next request to SGLang. Hosted mode
        sends timeout to the service and does not support cancel.
        """
        if (context is None) == (state is None):
            raise ValueError("provide exactly one of context or state")
        context = state if state is not None else context
        if not isinstance(context, str):
            raise ValueError("context or state must be a string")
        encoded_images = encode_images(images) if images is not None else ()
        if (schema is None) == (questions is None):
            raise SchemaError("provide exactly one of questions or schema")
        if questions is not None:
            if not isinstance(questions, Mapping):
                raise SchemaError("questions must be a mapping of field names to definitions")
            schema = {"type": "object", "properties": questions}
        if mode is None and temperature is None:
            active_mode, active_temperature = self.mode, self.temperature
        else:
            # A call's own temperature decides for it, whatever the client's mode.
            if temperature is None:
                temperature = self.temperature or None
            active_mode, active_temperature = _resolve_decoding(mode, temperature)
        if self.api_key is not None:
            if questions is None or cancel is not None or print_final_prompt:
                raise ValueError("with api_key, generate() takes questions; schema, cancel and "
                                 "print_final_prompt need your own server")
            return self._generate_hosted(context, questions, encoded_images, active_mode,
                                         active_temperature, seed, timeout)
        rng = self.rng if seed is None else random.Random(seed)
        # The time budget covers compiling too: labels are tokenized by SGLang.
        with call_scope(timeout, cancel) as scope:
            try:
                decisions = self.compile_schema(schema)
                count = getattr(self.sglang, "count_tokens", None)
                if callable(count):
                    scope.usage.input_tokens = sum(map(count, (
                        context, json.dumps(questions if questions is not None else schema, ensure_ascii=False))))
                    scope.unmeasured_images = len(encoded_images)
                # Independent fields run together; depends_on turns the fields into a
                # graph whose layers run in order.
                run = (_execute_dependency_decisions if any(d.depends_on is not None for d in decisions)
                       else _execute_batch_decisions)
                attach = self.sglang.images(encoded_images) if encoded_images else nullcontext()
                with attach:
                    rows, prompts = run(
                        self.sglang, context, decisions, active_mode,
                        active_temperature, rng, self.numeric_max_digits,
                        image_count=len(encoded_images),
                    )
            except BaseException as exc:
                exc.usage = scope.usage  # partial work, for callers that bill it
                raise
        self._last_prompts.set(prompts)
        thinking = {decision.name: row["thinking"] for decision, row in zip(decisions, rows)
                    if row.get("thinking")}

        output: dict[str, Any] = {}
        for decision, row in zip(decisions, rows):
            assert decision.name is not None
            value = row["value"]
            if decision.return_probabilities:
                probabilities = {
                    decision.choices[label]: probability
                    for label, probability in row["probabilities"].items()
                }
                output[decision.name] = {
                    "value": value,
                    "probabilities": probabilities,
                }
            else:
                output[decision.name] = value

        if print_final_prompt:
            _print_final_prompts(prompts)
        return Generation(output, thinking, scope.usage)

    def _generate_hosted(self, context: str, questions: Mapping[str, Any], images: Sequence[str],
                         mode: str, temperature: float, seed: int | None,
                         timeout: float | None) -> Generation:
        """One POST /v1/generate. HTTP errors carry their status in SGLangError."""
        active_timeout = self.timeout if timeout is None else timeout
        if active_timeout is not None and (type(active_timeout) not in (int, float) or
                                           not active_timeout > 0):
            raise ValueError("timeout must be a positive number of seconds or None")
        options: dict[str, Any] = {
            "mode": mode,
            # Without a seed, calls draw theirs from the client's stream, as locally.
            "seed": self.rng.randrange(2**32) if seed is None else seed,
        }
        if mode == "sample":
            options["temperature"] = temperature
        body: dict[str, Any] = {
            "context": context,
            "questions": questions,
            "images": list(images),
            "options": options,
        }
        if self.model is not None:
            body["model"] = self.model
        if active_timeout is not None:
            body["timeout"] = active_timeout  # the service's own default (60 s) otherwise
        try:
            with httpx.Client(transport=self._transport) as http:
                # The service times the call itself. The margin covers its queue, image
                # scaling and the grace it gives its own workers past the timeout.
                response = http.post(self.base_url + "/v1/generate", json=body,
                                     headers={"Authorization": f"Bearer {self.api_key}"},
                                     timeout=(active_timeout if active_timeout is not None
                                              else DEFAULT_SOCKET_TIMEOUT) + 60)
        except httpx.HTTPError as exc:
            raise SGLangError(f"Could not reach the TypeLLM API at {self.base_url}: {exc!r}") from exc
        if response.status_code != 200:
            error = GenerationTimeout if response.status_code == 504 else SGLangError
            raise error(f"TypeLLM API returned HTTP {response.status_code}: {response.text}",
                        status=response.status_code)
        try:
            data = response.json()
            usage = data["usage"]
            result = data["result"]
            thinking = data.get("thinking") or {}
            input_tokens = usage["input_tokens"]
            thinking_tokens = usage["thinking_tokens"]
        except (ValueError, TypeError, KeyError) as exc:
            raise SGLangError("TypeLLM API returned an invalid response",
                              status=response.status_code) from exc
        return Generation(result, thinking, Usage(input_tokens=input_tokens, thinking_tokens=thinking_tokens))


def candidate_softmax(
    logprobs: Mapping[str, float], temperature: float = 1.0
) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    scaled = {label: value / temperature for label, value in logprobs.items()}
    pivot = max(scaled.values())
    weights = {label: math.exp(value - pivot) for label, value in scaled.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def _sample(probs: Mapping[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last = ""
    for label, probability in probs.items():
        last = label
        cumulative += probability
        if threshold <= cumulative:
            return label
    return last


def _resolve_decoding(mode: str | None, temperature: float | None) -> tuple[str, float]:
    """The (mode, temperature) that temperature, and the deprecated mode, ask for.

    temperature 0 picks the most likely answer (argmax); above 0 samples. For
    older code, mode="sample" without a temperature samples at 1.0 and
    mode="argmax" is argmax whatever the temperature.
    """
    if mode is not None:
        warnings.warn("mode is deprecated; temperature 0 picks the most likely answer and "
                      "above 0 samples", DeprecationWarning, stacklevel=3)
        if mode not in {"argmax", "sample"}:
            raise ValueError("mode must be 'argmax' or 'sample'")
        if mode == "sample" and temperature is None:
            temperature = 1.0
    if temperature is None:
        temperature = 0.0
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("temperature must be a finite number >= 0")
    if mode == "argmax":
        return "argmax", float(temperature)
    if mode == "sample" and temperature == 0:
        raise ValueError("temperature must be > 0 in sample mode")
    return ("sample" if temperature > 0 else "argmax"), float(temperature)


def _numeric_text_is_complete(text: str, numeric_type: str) -> bool:
    if not text or text == "-":
        return False
    unsigned = text[1:] if text.startswith("-") else text
    if not unsigned or (
        len(unsigned) > 1
        and unsigned.startswith("0")
        and not unsigned.startswith("0.")
    ):
        return False
    if numeric_type == "integer":
        return unsigned.isdigit()
    if "." not in unsigned:
        return unsigned.isdigit()
    integer, fraction = unsigned.split(".", 1)
    return integer.isdigit() and bool(fraction) and fraction.isdigit()


def _parse_numeric_value(text: str, decision: Choice) -> int | float:
    if not _numeric_text_is_complete(text, decision.numeric_type or ""):
        raise ValueError(f"Generated invalid {decision.numeric_type}: {text!r}")
    value: int | float = (
        int(text) if decision.numeric_type == "integer" else float(text)
    )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Generated non-finite number for {decision.name!r}")
    return value


def numeric_pattern(numeric_type: str, max_digits: int, nullable: bool = False) -> str:
    """The regex a grammar backend decodes a prefilled number with, after '{"name":'.

    It admits JSON-style numbers: an optional sign, no leading zeros, no
    exponent, at most max_digits digits in all; the value ends with the
    object's '}' or the end of the message.
    """
    if numeric_type == "integer":
        body = "(?:0|[1-9][0-9]{0,%d})" % (max_digits - 1)
    else:
        options = []
        for size in range(1, max_digits + 1):
            integer = "[0-9]" if size == 1 else "[1-9][0-9]{%d}" % (size - 1)
            room = max_digits - size
            options.append(integer + (r"(?:\.[0-9]{1,%d})?" % room if room else ""))
        body = "(?:" + "|".join(options) + ")"
    value = "-?" + body
    if nullable:
        value = f"(?:{value}|null)"
    return " ?" + value + r"\}?"


def _parse_numbers(
    items: Sequence[tuple[str, Choice]], texts: Sequence[str],
) -> list[tuple[int | float | None, str, str]]:
    """Read the grammar-generated numbers: (value, closed prompt, value text)."""
    outputs = []
    for (prompt, decision), raw in zip(items, texts):
        text = raw.strip()
        if text.endswith("}"):
            text = text[:-1].rstrip()
        if decision.nullable and text == "null":
            outputs.append((None, prompt + " null}", "null"))
            continue
        value = _parse_numeric_value(text, decision)
        LOG.info("numeric name=%s grammar_text=%r value=%r", decision.name, raw, value)
        outputs.append((value, prompt + " " + text + "}", text))
    return outputs


def _balanced_orders(count: int) -> list[tuple[int, ...]]:
    """A balanced Latin square (Williams design) on positions 0..count-1.

    Every item takes every position equally often and follows every other item
    equally often: count orders when count is even, 2*count when it is odd.
    """
    first = [0]
    low, high = 1, count - 1
    while len(first) < count:
        first.append(low)
        low += 1
        if len(first) < count:
            first.append(high)
            high -= 1
    rows = [tuple((item + shift) % count for item in first) for shift in range(count)]
    if count % 2:
        rows += [row[::-1] for row in rows]
    return list(dict.fromkeys(rows))


def _choice_orderings(decision, rng):
    """Rebind values to fixed control labels, sampling ranks without enumeration."""
    labels = list(decision.choices)
    values = list(decision.choices.values())
    if decision.permutations == "auto" and len(values) > 1:
        # Start from a canonical order so the result does not depend on the
        # order the enum was written in, then balance positions and neighbours.
        canonical = sorted(range(len(values)), key=lambda i: json.dumps(values[i]))
        orders = [tuple(canonical[i] for i in row) for row in _balanced_orders(len(values))]
        return [(replace(decision, choices=dict(zip(labels, (values[i] for i in order))),
                         permutations=1), order) for order in orders]
    total = math.factorial(len(values))
    count = total if decision.permutations in ("all", "auto") else min(decision.permutations, total)
    if count == 1:
        return [(decision, tuple(range(len(values))))]
    if count == total:
        orders = all_permutations(range(len(values)))
    else:
        # Rejection sampling of integer ranks avoids materializing K! orders,
        # including when K! exceeds the platform's range length limit.
        ranks = set()
        ordered_ranks = []
        while len(ranks) < count:
            rank = rng.randrange(total)
            if rank not in ranks:
                ranks.add(rank)
                ordered_ranks.append(rank)
        orders = []
        for rank in ordered_ranks:
            available = list(range(len(values)))
            order = []
            while available:
                index, rank = divmod(rank, math.factorial(len(available) - 1))
                order.append(available.pop(index))
            orders.append(tuple(order))
    return [(replace(decision, choices=dict(zip(labels, (values[i] for i in order))),
                     permutations=1), order) for order in orders]


def _mean_order_probabilities(scored, orders, label_tokens, temperature):
    labels = list(label_tokens)
    aligned = {label: [] for label in labels}
    for (by_id, _meta), (_variant, order) in zip(scored, orders):
        probs = candidate_softmax({label: by_id[token_id]
                                  for label, (token_id, _) in label_tokens.items()}, temperature)
        for position, original_index in enumerate(order):
            aligned[labels[original_index]].append(probs[labels[position]])
    return {label: math.fsum(values) / len(scored) for label, values in aligned.items()}


def _execute_dependency_decisions(
    client, context, decisions, mode, temperature, rng, numeric_max_digits,
    image_count=0,
):
    rows_by_name = {}
    prompts_by_name = {}
    ancestors = {}
    for layer in dependency_layers(decisions):
        dependency_values = {}
        parent_prefixes = {}
        for decision in layer:
            visible = set(decision.depends_on or ())
            for name in decision.depends_on or ():
                visible.update(ancestors[name])
            ancestors[decision.name] = visible
            parents = decision.depends_on or ()
            if parents:
                # Longest serialized prefix is a deterministic heuristic; KV
                # from distinct branches cannot be concatenated.
                parent = max(parents, key=lambda name: len(prompts_by_name[name]))
                parent_prefixes[decision.name] = prompts_by_name[parent]
            dependency_values[decision.name] = {
                d.name: rows_by_name[d.name]["value"]
                for d in decisions if d.name in visible
            }
        rows, prompts = _execute_batch_decisions(
            client, context, layer, mode, temperature, rng, numeric_max_digits,
            dependency_values=dependency_values,
            parent_prefixes=parent_prefixes,
            image_count=image_count,
        )
        for decision, row, prompt in zip(layer, rows, prompts):
            rows_by_name[decision.name] = row
            prompts_by_name[decision.name] = prompt
    return ([rows_by_name[d.name] for d in decisions],
            [prompts_by_name[d.name] for d in decisions])


def _execute_batch_decisions(
    client: SGLangClient,
    context: str,
    decisions: Sequence[Choice],
    mode: str,
    temperature: float,
    rng: random.Random,
    numeric_max_digits: int = 32,
    dependency_values: Mapping[str, Mapping[str, Any]] | None = None,
    parent_prefixes: Mapping[str, str] | None = None,
    image_count: int = 0,
) -> tuple[list[dict], list[str]]:
    def question_content(decision):
        content = decision.opening_text()
        values = (dependency_values or {}).get(decision.name, {})
        if values:
            content = "Dependency results (JSON):\n" + json.dumps(values, ensure_ascii=False) + "\n\n" + content
        return content

    incremental = parent_prefixes is not None

    def complete(prompt, messages, answer):
        if incremental:
            return client.complete_chat_prefix(prompt, answer)
        return client.render_chat(
            messages + [{"role": "assistant", "content": answer}],
            add_generation_prompt=False,
        )

    shared_messages = [{"role": "user", "content": _user_content(context.rstrip(), image_count)}]
    shared_prefix = client.render_chat(
        shared_messages, add_generation_prompt=False
    )
    label_tokens_by_decision: list[dict[str, tuple[int, str]]] = []
    candidate_ids: list[list[int]] = []
    prompts: list[str] = []
    scoring_prompts = []
    scoring_ids = []
    ordering_groups = []

    # Warm each distinct parent once before siblings, including thinking/text
    # requests. Root context is warmed only in the first DAG layer.
    if incremental:
        prefixes = list(dict.fromkeys(parent_prefixes.values()))
        if any(d.name not in parent_prefixes for d in decisions):
            prefixes.insert(0, shared_prefix)
        for prefix in prefixes:
            client.cache_prefix(prefix)

    finite_indexes: list[int] = []
    open_results: dict[int, tuple[dict, str]] = {}
    text_pending = []
    numeric_pending = []
    # Clients that can batch thinking get every prompt of the layer, including
    # permutation variants, before any reasoning runs.
    defer = callable(getattr(client, "prepare_answer_prefixes", None))
    raw_prompts: list[str] = []

    raw_thinking: list[bool | None] = []
    raw_budgets: list[int | None] = []

    def generation_prompt(parent, content, messages, decision):
        # A field's own thinking settings are passed only when it has them, so
        # clients without per-prompt thinking keep working.
        own = {key: value for key, value in (("thinking", decision.thinking),
                                             ("thinking_budget", decision.thinking_budget))
               if value is not None}
        if parent is not None:
            prompt = (client.extend_chat_prefix(parent, content, finish_thinking=False, **own)
                      if defer else client.extend_chat_prefix(parent, content, **own))
        else:
            prompt = (client.render_chat(messages, add_generation_prompt=True, finish_thinking=False, **own)
                      if defer else client.render_chat(messages, add_generation_prompt=True, **own))
        raw_prompts.append(prompt)
        raw_thinking.append(decision.thinking)
        raw_budgets.append(decision.thinking_budget)
        return len(raw_prompts) - 1

    decision_slots = {}
    scoring_slots = []
    scoring_prefills = []
    for index, decision in enumerate(decisions):
        messages = shared_messages + [
            {"role": "user", "content": question_content(decision)}
        ]
        parent = (parent_prefixes or {}).get(decision.name)
        decision_slots[index] = generation_prompt(parent, question_content(decision), messages, decision)
        if decision.text_type:
            text_pending.append((index, decision, messages))
            continue
        if decision.numeric_type is not None:
            numeric_pending.append((index, decision, messages))
            continue
        label_tokens = {
            label: client.single_token(label) for label in decision.choices
        }
        ids = [token_id for token_id, _ in label_tokens.values()]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Decision {index} has labels with duplicate token IDs: {ids}")
        label_tokens_by_decision.append(label_tokens)
        candidate_ids.append(ids)
        orders = _choice_orderings(decision, rng)
        ordering_groups.append(orders)
        for variant, _order in orders:
            variant_content = question_content(variant)
            scoring_slots.append(
                decision_slots[index] if variant.choices == decision.choices else
                generation_prompt(parent, variant_content,
                                  shared_messages + [{"role": "user", "content": variant_content}], variant))
            scoring_ids.append(ids)
            scoring_prefills.append(decision.label_prefill)
        finite_indexes.append(index)
        LOG.info(
            "batch_decision=%d name=%s candidate_token_ids=%s",
            index,
            decision.name,
            {label: token_id for label, (token_id, _) in label_tokens.items()},
        )

    # Warm the exact common prefix once, before any batch that shares it: prompts
    # prefilled in the same batch cannot reuse each other's cache, so thinking,
    # number and text batches would each prefill the context once per prompt.
    # SGLang then forks the cached state; TypeLLM never reads or moves KV tensors.
    if finite_indexes and not incremental:
        cache_meta = client.cache_prefix(shared_prefix)
        LOG.info("batch_shared_prefix_cached_tokens=%s", cache_meta.get("cached_tokens"))

    per_field = any(t is not None for t in raw_thinking) or any(b is not None for b in raw_budgets)
    ready = (client.prepare_answer_prefixes(raw_prompts, **({"thinking": raw_thinking, "budgets": raw_budgets}
                                                           if per_field else {}))
             if defer else raw_prompts)
    prompts = [ready[decision_slots[index]] for index in finite_indexes]
    scoring_prompts = [ready[slot] + prefill for slot, prefill in zip(scoring_slots, scoring_prefills)]
    # Open fields continue from {"name": ; history gets the closed object.
    def open_row(decision, value):
        return {"name": decision.name, "question": decision.question,
                "label": None, "value": value, "probabilities": None}

    open_pending = [(index, decision, messages, ready[decision_slots[index]])
                    for index, decision, messages in numeric_pending + text_pending]
    numeric_pending = [item for item in open_pending if item[1].numeric_type is not None]
    text_pending = [item for item in open_pending if item[1].text_type]

    # A layer's numbers and strings decode side by side in one request.
    if numeric_pending or text_pending:
        number_items = [(prompt + decision.answer_prefill, decision)
                        for _, decision, _, prompt in numeric_pending]
        number_seed = rng.randrange(2**31) if numeric_pending else 0
        text_seed = rng.randrange(2**31) if text_pending else 0
        raw_numbers, values = client.generate_fields(
            [prompt for prompt, _ in number_items],
            [numeric_pattern(d.numeric_type or "", numeric_max_digits, d.nullable) for _, d in number_items],
            # Digits, sign, point and the closing brace; tokens hold one or more characters.
            numeric_max_digits + 4,
            # From {"name": the model writes the string's first token, quote included.
            [prompt + decision.answer_prefill for _, decision, _, prompt in text_pending],
            temperature=0 if mode == "argmax" else temperature,
            number_seed=number_seed,
            text_seed=text_seed,
            after_key=all(decision.answer_prefill for _, decision, _, _ in text_pending),
            # A nullable string writes null or its text in the same request.
            nullable=[decision.nullable for _, decision, _, _ in text_pending],
        )
        for (index, decision, messages, prompt), (value, _completed, generated_text) in zip(
                numeric_pending, _parse_numbers(number_items, raw_numbers)):
            open_results[index] = (open_row(decision, value),
                                   complete(prompt, messages, _closed_answer(decision, generated_text)))
        for (index, decision, messages, prompt), value in zip(text_pending, values):
            completed = complete(prompt, messages, _closed_answer(decision, json.dumps(value, ensure_ascii=False)))
            open_results[index] = (open_row(decision, value), completed)

    if prompts:
        scored, elapsed = client.score_candidates_batch(scoring_prompts, scoring_ids)
        grouped_scores = []
        offset = 0
        for orders in ordering_groups:
            grouped_scores.append(scored[offset:offset + len(orders)])
            offset += len(orders)
        scored = [group[0] for group in grouped_scores]
    else:
        scored, elapsed = [], 0.0

    results: list[dict] = []
    completed_prompts: list[str] = []
    for finite_index, (decision_index, label_tokens, (by_id, meta)) in enumerate(
        zip(finite_indexes, label_tokens_by_decision, scored)
    ):
        index = decision_index
        decision = decisions[index]
        raw = {
            label: by_id[token_id]
            for label, (token_id, _) in label_tokens.items()
        }
        probability_temperature = temperature if mode == "sample" else 1.0
        probabilities = _mean_order_probabilities(
            grouped_scores[finite_index], ordering_groups[finite_index],
            label_tokens, probability_temperature)
        selected = (
            max(probabilities, key=probabilities.__getitem__)
            if mode == "argmax"
            else _sample(probabilities, rng)
        )
        selected_text = label_tokens[selected][1]
        semantic_value = decision.choices[selected]
        completed_prompts.append(
            complete(
                prompts[finite_index],
                shared_messages + [{"role": "user", "content": question_content(decision)}],
                _closed_label(decision, selected_text),
            )
        )
        LOG.info("batch_decision=%d raw_candidate_logprobs=%s", index, raw)
        LOG.info(
            "batch_decision=%d renormalized_probabilities=%s", index, probabilities
        )
        LOG.info(
            "batch_decision=%d selected=%s value=%s batch_elapsed=%.4fs "
            "cached_tokens=%s",
            index,
            selected,
            semantic_value,
            elapsed,
            meta.get("cached_tokens"),
        )
        results.append(
            {
                "name": decision.name,
                "question": decision.question,
                "label": selected,
                "value": semantic_value,
                "probabilities": probabilities,
            }
        )

    if open_results:
        finite_rows = iter(zip(results, completed_prompts))

        merged_results: list[dict] = []
        merged_prompts: list[str] = []
        for index in range(len(decisions)):
            if index in open_results:
                row, prompt = open_results[index]
            else:
                row, prompt = next(finite_rows)
            merged_results.append(row)
            merged_prompts.append(prompt)
        results, completed_prompts = merged_results, merged_prompts
    # Each field's reasoning, when it thought: what finishing thinking added.
    read = getattr(client, "thinking_text", None)
    for index, row in enumerate(results):
        slot = decision_slots[index]
        row["thinking"] = read(raw_prompts[slot], ready[slot]) if callable(read) and defer else None
    return results, completed_prompts


def _print_final_prompts(prefixes: Sequence[str]) -> None:
    for index, prefix in enumerate(prefixes):
        print(f"\n===== FINAL BATCH PROMPT {index} =====")
        print(prefix)
        print(f"===== END BATCH PROMPT {index} =====")


def run_schema(
    context: str | None = None,
    schema: Mapping[str, Any] | None = None,
    mode: str | None = None,
    temperature: float | None = None,
    *,
    state: str | None = None,
    questions: Mapping[str, Any] | None = None,
    images: Sequence[Any] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    numeric_max_digits: int = 32,
    tokenizer: str | None = None,
    text_max_tokens: int = 128,
    print_final_prompt: bool = False,
) -> Generation:
    client = TypeLLMClient(
        base_url or os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
        model or os.environ.get("SGLANG_MODEL"),
        mode=mode,
        temperature=temperature,
        seed=seed,
        numeric_max_digits=numeric_max_digits,
        tokenizer=tokenizer,
        text_max_tokens=text_max_tokens,
    )
    return client.generate(
        context=context,
        state=state,
        schema=schema,
        questions=questions,
        images=images,
        print_final_prompt=print_final_prompt,
    )
