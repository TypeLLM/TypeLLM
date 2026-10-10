"""Schema binding and constrained decision execution."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
import warnings
from contextlib import nullcontext
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from itertools import permutations as all_permutations
from string import ascii_uppercase, digits
from typing import Any, Mapping, Sequence

import httpx

from .schema import (
    CONTINUE_FIELD,
    CONTINUE_QUESTION,
    MAX_ARRAY_ITEMS,
    MAX_ENUM_CHOICES,
    OBJECT_ITEM_SCOPE,
    THINKING_EFFORTS,
    ArrayField,
    SchemaError,
    compile_json_schema,
    condition_met,
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


def _closed_value(decision: "Choice", value: Any) -> str:
    """A choice answered by its value, as history keeps it, e.g. {"choice": "billing"}."""
    return decision.label_prefill + " " + json.dumps(value, ensure_ascii=False) + "}"


def _value_continuation(value: Any) -> str:
    """What follows {"choice": for a value: the scorer scores these, one per choice."""
    return " " + json.dumps(value, ensure_ascii=False) + "}"


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
    effort_from: str | None = None
    effort_for: str | None = None
    when: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] | None = None
    # Where the answer goes in the result, ("person", "age"); empty means (name,).
    path: tuple[str, ...] = ()
    # The name the prompt shows ("age" for "person.age"); None shows name.
    shown_name: str | None = None
    # Lines the prompt shows before the field's own, such as the object it belongs to.
    scope: str = ""
    # depends_on grouped by what was named: a group is skipped only when all of it is.
    dependency_groups: tuple[tuple[str, ...], ...] | None = None
    # A choice's description by value, from "choices"; empty for an enum.
    descriptions: tuple[tuple[Any, str], ...] = ()
    # A score's levels, lowest first; its value is the weighted average of their indices.
    levels: tuple[str, ...] = ()
    # More choices than labels: the prompt lists values, the answer writes one, and the client's
    # choice scorer gives each its probability. choices is keyed "0", "1", ... in enum order.
    by_value: bool = False

    def __post_init__(self) -> None:
        if not self.choices and self.numeric_type is None and not self.text_type:
            raise ValueError("Choice.choices must not be empty")
        if self.text_type and (self.choices or self.numeric_type is not None):
            raise ValueError("Text choices cannot have enum values or a numeric type")
        if self.numeric_type not in {None, "integer", "number"}:
            raise ValueError("numeric_type must be None, 'integer', or 'number'")
        if self.numeric_type is not None and self.choices:
            raise ValueError("Open numeric choices must be empty")
        if len(self.choices) > MAX_ENUM_CHOICES and not self.by_value:
            raise ValueError(
                f"Choice has {len(self.choices)} values; "
                f"the maximum is {MAX_ENUM_CHOICES}"
            )
        # One value or one level leaves nothing to choose, and no confidence to compute.
        if len(self.choices) == 1 or len(self.levels) == 1:
            raise ValueError("Choice needs at least 2 values")
        if any(not label for label in self.choices):
            raise ValueError("Choice labels must be non-empty strings")

    @property
    def field_name(self) -> str | None:
        """The name the prompt and the answer use."""
        return self.shown_name if self.shown_name is not None else self.name

    @property
    def answer_prefill(self) -> str:
        """Start of the answer for open fields; the model continues with the value.

        Chat models tend to answer {"name": value}, so the value is decoded right
        where they would write it.
        """
        if self.field_name is None or not (self.text_type or self.numeric_type is not None):
            return ""
        # No trailing space: in {"name": 12} the space belongs to the value's first token.
        return "{" + json.dumps(self.field_name, ensure_ascii=False) + ":"

    @property
    def label_prefill(self) -> str:
        """Start of a choice answer, {"label": "; the next token is the label. Not '{"type": "', after which
        a model writes the field's value, "C", which may be another choice's label."""
        if self.text_type or self.numeric_type is not None:
            return ""
        # Up to the colon: the value's tokens, from the space on, do not depend on the prompt.
        return '{"choice":' if self.by_value else '{"label": "'

    def opening_text(self) -> str:
        """The field's prompt: the same Field / Type / Instructions / Answer lines for every type,
        after the lines of its scope (the object it belongs to, the array item it is part of)."""
        name = self.field_name
        lines = self.scope.splitlines()
        if name is not None:
            lines.append(f"Field: {json.dumps(name, ensure_ascii=False)}")
        # A nullable field says "or null" in both its type and its answer: a bare
        # <number> reads as "a number is required" and pulls absent values to 0.
        or_null = " or null" if self.nullable else ""
        if self.text_type:
            kind = f"string{or_null}"
        elif self.numeric_type is not None:
            kind = self.numeric_type + or_null
        elif self.levels:
            kind = "score, from the lowest level to the highest"
        else:
            kind = "boolean" if self.syntax == "Bool" else "choice"
        lines.append(f"Type: {kind}")
        lines.append(f"Instructions: {self.question}")
        if self.text_type or self.numeric_type is not None:
            placeholder = "<string" + or_null + ">" if self.text_type else f"<{self.numeric_type}{or_null}>"
            answer = (f"Answer as {{{json.dumps(name, ensure_ascii=False)}: {placeholder}}}."
                      if name is not None else f"Answer with a JSON {placeholder[1:-1]} only.")
            if self.numeric_type == "number":
                answer += " Do not use exponent notation."
            if self.nullable:
                answer += " Return null only if there is no value."
        else:
            if self.by_value:
                said = {_value_key(value): text for value, text in self.descriptions}
                lines.append("Choices (value, description):" if self.descriptions else "Choices:")
                lines += [json.dumps(value, ensure_ascii=False)
                          + (f" ({said[_value_key(value)]})" if _value_key(value) in said else "")
                          for value in self.choices.values()]
                lines.append('Answer as {"choice": <value>}, with one of the values above as written.')
                return "\n".join(lines)
            # Say which side is the label, and answer under "label", not the field's name: after
            # '{"type": "' a model writes the field's value, such as "C", which may be another choice's label.
            if self.descriptions:
                # One line a choice, its description after its value, in the order the labels have now.
                said = {_value_key(value): text for value, text in self.descriptions}
                lines.append("Choices (label: value, description):")
                lines += [f"{label}: {json.dumps(value, ensure_ascii=False)}"
                          + (f" ({said[_value_key(value)]})" if _value_key(value) in said else "")
                          for label, value in self.choices.items()]
            else:
                lines.append(f"Choices (label: value): {json.dumps(dict(self.choices), ensure_ascii=False)}")
            answer = 'Answer as {"label": "<label>"}.'
        lines.append(answer)
        return "\n".join(lines)


@dataclass(frozen=True)
class ArrayChoice:
    """A runtime array: its item decisions with their labels bound, and its hidden continue question."""

    name: str
    question: str
    items: tuple[Choice, ...]
    item_object: bool
    continue_choice: Choice
    min_items: int = 0
    # A scalar array's nullable items past minItems, null ending it; None asks continue_choice instead.
    open_items: tuple[Choice, ...] | None = None
    max_items: int | None = None
    depends_on: tuple[str, ...] | None = None
    when: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] | None = None
    dependency_groups: tuple[tuple[str, ...], ...] | None = None
    effort_for: None = None
    effort_from: None = None
    return_probabilities: bool = False
    # Items the caller already has: the array starts from them (schema's continue_from).
    continue_from: tuple[Any, ...] = ()

    @property
    def path(self) -> tuple[str, ...]:
        return (self.name,)


def _path(decision: Any) -> tuple[str, ...]:
    return decision.path or (decision.name,)


def _put(output: dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Set a value at its path, making the objects on the way in the order they are first set."""
    for key in path[:-1]:
        output = output.setdefault(key, {})
    output[path[-1]] = value


def _canonical_json(value: Any) -> str:
    """The JSON an array's state commits: compact, keys in schema order, Unicode as is."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Generation:
    """What generate() returns, as the HTTP API does: the typed answers, the
    reasoning of each field that thought, the tokens of the call, the effort
    each thinking: "auto" field was given, and the fields a "when" skipped,
    which have no answer in result."""

    result: dict[str, Any]
    thinking: dict[str, str]
    usage: Usage
    thinking_effort: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


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
        max_retries: int = 2,
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

        A hosted call that gets HTTP 429, a 5xx other than 504 or a connection
        error is retried up to max_retries times. Hosted calls share connections
        kept open between them; close(), or leaving a with block, closes them.
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
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
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
            self.max_retries = max_retries
        # The hosted API's connections: opened by the first call, kept for the next.
        self._http_client: httpx.Client | None = None
        self._http_lock = threading.Lock()
        self.mode = mode
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.label_pool = tuple(label_pool or self.DEFAULT_LABEL_POOL)
        self.numeric_max_digits = numeric_max_digits
        # A field's enum may hold more values than labels once a choice scorer is set.
        self.max_choices = MAX_ENUM_CHOICES
        self.label_token_map: dict[str, int] = {}
        # The prompts of the last call, for tests; per thread or task, so concurrent
        # generate() calls never see each other's. print_final_prompt shows them.
        self._last_prompts: ContextVar[list[str]] = ContextVar(
            f"typellm_last_prompts_{id(self)}", default=[]
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_last_prompts"]
        del state["_http_lock"]
        state["_http_client"] = None  # Each copy opens its own connections.
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._last_prompts = ContextVar(f"typellm_last_prompts_{id(self)}", default=[])
        self._http_lock = threading.Lock()

    def close(self) -> None:
        """Close the connections to the hosted API or to SGLang; a later call reopens them."""
        with self._http_lock:
            http_client, self._http_client = self._http_client, None
        if http_client is not None:
            http_client.close()
        if self.sglang is not None:
            self.sglang.close()

    def __enter__(self) -> "TypeLLMClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


    def use_choice_scorer(self, scorer: Any) -> None:
        """Let a field's enum hold more values than there are labels, up to scorer.max_choices.

        Such a field lists its values and is answered by writing one. scorer.score(sglang, prompts,
        continuations) gets, for each prompt, the text that follows it for each choice, and returns
        each choice's probability, summing to 1 for each prompt.
        """
        if self.sglang is None:
            raise ValueError("a choice scorer needs your own server")
        self.sglang.choice_scorer = scorer
        self.max_choices = max(int(scorer.max_choices), MAX_ENUM_CHOICES)

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

    def compile_schema(self, schema: Mapping[str, Any]) -> list[Choice | ArrayChoice]:
        """Compile standard JSON Schema or the original ordered-list format.

        An object's properties are Choices named by their path ("person.age"); an array is an
        ArrayChoice.
        """
        if self.api_key is not None:
            raise ValueError("compile_schema needs your own server when using api_key")
        if not isinstance(schema, Mapping):
            raise SchemaError("schema must be a mapping")
        if schema.get("type", "object") != "object":
            raise SchemaError("the top-level schema type must be 'object'")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            decisions = compile_json_schema(schema, max_choices=self.max_choices)
            arrays = [item for item in decisions if isinstance(item, ArrayField)]
            scalars = [item for item in decisions if not isinstance(item, ArrayField)]
            scalars += [item for array in arrays for item in array.items + (array.open_items or ())]
            # An array's continue question is a boolean: two labels.
            finite_sizes = ([len(item.choices) for item in scalars if 0 < len(item.choices) <= MAX_ENUM_CHOICES]
                            + [2] * bool(arrays))
            labels = self._control_labels(max(finite_sizes)) if finite_sizes else []

            def bind(item):
                by_value = len(item.choices) > MAX_ENUM_CHOICES
                return Choice(
                    question=item.question,
                    choices=({str(i): value for i, value in enumerate(item.choices)} if by_value
                             else dict(zip(labels, item.choices))),
                    by_value=by_value,
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
                    effort_from=item.effort_from,
                    effort_for=item.effort_for,
                    when=item.when,
                    path=item.path,
                    shown_name=item.shown_name,
                    scope=item.scope,
                    dependency_groups=item.dependency_groups,
                    descriptions=item.descriptions,
                    levels=item.levels,
                )

            def bind_array(array):
                return ArrayChoice(
                    name=array.name,
                    question=array.question,
                    items=tuple(bind(item) for item in array.items),
                    item_object=array.item_object,
                    continue_choice=Choice(question=CONTINUE_QUESTION, choices=dict(zip(labels, (True, False))),
                                           name=CONTINUE_FIELD, syntax="Bool", thinking=False),
                    min_items=array.min_items,
                    max_items=array.max_items,
                    depends_on=array.depends_on,
                    when=array.when,
                    dependency_groups=array.dependency_groups,
                    open_items=None if array.open_items is None else tuple(bind(item) for item in array.open_items),
                    continue_from=array.continue_from,
                )

            return [bind_array(item) if isinstance(item, ArrayField) else bind(item) for item in decisions]
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
                # graph whose layers run in order, and so do arrays, which take turns of their own.
                run = (_execute_dependency_decisions
                       if any(d.depends_on is not None or isinstance(d, ArrayChoice) for d in decisions)
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
        # A field a "when" skipped has no row and no prompt; an array's prompts are its row's.
        self._last_prompts.set([text for decision, row, prompt in zip(decisions, rows, prompts)
                                for text in (row["prompts"] if isinstance(decision, ArrayChoice) and row
                                             else [prompt]) if text is not None])
        output, thinking, effort, skipped = _assemble(decisions, rows)

        if print_final_prompt:
            _print_final_prompts(self._last_prompts.get())
        return Generation(output, thinking, scope.usage, effort, skipped)

    def _generate_hosted(self, context: str, questions: Mapping[str, Any], images: Sequence[str],
                         mode: str, temperature: float, seed: int | None,
                         timeout: float | None) -> Generation:
        """Run the call on the hosted API. HTTP errors carry their status in SGLangError."""
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
        # The service times the call itself. The margin covers its queue, image
        # scaling and the grace it gives its own workers past the timeout.
        response = self._post_hosted(body, (active_timeout if active_timeout is not None
                                            else DEFAULT_SOCKET_TIMEOUT) + 60)
        if response.status_code != 200:
            error = GenerationTimeout if response.status_code == 504 else SGLangError
            raise error(f"TypeLLM API returned HTTP {response.status_code}: {response.text}",
                        status=response.status_code)
        try:
            data = response.json()
            usage = data["usage"]
            result = data["result"]
            thinking = data.get("thinking") or {}
            effort = data.get("thinking_effort") or {}
            skipped = data.get("skipped") or []
            input_tokens = usage["input_tokens"]
            thinking_tokens = usage["thinking_tokens"]
            # JSON keys are strings: key probabilities by each field's values again, as locally.
            # The service writes booleans as JSON does and other values as str() does, so an
            # enum holding both "None" and null gets one entry.
            for name, question in questions.items():
                answer = result.get(name)
                if isinstance(answer, dict) and "probabilities" in answer:  # a return_probabilities answer
                    given = question.get("enum") or [choice.get("value") for choice in question.get("choices") or []]
                    values = {json.dumps(value) if type(value) is bool else str(value): value
                              for value in given or (True, False, None)}
                    answer["probabilities"] = {values.get(key, key): probability
                                               for key, probability in answer["probabilities"].items()}
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise SGLangError("TypeLLM API returned an invalid response",
                              status=response.status_code) from exc
        return Generation(result, thinking, Usage(input_tokens=input_tokens, thinking_tokens=thinking_tokens),
                          effort, skipped)

    def _http(self) -> httpx.Client:
        with self._http_lock:
            if self._http_client is None:
                # One client for every call: making one loads the CA bundle (tens of ms of CPU),
                # and a new connection costs a TCP and a TLS handshake. A connection idle for a
                # minute is dropped at the next call, long before the load balancer drops it
                # (610 s). No caps on how many: the service limits concurrent calls itself.
                self._http_client = httpx.Client(limits=httpx.Limits(keepalive_expiry=60))
            return self._http_client

    def _post_hosted(self, body: dict[str, Any], timeout: float) -> httpx.Response:
        """POST /v1/generate, retrying a 429, a 5xx other than 504 or a connection error.

        A timeout, the socket's or the service's 504, is final: the call has had its time.
        """
        http = self._http()
        attempt = 0
        while True:
            try:
                response = http.post(self.base_url + "/v1/generate", json=body, timeout=timeout,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as exc:
                if attempt == self.max_retries or isinstance(exc, httpx.TimeoutException):
                    raise SGLangError(
                        f"Could not reach the TypeLLM API at {self.base_url}: {exc!r}") from exc
                problem, retry_after = repr(exc), None
            else:
                status = response.status_code
                if attempt == self.max_retries or not (status == 429 or (status >= 500 and status != 504)):
                    return response
                problem, retry_after = f"HTTP {status}", response.headers.get("Retry-After")
            try:  # the service's Retry-After, up to a minute
                delay = float(retry_after or 0)
            except ValueError:
                delay = 0.0
            if not 0 < delay <= 60:  # else 0.5 s, 1 s, 2 s ... 8 s, less up to 25% jitter
                delay = 0.5 * 2 ** min(attempt, 4) * (1 - 0.25 * random.random())
            LOG.info("TypeLLM API call failed (%s); retrying in %.2f s", problem, delay)
            time.sleep(delay)
            attempt += 1


def _level_probabilities(decision: Choice, probabilities: Mapping[str, float]) -> list[float]:
    """A score's probabilities by level index, from its probabilities by label."""
    by_level = {decision.choices[label]: probability for label, probability in probabilities.items()}
    return [by_level.get(level, 0.0) for level in decision.levels]


def _score_value(decision: Choice, probabilities: Mapping[str, float]) -> float:
    """A score: the probability-weighted average of its levels' indices, 0 to n - 1."""
    return sum(index * p for index, p in enumerate(_level_probabilities(decision, probabilities)))


def _confidence(decision: Choice, probabilities: Mapping[str, float]) -> float:
    """How sure an answer is, 0 for an even spread and 1 for certainty.

    A choice: (p_max - 1/n) / (1 - 1/n), how far its top probability sits above an even split.
    A score: 1 - (expected distance from the most likely level) / (the same for an even spread,
    from the middle level), floored at 0: probability on a neighbouring level costs less than on a
    far one.
    """
    if decision.levels:
        levels = _level_probabilities(decision, probabilities)
        n = len(levels)
        top = max(range(n), key=lambda index: levels[index])
        spread = sum(p * abs(index - top) for index, p in enumerate(levels))
        even = sum(abs(index - (n - 1) / 2) for index in range(n)) / n
        return max(0.0, 1 - spread / even)
    n = len(probabilities)
    return (max(probabilities.values()) - 1 / n) / (1 - 1 / n)


def _answer(decision: Choice, row: Mapping[str, Any]) -> Any:
    """A row's value as the result gives it: with its probabilities and confidence when the field asks
    for them."""
    if not decision.return_probabilities:
        return row["value"]
    return {"value": row["value"],
            "probabilities": {decision.choices[label]: probability
                              for label, probability in row["probabilities"].items()},
            "confidence": _confidence(decision, row["probabilities"])}


def _assemble(decisions: Sequence[Any], rows: Sequence[dict | None], prefix: str = ""
              ) -> tuple[dict[str, Any], dict[str, str], dict[str, str], list[str]]:
    """The result, each field's reasoning, each "auto" field's effort and the skipped fields.

    Values go to their paths, so an object's properties build it in schema order. A skipped
    object is named once; a property a "when" skipped is named by its path. prefix names an
    array item ("skills[0]") for the reasoning, efforts and skips inside it.
    """
    def label(path):
        return prefix + ("." if prefix else "") + ".".join(path) if path else prefix

    output: dict[str, Any] = {}
    for decision, row in zip(decisions, rows):
        if decision.effort_for is None and row is not None:  # not a hidden effort question, not skipped
            _put(output, _path(decision), row["value"] if isinstance(decision, ArrayChoice)
                 else _answer(decision, row))
    thinking: dict[str, str] = {}
    effort: dict[str, str] = {}
    skipped: list[str] = []
    for decision, row in zip(decisions, rows):
        path = _path(decision)
        if decision.effort_for is not None:
            if row is not None:
                effort[label(path[:-1])] = row["value"]
        elif row is None:
            name = label(path[:1]) if len(path) > 1 and path[0] not in output else label(path)
            if name not in skipped:
                skipped.append(name)
        elif isinstance(decision, ArrayChoice):
            thinking.update(row["thinking"])
            effort.update(row["thinking_effort"])
            skipped.extend(row["skipped"])
        elif row.get("thinking"):
            thinking[label(path)] = row["thinking"]
    return output, thinking, effort, skipped


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


# "auto" balances every order position for up to this many choices, and takes this many orders past it.
MAX_AUTO_ORDERS = 8
# A choice answered by value is scored in its enum's order under "auto", and in at most this many
# orders when it asks for more: the first sorted, the rest shuffled.
MAX_VALUE_ORDERS = 3


def _value_orderings(decision, rng):
    """The orders a choice answered by value is listed in; choices keep their keys."""
    values = list(decision.choices.values())
    keys = list(decision.choices)
    if decision.permutations in (1, "auto"):
        return [decision]
    count = MAX_VALUE_ORDERS if decision.permutations == "all" else min(decision.permutations, MAX_VALUE_ORDERS)
    canonical = sorted(range(len(values)), key=lambda i: json.dumps(values[i]))
    orders = [canonical]
    while len(orders) < count:
        order = list(canonical)
        rng.shuffle(order)
        orders.append(order)
    return [replace(decision, choices={keys[i]: values[i] for i in order}, permutations=1) for order in orders]


def _choice_orderings(decision, rng):
    """Rebind values to fixed control labels, sampling ranks without enumeration."""
    labels = list(decision.choices)
    values = list(decision.choices.values())
    # Orders start from a canonical one, so the orders used do not depend on the order the enum
    # was written in.
    canonical = sorted(range(len(values)), key=lambda i: json.dumps(values[i]))
    if decision.permutations == "auto" and len(values) > MAX_AUTO_ORDERS:
        # Past MAX_AUTO_ORDERS choices, as many rotations, evenly spaced: each choice takes positions
        # spread from the first to the last.
        shifts = sorted({round(k * len(values) / MAX_AUTO_ORDERS) for k in range(MAX_AUTO_ORDERS)})
        orders = [tuple(canonical[(i + shift) % len(values)] for i in range(len(values))) for shift in shifts]
        return [(replace(decision, choices=dict(zip(labels, (values[i] for i in order))),
                         permutations=1), order) for order in orders]
    if decision.permutations == "auto" and len(values) > 1:
        # Balance positions and neighbours.
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
            orders.append(tuple(canonical[i] for i in order))
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
    image_count=0, prefix_cached=False,
):
    rows_by_name = {}
    prompts_by_name = {}
    ancestors = {}
    hidden = {d.name for d in decisions if d.effort_for is not None}
    by_name = {d.name: d for d in decisions}

    def confidence(name: str) -> float | None:
        """A dependency's confidence, for a "when" that tests it; None without probabilities."""
        row = rows_by_name[name]
        return _confidence(by_name[name], row["probabilities"]) if row.get("probabilities") else None

    for layer in dependency_layers(decisions):
        dependency_values = {}
        parent_prefixes = {}
        settled = []
        arrays = []
        for decision in layer:
            # Skipped: a dependency was skipped, so its answer does not exist, or the
            # field's "when" does not match. Either way nothing is sent for it. An object
            # is skipped only when all of its properties are.
            groups = (decision.dependency_groups if decision.dependency_groups is not None
                      else tuple((name,) for name in decision.depends_on or ()))
            if any(all(rows_by_name[name] is None for name in group) for group in groups) or (
                    decision.when is not None and not condition_met(
                        decision, {name: rows_by_name[name]["value"] for name, _ in decision.when},
                        {name: confidence(name) for name, tests in decision.when
                         if any(operator == "confidence" for operator, _ in tests)})):
                rows_by_name[decision.name] = prompts_by_name[decision.name] = None
                continue
            visible = set(decision.depends_on or ())
            for name in decision.depends_on or ():
                visible.update(ancestors[name])
            ancestors[decision.name] = visible
            # An effort question is invisible to the field it serves: the field's
            # prompt is the one it would have without thinking: "auto".
            visible -= hidden
            # An array leaves no prompt to continue from: its turns each had their own.
            parents = [name for name in decision.depends_on or ()
                       if name not in hidden and prompts_by_name.get(name) is not None]
            if parents and not isinstance(decision, ArrayChoice):
                # Longest serialized prefix is a deterministic heuristic; KV
                # from distinct branches cannot be concatenated.
                parent = max(parents, key=lambda name: len(prompts_by_name[name]))
                parent_prefixes[decision.name] = prompts_by_name[parent]
            # An object's properties are shown as the object.
            values = {}
            for d in decisions:
                if d.name in visible and rows_by_name[d.name] is not None:
                    _put(values, _path(d), rows_by_name[d.name]["value"])
            dependency_values[decision.name] = values
            if isinstance(decision, ArrayChoice):
                arrays.append(decision)
                continue
            if decision.effort_from is not None:
                budget = THINKING_EFFORTS[rows_by_name[decision.effort_from]["value"]]
                decision = replace(decision, thinking=budget is not None, thinking_budget=budget)
            settled.append(decision)
        if settled:
            rows, prompts = _execute_batch_decisions(
                client, context, settled, mode, temperature, rng, numeric_max_digits,
                dependency_values=dependency_values,
                parent_prefixes=parent_prefixes,
                image_count=image_count,
                prefix_cached=prefix_cached,
            )
            for decision, row, prompt in zip(settled, rows, prompts):
                rows_by_name[decision.name] = row
                prompts_by_name[decision.name] = prompt
        # The layer's arrays, after its other fields, each in turns of its own.
        for array in arrays:
            rows_by_name[array.name] = _generate_array(
                client, context, array, dependency_values[array.name], mode, temperature, rng,
                numeric_max_digits, image_count)
            prompts_by_name[array.name] = None
    return ([rows_by_name[d.name] for d in decisions],
            [prompts_by_name[d.name] for d in decisions])


def _value_key(value: Any) -> str:
    """A choice's value as a key: 1 and 1.0 alike, true and 1 apart."""
    return json.dumps(float(value) if type(value) in (int, float) else value)


def _describe_item(decision: Choice) -> str:
    """An item's or a property's type in an array's specification."""
    or_null = " or null" if decision.nullable else ""
    if decision.text_type:
        return "string" + or_null
    if decision.numeric_type is not None:
        return decision.numeric_type + or_null
    if decision.syntax == "Bool":
        return "boolean" + or_null
    if not decision.descriptions:
        return "one of " + json.dumps(list(decision.choices.values()), ensure_ascii=False)
    said = {_value_key(value): text for value, text in decision.descriptions}
    return "one of " + ", ".join(json.dumps(value, ensure_ascii=False)
                                 + (f" ({said[_value_key(value)]})" if _value_key(value) in said else "")
                                 for value in decision.choices.values())


def _whole_items(array: ArrayChoice) -> bool:
    """Whether an array writes each item as one JSON object under its grammar: every object array.

    Its properties then follow one another in a single request, each seeing the ones before it, so
    they all describe the same item.
    """
    return array.item_object


def _item_schema(decision: Choice) -> dict[str, Any]:
    """A property's grammar. A choice, enum or boolean, is written as its value: a model writing an item
    writes what it read, and a label could be taken for a value of the same name."""
    if decision.text_type:
        schema: dict[str, Any] = {"type": "string"}
    elif decision.numeric_type is not None:
        schema = {"type": decision.numeric_type}
    elif decision.syntax == "Bool":
        schema = {"type": "boolean"}
    else:
        return {"enum": list(decision.choices.values())}  # null is a choice when the enum lists it
    return {"anyOf": [schema, {"type": "null"}]} if decision.nullable else schema


def _item_line(decision: Choice) -> str:
    """A property's line in a whole item's prompt: its path, its type or values, and its instructions."""
    name = json.dumps(".".join(_path(decision)), ensure_ascii=False)
    # A property's own instructions; the generated default says nothing here, as in the specification.
    said = "" if decision.question.startswith("Choose the value for ") else f": {decision.question}"
    return f"- {name} ({_describe_item(decision)}){said}"


def _object_grammar(decisions: Sequence[Choice]) -> dict[str, Any]:
    """The JSON Schema of an object item: its properties by path, nested objects included, in order."""
    tree: dict[str, Any] = {}
    for decision in decisions:
        *parents, leaf = _path(decision)
        node = tree
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = decision

    def schema(node: Mapping[str, Any]) -> dict[str, Any]:
        return {"type": "object", "additionalProperties": False, "required": list(node),
                "properties": {name: _item_schema(child) if isinstance(child, Choice) else schema(child)
                               for name, child in node.items()}}
    return schema(tree)


def _generate_whole_item(client, state: str, array: ArrayChoice, temperature: float, rng: random.Random,
                         numeric_max_digits: int, image_count: int) -> tuple[dict[str, Any], str]:
    """An object array's next item in one request: the whole object, its keys in schema order."""
    lines = [OBJECT_ITEM_SCOPE.split(" Answer one of its properties.")[0],
             "Answer with the whole object as JSON, with these properties in this order:"]
    lines += [_item_line(decision) for decision in array.items]
    schema = _object_grammar(array.items)
    names = schema["required"]
    messages = [{"role": "user", "content": _user_content(state.rstrip(), image_count)},
                {"role": "user", "content": "\n".join(lines)}]
    prompt = client.render_chat(messages, add_generation_prompt=True)
    # Room for every value at its own limit, and for the keys and punctuation around them.
    budget = sum(getattr(client, "text_max_tokens", 128) if decision.text_type
                 else numeric_max_digits + 4 if decision.numeric_type is not None else 16
                 for decision in array.items) + 8 * len(array.items) + 8
    seed = rng.randrange(2**31) if temperature else 0
    value = client.generate_json(prompt, schema, budget, temperature=temperature or 0, seed=seed)
    if not isinstance(value, Mapping) or list(value) != names:
        raise SGLangError(f"array {array.name!r}: an item did not match its schema")
    return dict(value), prompt


def _array_specification(array: ArrayChoice, dependency_values: Mapping[str, Any]) -> str:
    """What an array is to hold: the part of its state that never changes between turns."""
    lines = [f"Array: {json.dumps(array.name, ensure_ascii=False)}", f"Instructions: {array.question}"]
    items = [item for item in array.items if item.effort_for is None]
    def said(item):  # a property's own instructions; the generated default says nothing here
        return "" if item.question.startswith("Choose the value for ") else f": {item.question}"

    if array.item_object:
        lines.append("Each item is an object with these properties:")
        lines += [f"- {json.dumps('.'.join(_path(item)), ensure_ascii=False)} ({_describe_item(item)}){said(item)}"
                  for item in items]
    else:
        lines.append(f"Each item: {_describe_item(items[0])}. {items[0].question}")
    if dependency_values:
        lines.append("Dependency results (JSON):\n" + json.dumps(dependency_values, ensure_ascii=False))
    return "\n".join(lines)


def _array_state(context: str, specification: str, items: Sequence[Any]) -> str:
    """The context, the array's specification and its committed items. The items come last, so each
    turn's state starts with the one before it, up to the closing bracket."""
    return f"{context.rstrip()}\n\n{specification}\nCurrent array (JSON): {_canonical_json(list(items))}"


# How many times sampling may give an item already in the array before the array stops.
DUPLICATE_RETRIES = 2


def _next_item_kind(client, state, array, mode, temperature, rng, numeric_max_digits, image_count):
    """What the next item of an object array is: "item", or None to end the array.

    Today an array holds one kind of item, so this is the hidden question whether to append a new
    item: false ends the array. An array of several kinds would ask which kind comes next, or none,
    in the same single scoring. The question branches off the state and is never committed to it.
    """
    rows, prompts = _execute_batch_decisions(client, state, [array.continue_choice], mode, temperature, rng,
                                             numeric_max_digits, image_count=image_count)
    return ("item" if rows[0]["value"] else None), prompts


def _generate_array(client, context, array, dependency_values, mode, temperature, rng,
                    numeric_max_digits, image_count=0):
    """An array's items, one turn at a time, each turn on the full state so far.

    Below minItems a turn simply generates an item. Past it, a scalar item is asked for as nullable,
    and null ends the array: one request a turn. An object array first asks what the next item is
    (_next_item_kind); an item follows only if there is one, its properties forking from the state
    side by side. Each item is committed to the state. An item already in the array is not added
    again: with argmax the next turn would give it again, so the array ends; sampling may try again
    a few times. The array also ends at maxItems, or at MAX_ARRAY_ITEMS without one.

    An array that continues from items (continue_from) starts with them committed: every turn sees
    them, an item repeating one ends the array, and they come back first. minItems and maxItems count
    them; MAX_ARRAY_ITEMS counts the items this call adds.
    """
    specification = _array_specification(array, dependency_values)
    items: list[Any] = list(array.continue_from)
    committed: set[str] = {_canonical_json(item) for item in items}
    prompts: list[str] = []
    thinking: dict[str, str] = {}
    efforts: dict[str, str] = {}
    skipped: list[str] = []
    retries = 0 if mode == "argmax" else DUPLICATE_RETRIES
    whole = _whole_items(array) and callable(getattr(client, "generate_json", None))
    limit = len(items) + MAX_ARRAY_ITEMS if array.max_items is None else array.max_items
    while len(items) < limit:
        state = _array_state(context, specification, items)
        may_end = len(items) >= array.min_items
        asked = may_end and array.open_items is None
        if asked:
            kind, turn = _next_item_kind(client, state, array, mode, temperature, rng, numeric_max_digits,
                                         image_count)
            prompts += turn
            if kind is None:
                break
        item_prefix = f"{array.name}[{len(items)}]"
        if whole:
            item, prompt = _generate_whole_item(client, state, array, temperature, rng, numeric_max_digits,
                                                image_count)
            prompts.append(prompt)
            item_thinking, item_efforts, item_skipped = {}, {}, []
        else:
            decisions = list(array.open_items if may_end and array.open_items is not None else array.items)
            run_item = (_execute_dependency_decisions if any(d.depends_on is not None for d in decisions)
                        else _execute_batch_decisions)
            # The continue question's prompt starts with the state, so once it is asked the state is cached.
            rows, turn = run_item(client, state, decisions, mode, temperature, rng, numeric_max_digits,
                                  image_count=image_count, prefix_cached=asked)
            prompts += [prompt for prompt in turn if prompt is not None]
            output, item_thinking, item_efforts, item_skipped = _assemble(decisions, rows, item_prefix)
            item = output if array.item_object else output.get("item")
        if not array.item_object:
            if may_end and array.open_items is not None and item is None:
                break  # null: the array is complete
            # A scalar item is named by its index alone, "skills[0]".
            item_thinking = {item_prefix: text for text in item_thinking.values()}
            item_efforts = {item_prefix: effort for effort in item_efforts.values()}
        key = _canonical_json(item)
        if key in committed:
            LOG.info("array=%s duplicate item %s", array.name, key)
            if retries == 0:
                break
            retries -= 1
            continue
        committed.add(key)
        items.append(item)
        thinking.update(item_thinking)
        efforts.update(item_efforts)
        skipped += item_skipped
    return {"name": array.name, "question": array.question, "label": None, "value": items,
            "probabilities": None, "thinking": thinking, "thinking_effort": efforts, "skipped": skipped,
            "prompts": prompts}


def _prompts_sent(decision: Choice) -> int:
    """How many prompts a decision sends from its prefix: two or more for a choice averaged over orderings."""
    if decision.text_type or decision.numeric_type is not None or decision.permutations == 1:
        return 1
    return 2


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
    prefix_cached: bool = False,
) -> tuple[list[dict], list[str]]:
    """prefix_cached: the context's prompt was just sent, so SGLang has it cached and it needs no warm-up."""
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

    # Warm each parent that two or more of the layer's prompts continue, before them, including
    # thinking/text requests: prompts sent in one batch cannot reuse each other's cache. A prefix
    # only one prompt continues is cached by that prompt. Root context is warmed only in the first DAG layer.
    if incremental:
        users = Counter()
        for decision in decisions:
            users[parent_prefixes.get(decision.name, shared_prefix)] += _prompts_sent(decision)
        prefixes = list(dict.fromkeys(parent_prefixes.values()))
        if any(d.name not in parent_prefixes for d in decisions):
            prefixes.insert(0, shared_prefix)
        for prefix in prefixes:
            if users[prefix] >= 2 and not (prefix_cached and prefix == shared_prefix):
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
    value_jobs = []  # (decision index, [(prompt slot, variant), ...]) for choices answered by value
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
        if decision.by_value:
            if getattr(client, "choice_scorer", None) is None:
                raise SchemaError(f"{decision.name!r} has {len(decision.choices)} choices; "
                                  f"the maximum is {MAX_ENUM_CHOICES}")
            jobs = []
            for variant in _value_orderings(decision, rng):
                variant_content = question_content(variant)
                jobs.append((decision_slots[index] if list(variant.choices) == list(decision.choices) else
                             generation_prompt(parent, variant_content,
                                               shared_messages + [{"role": "user", "content": variant_content}],
                                               variant), variant))
            value_jobs.append((index, jobs))
            finite_indexes.append(index)
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
    # A lone prompt, such as an array's continue question, caches the prefix itself.
    if finite_indexes and not incremental and not prefix_cached and sum(map(_prompts_sent, decisions)) >= 2:
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

    # A layer's numbers and strings decode side by side in one request, and its choices are
    # scored in that same request when the client can: one round trip for the layer, not two.
    merged = bool((numeric_pending or text_pending) and scoring_prompts
                  and callable(getattr(client, "generate_and_score", None)))
    if numeric_pending or text_pending:
        number_items = [(prompt + decision.answer_prefill, decision)
                        for _, decision, _, prompt in numeric_pending]
        number_seed = rng.randrange(2**31) if numeric_pending else 0
        text_seed = rng.randrange(2**31) if text_pending else 0
        fields = (
            [prompt for prompt, _ in number_items],
            [numeric_pattern(d.numeric_type or "", numeric_max_digits, d.nullable) for _, d in number_items],
            # Digits, sign, point and the closing brace; tokens hold one or more characters.
            numeric_max_digits + 4,
            # From {"name": the model writes the string's first token, quote included.
            [prompt + decision.answer_prefill for _, decision, _, prompt in text_pending],
        )
        options = dict(
            temperature=0 if mode == "argmax" else temperature,
            number_seed=number_seed,
            text_seed=text_seed,
            after_key=all(decision.answer_prefill for _, decision, _, _ in text_pending),
            # A nullable string writes null or its text in the same request.
            nullable=[decision.nullable for _, decision, _, _ in text_pending],
        )
        if merged:
            raw_numbers, values, scored, elapsed = client.generate_and_score(
                *fields, scoring_prompts, scoring_ids, **options)
        else:
            raw_numbers, values = client.generate_fields(*fields, **options)
        for (index, decision, messages, prompt), (value, _completed, generated_text) in zip(
                numeric_pending, _parse_numbers(number_items, raw_numbers)):
            open_results[index] = (open_row(decision, value),
                                   complete(prompt, messages, _closed_answer(decision, generated_text)))
        for (index, decision, messages, prompt), value in zip(text_pending, values):
            completed = complete(prompt, messages, _closed_answer(decision, json.dumps(value, ensure_ascii=False)))
            open_results[index] = (open_row(decision, value), completed)

    grouped_scores = []
    if scoring_prompts:
        if not merged:
            scored, elapsed = client.score_candidates_batch(scoring_prompts, scoring_ids)
        offset = 0
        for orders in ordering_groups:
            grouped_scores.append(scored[offset:offset + len(orders)])
            offset += len(orders)
    else:
        elapsed = 0.0
    probability_temperature = temperature if mode == "sample" else 1.0

    # Choices answered by value: the scorer gives each value's probability after each order's
    # prompt; a value's probability is its mean over the orders.
    value_probabilities: dict[int, dict[str, float]] = {}
    if value_jobs:
        requests = [(index, variant, ready[slot] + variant.label_prefill)
                    for index, jobs in value_jobs for slot, variant in jobs]
        scored_values = client.choice_scorer.score(
            client, [prompt for _, _, prompt in requests],
            [[_value_continuation(value) for value in variant.choices.values()] for _, variant, _ in requests])
        sums: dict[int, dict[str, list[float]]] = {}
        for (index, variant, _), probabilities in zip(requests, scored_values):
            per_key = sums.setdefault(index, {key: [] for key in decisions[index].choices})
            for key, probability in zip(variant.choices, probabilities):
                per_key[key].append(float(probability))
        for index, per_key in sums.items():
            means = {key: math.fsum(values) / len(values) for key, values in per_key.items()}
            if probability_temperature != 1.0:
                weights = {key: p ** (1 / probability_temperature) for key, p in means.items()}
                total = math.fsum(weights.values())
                means = {key: weight / total for key, weight in weights.items()}
            value_probabilities[index] = means

    results: list[dict] = []
    completed_prompts: list[str] = []
    labelled = iter(zip(label_tokens_by_decision, grouped_scores, ordering_groups))
    for finite_index, index in enumerate(finite_indexes):
        decision = decisions[index]
        if decision.by_value:
            label_tokens, raw, meta = None, None, {}
            probabilities = value_probabilities[index]
        else:
            label_tokens, group, orders = next(labelled)
            by_id, meta = group[0]
            raw = {
                label: by_id[token_id]
                for label, (token_id, _) in label_tokens.items()
            }
            probabilities = _mean_order_probabilities(group, orders, label_tokens, probability_temperature)
        selected = (
            max(probabilities, key=probabilities.__getitem__)
            if mode == "argmax"
            else _sample(probabilities, rng)
        )
        semantic_value = decision.choices[selected]
        completed_prompts.append(
            complete(
                prompts[finite_index],
                shared_messages + [{"role": "user", "content": question_content(decision)}],
                _closed_value(decision, semantic_value) if decision.by_value
                else _closed_label(decision, label_tokens[selected][1]),
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
        if decision.levels:
            semantic_value = _score_value(decision, probabilities)
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
