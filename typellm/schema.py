"""Compilation for TypeLLM's schema subset and bounded open values."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


MAX_ENUM_CHOICES = 24
MAX_PERMUTATIONS = 720


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    """Tokenizer-independent decision compiled from ordinary JSON Schema."""

    name: str
    question: str
    choices: tuple[Any, ...]
    syntax: str = "Choice"
    numeric_type: str | None = None
    text_type: bool = False
    permutations: int | str = 1
    return_probabilities: bool = False
    depends_on: tuple[str, ...] | None = None
    nullable: bool = False
    # None follows the client's thinking and thinking_budget.
    thinking: bool | None = None
    thinking_budget: int | None = None


def _has_duplicates(values: Sequence[Any]) -> bool:
    for index, value in enumerate(values):
        for previous in values[:index]:
            # JSON booleans are distinct from numbers, while 1 and 1.0 denote
            # the same JSON numeric value.
            both_numbers = (
                type(value) in {int, float} and type(previous) in {int, float}
            )
            if (both_numbers and value == previous) or (
                type(value) is type(previous) and value == previous
            ):
                return True
    return False


def _is_finite_number(value: Any) -> bool:
    # Exact types exclude bool; Python ints are finite at any magnitude.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def compile_json_schema(schema: Mapping[str, Any]) -> list[Decision]:
    """Compile an ordered JSON Schema object into TypeLLM decisions."""
    if not isinstance(schema, Mapping):
        raise SchemaError("schema must be a mapping")
    if schema.get("type") != "object":
        raise SchemaError("root JSON Schema type must be 'object'")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise SchemaError("JSON Schema properties must be a non-empty object")

    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
        raise SchemaError("JSON Schema required must be a list of field names")
    if len(set(required)) != len(required):
        raise SchemaError("JSON Schema required contains duplicate field names")
    unknown_required = [name for name in required if name not in properties]
    if unknown_required:
        raise SchemaError(
            f"required fields are missing from properties: {unknown_required!r}"
        )

    decisions: list[Decision] = []
    for name, field in properties.items():
        if not isinstance(name, str) or not name:
            raise SchemaError("property names must be non-empty strings")
        if not isinstance(field, Mapping):
            raise SchemaError(f"property {name!r} must be a schema object")

        instructions = field.get("instructions")
        if "instructions" in field and not isinstance(instructions, str):
            raise SchemaError(f"instructions for {name!r} must be a string")
        for old_key in ("question", "x-question"):
            if old_key in field:
                raise SchemaError(f"{old_key} for {name!r} is no longer supported; use instructions")
        # Decoding cannot hold a model to a range, so numeric bounds are not offered.
        for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if keyword in field:
                raise SchemaError(f"{keyword} is not supported (on {name!r}); use an enum for a fixed set of values")
        description = field.get("description")
        if "description" in field and not isinstance(description, str):
            raise SchemaError(f"description for {name!r} must be a string")
        question = (
            instructions
            if instructions is not None
            else description
            if description is not None
            else f'Choose the value for "{name}".'
        )

        field_type = field.get("type")
        # ["string", "null"] and the like: one value type that may also be null.
        nullable = False
        if isinstance(field_type, list):
            kinds = [kind for kind in field_type if kind != "null"]
            if len(field_type) != 2 or len(kinds) != 1 or not isinstance(kinds[0], str):
                raise SchemaError(f'type for {name!r} must be one type or [type, "null"]')
            field_type, nullable = kinds[0], True
        enum = field.get("enum")
        permutations = field.get("permutations", 1)
        if "permutations" in field:
            if enum is None:
                raise SchemaError(f"permutations for {name!r} requires an explicit enum")
            if not (permutations in ("auto", "all") or type(permutations) is int and permutations > 0):
                raise SchemaError(f"permutations for {name!r} must be 'auto', 'all' or a positive integer")
            if isinstance(enum, list) and permutations != "auto":
                count = math.factorial(len(enum))
                budget = count if permutations == "all" else min(permutations, count)
                if budget > MAX_PERMUTATIONS:
                    raise SchemaError(f"permutations for {name!r} exceeds {MAX_PERMUTATIONS}; use a smaller integer budget")
        return_probabilities = field.get("return_probabilities", False)
        if type(return_probabilities) is not bool:
            raise SchemaError(f"return_probabilities for {name!r} must be a boolean")
        if "return_probabilities" in field and field_type != "boolean" and enum is None:
            raise SchemaError(f"return_probabilities for {name!r} is only supported for enum or boolean fields")
        if "x-score" in field:
            raise SchemaError(
                f"x-score for {name!r} is not supported; use a number enum"
            )
        if "x-other" in field:
            raise SchemaError(
                f"x-other for {name!r} is not supported; use a closed enum"
            )
        # Checking a length means decoding every token back to text, which slows every
        # string; answers stop at the client's text_max_tokens instead.
        if "maxLength" in field:
            raise SchemaError(f"maxLength is not supported (on {name!r}); string answers stop at "
                              "text_max_tokens, so ask for the length you want in instructions")
        if field_type == "string" and enum is None:
            for keyword in ("minLength", "pattern", "format"):
                if keyword in field:
                    raise SchemaError(f"{keyword} is not supported for text fields")
            decisions.append(Decision(name, question, (), "Text", text_type=True, nullable=nullable))
            continue
        if field_type == "boolean":
            values = ([True, False] + [None] * nullable) if enum is None else enum
            if not isinstance(values, list) or not values:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            if any(type(value) is not bool and not (nullable and value is None) for value in values):
                raise SchemaError(f"boolean enum for {name!r} may contain only booleans")
            syntax = "Bool"
        elif field_type in {"integer", "number"} and enum is None:
            decisions.append(
                Decision(
                    name=name,
                    question=question,
                    choices=(),
                    syntax="Integer" if field_type == "integer" else "Number",
                    numeric_type=field_type,
                    nullable=nullable,
                )
            )
            continue
        elif field_type in {"string", "integer", "number"}:
            if enum is None:
                raise NotImplementedError(
                    f"property {name!r} has type {field_type!r} without a finite enum"
                )
            if not isinstance(enum, list) or not enum:
                raise SchemaError(f"enum for {name!r} must be a non-empty list")
            if len(enum) > MAX_ENUM_CHOICES:
                raise SchemaError(
                    f"enum for {name!r} has {len(enum)} values; "
                    f"the maximum is {MAX_ENUM_CHOICES}"
                )
            values = enum
            # As in JSON Schema, null is allowed only when the enum lists it.
            typed = [value for value in values if not (nullable and value is None)]
            if field_type == "string":
                valid = all(isinstance(value, str) for value in typed)
            elif field_type == "integer":
                valid = all(type(value) is int for value in typed)
            else:
                valid = all(_is_finite_number(value) for value in typed)
            if not valid:
                raise SchemaError(
                    f"enum values for {name!r} do not match type {field_type!r}"
                )
            syntax = "Choice"
        else:
            raise NotImplementedError(
                f"property {name!r} has unsupported JSON Schema type {field_type!r}"
            )

        if len(values) > MAX_ENUM_CHOICES:
            raise SchemaError(
                f"enum for {name!r} has {len(values)} values; "
                f"the maximum is {MAX_ENUM_CHOICES}"
            )
        if _has_duplicates(values):
            raise SchemaError(f"enum for {name!r} contains duplicate values")
        decisions.append(
            Decision(name, question, tuple(values), syntax, return_probabilities=return_probabilities,
                     permutations=permutations, nullable=nullable)
        )

    compiled = []
    for decision in decisions:
        field = properties[decision.name]
        dependencies = field.get("depends_on")
        if "depends_on" in field:
            if not isinstance(dependencies, list) or any(
                not isinstance(name, str) or not name for name in dependencies
            ):
                raise SchemaError(f"depends_on for {decision.name!r} must be a list of field names")
            if len(set(dependencies)) != len(dependencies):
                raise SchemaError(f"depends_on for {decision.name!r} contains duplicates")
            dependencies = tuple(dependencies)
        thinking = field.get("thinking")
        if thinking is not None and type(thinking) is not bool:
            raise SchemaError(f"thinking for {decision.name!r} must be a boolean")
        budget = field.get("thinking_budget")
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise SchemaError(f"thinking_budget for {decision.name!r} must be a positive integer")
        compiled.append(replace(decision, depends_on=dependencies, thinking=thinking, thinking_budget=budget))
    dependency_layers(compiled)
    return compiled


def dependency_layers(decisions: Sequence) -> list[list]:
    """Stable topological layers, validated before any model requests."""
    names = {decision.name for decision in decisions}
    for decision in decisions:
        for dependency in decision.depends_on or ():
            if dependency not in names:
                raise SchemaError(f"unknown dependency {dependency!r} for {decision.name!r}")
            if dependency == decision.name:
                raise SchemaError(f"field {decision.name!r} cannot depend on itself")
    remaining = list(decisions)
    completed = set()
    layers = []
    while remaining:
        layer = [d for d in remaining if set(d.depends_on or ()) <= completed]
        if not layer:
            raise SchemaError(f"dependency cycle among fields: {[d.name for d in remaining]!r}")
        layers.append(layer)
        completed.update(d.name for d in layer)
        remaining = [d for d in remaining if d.name not in completed]
    return layers
