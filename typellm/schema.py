"""Compilation for TypeLLM's schema subset and bounded open values."""

from __future__ import annotations

import json
import math
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


MAX_ENUM_CHOICES = 26
# A field's enum may be longer when the runtime scores choices by value (compile_json_schema's
# max_choices); array items and score levels keep MAX_ENUM_CHOICES.
_MAX_CHOICES: ContextVar[int] = ContextVar("max_choices", default=MAX_ENUM_CHOICES)
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
    # thinking: "auto" adds a hidden question that picks the effort per call:
    # effort_from on the field names that question, effort_for on it names the field.
    effort_from: str | None = None
    effort_for: str | None = None
    # "when": run only if every named dependency's answer passes its tests;
    # kept as ((field, ((operator, operand), ...)), ...). A skipped field skips its dependents too.
    when: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] | None = None
    # Where the answer goes in the result: ("total",), or ("person", "age") for an object's property,
    # whose name is then "person.age". Empty means (name,).
    path: tuple[str, ...] = ()
    # The name the prompt shows, "age" for "person.age"; None shows name.
    shown_name: str | None = None
    # Lines the prompt puts before the field's own, such as the object it belongs to.
    scope: str = ""
    # depends_on, grouped by what was named: an object's properties are one group, skipped only
    # when all of them are. None: each dependency is its own group.
    dependency_groups: tuple[tuple[str, ...], ...] | None = None
    # A choice's description by value, from "choices": ((value, description), ...); empty for an enum.
    descriptions: tuple[tuple[Any, str], ...] = ()
    # A score's ordered levels, lowest first: its choices are their labels, and its value is the
    # probability-weighted average of their indices. Empty for every other field.
    levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArrayField:
    """A variable number of items, each one value of `items`: a scalar decision, or an object's decisions."""

    name: str
    question: str
    items: tuple[Decision, ...]
    item_object: bool
    min_items: int = 0
    max_items: int | None = None
    depends_on: tuple[str, ...] | None = None
    when: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] | None = None
    dependency_groups: tuple[tuple[str, ...], ...] | None = None
    # A scalar array's items past minItems: the same decisions, the item nullable, null ending the
    # array. None for an object array, or an item already nullable: those ask the continue question.
    open_items: tuple[Decision, ...] | None = None
    # As on Decision, for code that handles both: an array is never an effort question.
    effort_for: None = None
    effort_from: None = None
    # continue_from: items a caller already has, from an earlier part of a long input. The array
    # starts from them and returns them first; new items follow.
    continue_from: tuple[Any, ...] = ()

    @property
    def path(self) -> tuple[str, ...]:
        return (self.name,)


# Without maxItems, an array stops after this many items whatever the model says.
MAX_ARRAY_ITEMS = 50

# The hidden question an object array asks before each item past its minItems; false ends the array.
# Of the wordings tried on Qwen3.8 (2026-10-05), this plain one was right on arrays found in the
# context and arrays the model produces alike: "would another item improve the array" stopped empty
# arrays early, and "does the context mention another" stopped produced ones.
CONTINUE_QUESTION = "Should a new item be appended to the current array?"
# The name the question is asked under, so its answer starts as a field's does.
CONTINUE_FIELD = "append_item"
# What opens the prompt of an item, or of each property of an object item.
ITEM_SCOPE = ("Give the next item of the array above: one new item, distinct from the items "
              "already in the current array.\n")
# A scalar item past minItems may end the array instead, with null.
OPEN_ITEM_SCOPE = ("Give the next item of the array above: one new item, distinct from the items already in "
                   "the current array. Answer null instead if the current array already satisfies the "
                   "specification, with no distinct item left that would materially improve it.\n")
OBJECT_ITEM_SCOPE = ("Give the next item of the array above: one new object, distinct from the objects "
                     "already in the current array. Answer one of its properties.\n")


# thinking_effort levels and their budgets; "none" does not think.
THINKING_EFFORTS = {"none": None, "low": 512, "medium": 2048, "high": 4096}

EFFORT_QUESTION = (
    "Before the question below is answered, decide how much step-by-step reasoning it needs.\n"
    "none: the answer is stated directly or is obvious.\n"
    "low: a short check or one simple inference.\n"
    "medium: several steps of reasoning or calculation.\n"
    "high: hard, many-step or high-stakes reasoning.\n\n"
    "The question:\n{question}"
)


def _describe(decision: Decision) -> str:
    """A field's type, instructions and choices, as its own prompt states them."""
    or_null = " or null" if decision.nullable else ""
    if decision.text_type:
        kind = "string" + or_null
    elif decision.numeric_type is not None:
        kind = decision.numeric_type + or_null
    else:
        kind = "boolean" if decision.syntax == "Bool" else "choice"
    lines = [f"Type: {kind}", f"Instructions: {decision.question}"]
    if decision.choices and decision.syntax != "Bool":
        lines.append(f"Choices: {json.dumps(list(decision.choices), ensure_ascii=False)}")
    return "\n".join(lines)


def _same_value(a: Any, b: Any) -> bool:
    """JSON equality: 1 and 1.0 match, true and 1 do not."""
    if type(a) in {int, float} and type(b) in {int, float}:
        return a == b
    return type(a) is type(b) and a == b


def _has_duplicates(values: Sequence[Any]) -> bool:
    for index, value in enumerate(values):
        if any(_same_value(value, previous) for previous in values[:index]):
            return True
    return False


def _is_finite_number(value: Any) -> bool:
    # Exact types exclude bool; Python ints are finite at any magnitude.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def compile_json_schema(schema: Mapping[str, Any], max_choices: int = MAX_ENUM_CHOICES) -> list[Decision | ArrayField]:
    """Compile an ordered JSON Schema object into TypeLLM decisions.

    max_choices bounds a field's enum; array items and score levels take MAX_ENUM_CHOICES.

    A scalar field is one decision. An object is one decision per scalar property, named by its
    path ("person.age"), in declaration order. An array is an ArrayField whose items are compiled
    the same way, apart from the fields around it.
    """
    token = _MAX_CHOICES.set(max(max_choices, MAX_ENUM_CHOICES))
    try:
        return _compile(schema)
    finally:
        _MAX_CHOICES.reset(token)


def _compile(schema: Mapping[str, Any]) -> list[Decision | ArrayField]:
    if not isinstance(schema, Mapping):
        raise SchemaError("schema must be a mapping")
    if schema.get("type") != "object":
        raise SchemaError("root JSON Schema type must be 'object'")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise SchemaError("JSON Schema properties must be a non-empty object")
    _check_required(schema, properties, "the schema")

    entries: list[_Entry] = []
    scope: dict[str, tuple[str, Any]] = {}
    order: list[str | ArrayField] = []
    for name, field in properties.items():
        if not isinstance(name, str) or not name:
            raise SchemaError("property names must be non-empty strings")
        if not isinstance(field, Mapping):
            raise SchemaError(f"property {name!r} must be a schema object")
        kind = _kind(name, field)
        if kind == "object":
            members = _object_entries((name,), field, name, inside_array=False, around=scope)
            entries.extend(members)
            scope[name] = ("object", tuple(entry.decision.name for entry in members))
            order.append(name)
        elif kind == "array":
            scope[name] = ("array", name)
            order.append(_array(name, field))
        else:
            entries.append(_Entry(_scalar(name, name, field), field, None))
            scope[name] = ("scalar", name)
            order.append(name)
    # Fields named in depends_on and when are found in the scope they were written in: the top
    # level for top-level fields, objects and arrays; the object for a property.
    for entry in entries:
        if entry.scope is None:
            entry.scope = scope
    linked = _link(entries)
    by_owner: dict[str, list[Decision]] = {}
    for decision in linked:
        by_owner.setdefault(decision.path[0] if decision.path else decision.name, []).append(decision)
    compiled: list[Decision | ArrayField] = []
    for item in order:
        if isinstance(item, ArrayField):
            compiled.append(_link_array(item, properties[item.name], scope, linked))
        else:
            compiled.extend(by_owner.pop(item))
    names = [item.name for item in compiled]
    if len(set(names)) != len(names):
        clash = next(name for name in names if names.count(name) > 1)
        raise SchemaError(f"field name {clash!r} is used twice: an object's property path "
                          "or a hidden question takes it")
    dependency_layers(compiled)
    return compiled


@dataclass
class _Entry:
    """A scalar decision before linking: its field, the names its depends_on and when may use, and
    what it takes from the objects around it."""

    decision: Decision
    field: Mapping[str, Any]
    scope: dict[str, tuple[str, Any]] | None
    # The objects around it: their own depends_on and when, linked in the scope around each.
    outer: tuple[tuple[Mapping[str, Any], dict[str, tuple[str, Any]], str], ...] = ()


def _check_required(schema: Mapping[str, Any], properties: Mapping[str, Any], label: str) -> None:
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
        raise SchemaError(f"required for {label} must be a list of field names")
    if len(set(required)) != len(required):
        raise SchemaError(f"required for {label} contains duplicate field names")
    unknown_required = [name for name in required if name not in properties]
    if unknown_required:
        raise SchemaError(
            f"required fields are missing from properties: {unknown_required!r}"
        )


# Keys of one kind of schema that mean nothing, or something else, on another.
_OBJECT_ONLY = ("properties",)
_ARRAY_ONLY = ("items", "minItems", "maxItems", "continue_from")
_SCALAR_ONLY = ("enum", "choices", "levels", "permutations", "return_probabilities", "thinking", "thinking_effort", "thinking_budget")


def _kind(label: str, field: Mapping[str, Any]) -> str:
    """"object", "array" or "scalar", with the keys only another kind takes rejected."""
    field_type = field.get("type")
    if isinstance(field_type, list) and ("object" in field_type or "array" in field_type):
        raise SchemaError(f'type for {label!r}: an object or an array cannot also be null')
    kind = field_type if field_type in ("object", "array") else "scalar"
    for key in _OBJECT_ONLY:
        if key in field and kind != "object":
            raise SchemaError(f"{key} for {label!r} is only for type 'object'")
    for key in _ARRAY_ONLY:
        if key in field and kind != "array":
            raise SchemaError(f"{key} for {label!r} is only for type 'array'")
    for key in _SCALAR_ONLY:
        if key in field and kind != "scalar":
            where = "its properties" if kind == "object" else "its items"
            raise SchemaError(f"{key} is not supported on {kind} {label!r}; set it on {where}")
    return kind


def _question(label: str, field: Mapping[str, Any], default: str) -> str:
    instructions = field.get("instructions")
    if "instructions" in field and not isinstance(instructions, str):
        raise SchemaError(f"instructions for {label!r} must be a string")
    for old_key in ("question", "x-question"):
        if old_key in field:
            raise SchemaError(f"{old_key} for {label!r} is no longer supported; use instructions")
    description = field.get("description")
    if "description" in field and not isinstance(description, str):
        raise SchemaError(f"description for {label!r} must be a string")
    return instructions if instructions is not None else description if description is not None else default


# Keys an array item's properties do not take: the item is written whole, in one request.
_NOT_IN_ITEMS = ("thinking", "thinking_effort", "thinking_budget", "depends_on", "when", "permutations",
                 "return_probabilities", "levels")


def _object_entries(path: tuple[str, ...], field: Mapping[str, Any], label: str, *, inside_array: bool,
                    around: dict[str, tuple[str, Any]], outer: tuple = ()) -> list[_Entry]:
    """The scalar properties of an object, nested objects' included, as entries named by path.

    The object's instructions open every property's prompt; its depends_on and when hold for
    every property, linked in `around`, the scope the object is named in (filled by the time of linking).
    """
    properties = field.get("properties")
    if "properties" not in field:
        raise SchemaError(f"object {label!r} needs properties")
    if not isinstance(properties, Mapping):
        raise SchemaError(f"properties for {label!r} must be a mapping of property names to schemas")
    if not properties:
        raise SchemaError(f"properties for {label!r} must not be empty")
    _check_required(field, properties, repr(label))
    question = _question(label, field, "")
    header = "" if not path else f"Object: {json.dumps('.'.join(path), ensure_ascii=False)}\n"
    if question:
        header += f"Object instructions: {question}\n"
    scope: dict[str, tuple[str, Any]] = {}
    entries: list[_Entry] = []
    outer = outer + ((field, around, label),)
    for name, member in properties.items():
        if not isinstance(name, str) or not name:
            raise SchemaError(f"property names of {label!r} must be non-empty strings")
        member_label = f"{label}.{name}"
        if not isinstance(member, Mapping):
            raise SchemaError(f"property {member_label!r} must be a schema object")
        if inside_array:
            for key in _NOT_IN_ITEMS:
                if key in member:
                    raise SchemaError(f"{key} is not supported inside array items (on {member_label!r}): "
                                      "each item is written as one JSON object")
        kind = _kind(member_label, member)
        member_path = path + (name,)
        if kind == "array":
            raise SchemaError(f"{member_label!r}: arrays inside objects are not supported yet")
        if kind == "object":
            members = _object_entries(member_path, member, member_label, inside_array=inside_array,
                                      around=scope, outer=outer)
            for entry in members:
                entry.decision = replace(entry.decision, scope=header + entry.decision.scope)
            entries.extend(members)
            scope[name] = ("object", tuple(entry.decision.name for entry in members))
            continue
        decision = _scalar(".".join(member_path), member_label, member,
                           MAX_ENUM_CHOICES if inside_array else None)
        entries.append(_Entry(replace(decision, path=member_path, shown_name=name, scope=header),
                              member, scope, outer=outer))
        scope[name] = ("scalar", decision.name)
    return entries


def _array(name: str, field: Mapping[str, Any]) -> ArrayField:
    """An array field, its items compiled and linked; its own depends_on and when come later."""
    question = _question(name, field, f'All the items of "{name}".')
    if "items" not in field:
        raise SchemaError(f"array {name!r} needs items")
    items = field["items"]
    if not isinstance(items, Mapping):
        raise SchemaError(f"items for {name!r} must be a schema object")
    min_items = field.get("minItems", 0)
    if type(min_items) is not int or min_items < 0:
        raise SchemaError(f"minItems for {name!r} must be a non-negative integer")
    max_items = field.get("maxItems")
    if max_items is not None and (type(max_items) is not int or max_items < 1):
        raise SchemaError(f"maxItems for {name!r} must be a positive integer")
    if max_items is not None and min_items > max_items:
        raise SchemaError(f"minItems for {name!r} is more than its maxItems")
    if min_items > MAX_ARRAY_ITEMS or (max_items or 0) > MAX_ARRAY_ITEMS:
        raise SchemaError(f"an array holds at most {MAX_ARRAY_ITEMS} items (on {name!r})")
    label = f"{name}.items"
    kind = _kind(label, items)
    if kind == "array":
        raise SchemaError(f"items of {name!r} cannot be arrays: nested arrays are not supported")
    if kind == "object":
        for key in ("depends_on", "when"):
            if key in items:
                raise SchemaError(f"{key} is not supported on the items of {name!r}; "
                                  "set it on the array, or on the items' properties")
        # An item's properties fork from the turn's state side by side; a depends_on between them orders them.
        entries = _object_entries((), items, label, inside_array=True, around={})
        decisions = [replace(decision, scope=OBJECT_ITEM_SCOPE + decision.scope) if decision.effort_for is None
                     else decision for decision in _link(entries)]
    else:
        for key in ("depends_on", "when"):
            if key in items:
                raise SchemaError(f"{key} is not supported on the items of {name!r}; set it on the array")
        item = _scalar("item", label, items, MAX_ENUM_CHOICES)
        if item.return_probabilities or item.levels:
            key = "levels" if item.levels else "return_probabilities"
            raise SchemaError(f"{key} is not supported inside arrays (on {label!r})")
        item_question = _question(label, items, "One item of the array.")
        decisions = [replace(decision, scope=ITEM_SCOPE + decision.scope) if decision.effort_for is None
                     else decision
                     for decision in _link([_Entry(replace(item, path=("item",), question=item_question),
                                                   items, {})])]
        main = next(decision for decision in decisions if decision.effort_for is None)
        open_items = None
        # null as one more choice must fit the enum's labels.
        if not main.nullable and len(main.choices) < MAX_ENUM_CHOICES:
            nullable = replace(main, nullable=True, scope=OPEN_ITEM_SCOPE,
                               choices=main.choices + ((None,) if main.choices else ()))
            open_items = tuple(nullable if decision is main else decision for decision in decisions)
        start = _continue_from(name, field, decisions, item_object=False, max_items=max_items)
        return ArrayField(name, question, tuple(decisions), False, min_items, max_items, open_items=open_items,
                          continue_from=start)
    start = _continue_from(name, field, decisions, item_object=True, max_items=max_items)
    return ArrayField(name, question, tuple(decisions), True, min_items, max_items, continue_from=start)


def _continue_from(name: str, field: Mapping[str, Any], decisions: Sequence[Decision], *, item_object: bool,
                   max_items: int | None) -> tuple[Any, ...]:
    """The items an array continues from, each checked against its items' schema."""
    if "continue_from" not in field:
        return ()
    start = field["continue_from"]
    if not isinstance(start, list):
        raise SchemaError(f"continue_from for {name!r} must be a list of items")
    if max_items is not None and len(start) > max_items:
        raise SchemaError(f"continue_from for {name!r} has {len(start)} items; its maxItems is {max_items}")
    answers = [decision for decision in decisions if decision.effort_for is None]
    for index, item in enumerate(start):
        label = f"{name}.continue_from[{index}]"
        if item_object:
            _check_object_item(label, item, {decision.path: decision for decision in answers}, ())
        else:
            _check_value(label, item, answers[0])
    return tuple(start)


def _check_object_item(label: str, value: Any, leaves: Mapping[tuple[str, ...], Decision],
                       at: tuple[str, ...]) -> None:
    """An object item: known properties only, each a valid value. A property may be missing, as a
    "when" may have skipped it."""
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    for key, item in value.items():
        path = at + (key,)
        if path in leaves:
            _check_value(f"{label}.{key}", item, leaves[path])
        elif any(leaf[:len(path)] == path for leaf in leaves):
            _check_object_item(f"{label}.{key}", item, leaves, path)
        else:
            raise SchemaError(f"{label} has no property {key!r}")


def _check_value(label: str, value: Any, decision: Decision) -> None:
    """A value the decision could have answered."""
    if value is None:
        if not decision.nullable:
            raise SchemaError(f"{label} cannot be null")
        return
    if decision.choices:
        # Compared as JSON does: true is not 1, and 1 is 1.0.
        if not any(choice == value and (type(choice) is type(value)
                                        or (_is_finite_number(choice) and _is_finite_number(value)))
                   for choice in decision.choices):
            raise SchemaError(f"{label} is not one of the choices {list(decision.choices)!r}")
    elif decision.text_type:
        if not isinstance(value, str):
            raise SchemaError(f"{label} must be a string")
    elif decision.numeric_type == "integer":
        if type(value) is not int:
            raise SchemaError(f"{label} must be an integer")
    elif decision.numeric_type == "number":
        if not _is_finite_number(value):
            raise SchemaError(f"{label} must be a number")


def _link_array(array: ArrayField, field: Mapping[str, Any], scope, linked: Sequence[Decision]) -> ArrayField:
    by_id = {decision.name: decision for decision in linked}
    groups, when = _dependencies(array.name, field, scope, by_id)
    if groups is None:
        return array
    return replace(array, depends_on=_flatten(groups), dependency_groups=groups, when=when)


def _flatten(groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for group in groups for name in group))


def _dependencies(label: str, field: Mapping[str, Any], scope: Mapping[str, tuple[str, Any]],
                  by_id: Mapping[str, Decision]) -> tuple[tuple[tuple[str, ...], ...] | None, tuple | None]:
    """A field's depends_on and when, resolved in its scope: (dependency groups, condition).

    A named object is one group of its properties; a "when" names scalar fields only, which
    become dependencies too. Groups are None when the field has neither.
    """
    if "depends_on" not in field and "when" not in field:
        return None, None
    groups: list[tuple[str, ...]] = []
    if "depends_on" in field:
        dependencies = field["depends_on"]
        if not isinstance(dependencies, list) or any(not isinstance(name, str) or not name for name in dependencies):
            raise SchemaError(f"depends_on for {label!r} must be a list of field names")
        if len(set(dependencies)) != len(dependencies):
            raise SchemaError(f"depends_on for {label!r} contains duplicates")
        for dependency in dependencies:
            if dependency not in scope:
                raise SchemaError(f"unknown dependency {dependency!r} for {label!r}")
            kind, target = scope[dependency]
            groups.append(target if kind == "object" else (target,))
    when = None
    if "when" in field:
        tested = field["when"]
        for name in tested if isinstance(tested, Mapping) else ():
            if name in scope and scope[name][0] != "scalar":
                raise SchemaError(f"when for {label!r} names {name!r}, which is an {scope[name][0]}; "
                                  "a condition can only test a scalar field")
        local = {name: by_id[target] for name, (kind, target) in scope.items() if kind == "scalar"}
        when = tuple((scope[name][1], tests) for name, tests in _condition(label, field, local))
        # A field waits for the answers its "when" tests: they become dependencies.
        for parent, _ in when:
            if (parent,) not in groups:
                groups.append((parent,))
    return tuple(groups), when


def _link(entries: Sequence[_Entry]) -> list[Decision]:
    """Decisions with their dependencies, conditions and thinking, the hidden effort questions included."""
    by_id = {entry.decision.name: entry.decision for entry in entries}
    compiled = []
    for entry in entries:
        decision, field = entry.decision, entry.field
        label = ".".join(decision.path) if decision.path else decision.name
        groups: list[tuple[str, ...]] = []
        whens: list = []
        found = False
        # The objects around the field first, outermost first, then its own.
        for outer_field, outer_scope, outer_label in entry.outer:
            outer_groups, outer_when = _dependencies(outer_label, outer_field, outer_scope, by_id)
            found |= outer_groups is not None
            groups.extend(outer_groups or ())
            whens.extend(outer_when or ())
        own_groups, own_when = _dependencies(label, field, entry.scope or {}, by_id)
        found |= own_groups is not None
        groups.extend(own_groups or ())
        whens.extend(own_when or ())
        groups_t = tuple(dict.fromkeys(groups)) if found else None
        dependencies = _flatten(groups_t) if groups_t is not None else None
        when = tuple(whens) or None
        thinking, budget = _thinking_settings(label, field)
        if thinking == "auto":
            effort = f"{decision.name}.thinking_effort"
            # Asked with the field's own dependencies, never thinks, and runs one layer before it.
            # The effort question shares the field's condition: a skipped field asks nothing.
            compiled.append(Decision(effort, EFFORT_QUESTION.format(question=_describe(decision)),
                                     tuple(THINKING_EFFORTS), depends_on=dependencies, thinking=False,
                                     effort_for=decision.name, when=when, dependency_groups=groups_t,
                                     path=(decision.path or (decision.name,)) + ("thinking_effort",)))
            compiled.append(replace(decision, depends_on=(dependencies or ()) + (effort,), effort_from=effort,
                                    when=when, dependency_groups=(groups_t or ()) + ((effort,),)))
            continue
        compiled.append(replace(decision, depends_on=dependencies, thinking=thinking, thinking_budget=budget,
                                when=when, dependency_groups=groups_t))
    return compiled


def _scalar(name: str, label: str, field: Mapping[str, Any], limit: int | None = None) -> Decision:
    """One scalar field's decision: a string, integer, number or boolean, open or from an enum.

    name is the decision's; label names the field in errors (a property's path). limit bounds an enum;
    None takes compile_json_schema's max_choices.
    """
    limit = _MAX_CHOICES.get() if limit is None else limit
    question = _question(label, field, f'Choose the value for "{label}".')
    # Decoding cannot hold a model to a range, so numeric bounds are not offered.
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in field:
            raise SchemaError(f"{keyword} is not supported (on {label!r}); use an enum for a fixed set of values")

    field_type = field.get("type")
    # ["string", "null"] and the like: one value type that may also be null.
    nullable = False
    if isinstance(field_type, list):
        kinds = [kind for kind in field_type if kind != "null"]
        if len(field_type) != 2 or len(kinds) != 1 or not isinstance(kinds[0], str):
            raise SchemaError(f'type for {label!r} must be one type or [type, "null"]')
        field_type, nullable = kinds[0], True
    if "levels" in field:
        return _score(name, label, question, field, field_type, nullable)
    enum = field.get("enum")
    descriptions: tuple[tuple[Any, str], ...] = ()
    if "choices" in field:
        enum, descriptions = _choices(label, field)
    permutations = field.get("permutations", 1)
    if "permutations" in field:
        # A boolean's choices are true and false, enum or not.
        if enum is None and field_type != "boolean":
            raise SchemaError(f"permutations for {label!r} requires an explicit enum")
        if not (permutations in ("auto", "all") or type(permutations) is int and permutations > 0):
            raise SchemaError(f"permutations for {label!r} must be 'auto', 'all' or a positive integer")
        if isinstance(enum, list) and permutations != "auto":
            count = math.factorial(len(enum))
            budget = count if permutations == "all" else min(permutations, count)
            if budget > MAX_PERMUTATIONS:
                raise SchemaError(f"permutations for {label!r} exceeds {MAX_PERMUTATIONS}; use a smaller integer budget")
    return_probabilities = field.get("return_probabilities", False)
    if type(return_probabilities) is not bool:
        raise SchemaError(f"return_probabilities for {label!r} must be a boolean")
    if "return_probabilities" in field and field_type != "boolean" and enum is None:
        raise SchemaError(f"return_probabilities for {label!r} is only supported for enum or boolean fields")
    # Probabilities are averaged over choice orders unless the field says otherwise: read in one
    # order, they lean to the choices listed first. permutations: 1 keeps the order as written.
    if return_probabilities and "permutations" not in field:
        permutations = "auto"
    if "x-score" in field:
        raise SchemaError(
            f"x-score for {label!r} is not supported; use a number enum"
        )
    if "x-other" in field:
        raise SchemaError(
            f"x-other for {label!r} is not supported; use a closed enum"
        )
    # Checking a length means decoding every token back to text, which slows every
    # string; answers stop at the client's text_max_tokens instead.
    if "maxLength" in field:
        raise SchemaError(f"maxLength is not supported (on {label!r}); string answers stop at "
                          "text_max_tokens, so ask for the length you want in instructions")
    if field_type == "string" and enum is None:
        for keyword in ("minLength", "pattern", "format"):
            if keyword in field:
                raise SchemaError(f"{keyword} is not supported for text fields")
        return Decision(name, question, (), "Text", text_type=True, nullable=nullable)
    if field_type == "boolean":
        values = ([True, False] + [None] * nullable) if enum is None else enum
        if not isinstance(values, list) or not values:
            raise SchemaError(f"enum for {label!r} must be a non-empty list")
        if any(type(value) is not bool and not (nullable and value is None) for value in values):
            raise SchemaError(f"boolean enum for {label!r} may contain only booleans")
        syntax = "Bool"
    elif field_type in {"integer", "number"} and enum is None:
        return Decision(
            name=name,
            question=question,
            choices=(),
            syntax="Integer" if field_type == "integer" else "Number",
            numeric_type=field_type,
            nullable=nullable,
        )
    elif field_type in {"string", "integer", "number"}:
        if enum is None:
            raise NotImplementedError(
                f"property {label!r} has type {field_type!r} without a finite enum"
            )
        if not isinstance(enum, list) or not enum:
            raise SchemaError(f"enum for {label!r} must be a non-empty list")
        if len(enum) > limit:
            raise SchemaError(
                f"enum for {label!r} has {len(enum)} values; "
                f"the maximum is {limit}"
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
                f"enum values for {label!r} do not match type {field_type!r}"
            )
        syntax = "Choice"
    else:
        raise NotImplementedError(
            f"property {label!r} has unsupported JSON Schema type {field_type!r}"
        )

    if len(values) > limit:
        raise SchemaError(
            f"enum for {label!r} has {len(values)} values; "
            f"the maximum is {limit}"
        )
    # One value leaves nothing to choose.
    if len(values) < 2:
        raise SchemaError(f"enum for {label!r} needs at least 2 values")
    if _has_duplicates(values):
        raise SchemaError(f"enum for {label!r} contains duplicate values")
    return Decision(name, question, tuple(values), syntax, return_probabilities=return_probabilities,
                    permutations=permutations, nullable=nullable, descriptions=descriptions)


def _score(name: str, label: str, question: str, field: Mapping[str, Any], field_type: Any,
           nullable: bool) -> Decision:
    """A score: a number from ordered "levels", [{"label": ..., "description": ...}, ...], lowest first.

    It is asked as a choice among the levels' labels; its value is the probability-weighted average
    of their indices, 0 to n - 1. The levels keep their order: the order is the scale, so a score
    takes no permutations.
    """
    if field_type != "number" or nullable:
        raise SchemaError(f'levels for {label!r} need "type": "number"')
    for key in ("enum", "choices"):
        if key in field:
            raise SchemaError(f"{label!r} takes levels or {key}, not both")
    levels = field["levels"]
    if not isinstance(levels, list) or not 2 <= len(levels) <= MAX_ENUM_CHOICES:
        raise SchemaError(f"levels for {label!r} must be a list of 2 to {MAX_ENUM_CHOICES} levels")
    labels, descriptions = [], []
    for index, level in enumerate(levels):
        if not isinstance(level, Mapping) or set(level) - {"label", "description"} \
                or not isinstance(level.get("label"), str) or not level["label"].strip():
            raise SchemaError(f'levels[{index}] for {label!r} must be {{"label": "...", "description": "..."}}')
        description = level.get("description")
        if description is not None and (not isinstance(description, str) or not description.strip()):
            raise SchemaError(f"levels[{index}].description for {label!r} must be a non-empty string")
        labels.append(level["label"].strip())
        if description is not None:
            descriptions.append((labels[-1], description.strip()))
    if len(set(labels)) != len(labels):
        raise SchemaError(f"levels for {label!r} have duplicate labels")
    return_probabilities = field.get("return_probabilities", False)
    if type(return_probabilities) is not bool:
        raise SchemaError(f"return_probabilities for {label!r} must be a boolean")
    if "permutations" in field:
        raise SchemaError(f"{label!r} is a score: its levels keep their order, so it takes no permutations")
    return Decision(name, question, tuple(labels), "Choice", return_probabilities=return_probabilities,
                    permutations=1, descriptions=tuple(descriptions), levels=tuple(labels))


def _choices(label: str, field: Mapping[str, Any]) -> tuple[list[Any], tuple[tuple[Any, str], ...]]:
    """An enum written as "choices": [{"value": ..., "description": ...}, ...]: its values, and the
    descriptions given. The values are then checked as an enum's are."""
    if "enum" in field:
        raise SchemaError(f"{label!r} takes enum or choices, not both")
    choices = field["choices"]
    if not isinstance(choices, list) or not choices:
        raise SchemaError(f"choices for {label!r} must be a non-empty list")
    values, descriptions = [], []
    for index, choice in enumerate(choices):
        if not isinstance(choice, Mapping) or "value" not in choice or set(choice) - {"value", "description"}:
            raise SchemaError(f"choices[{index}] for {label!r} must be {{\"value\": ..., \"description\": ...}}")
        description = choice.get("description")
        if description is not None and (not isinstance(description, str) or not description.strip()):
            raise SchemaError(f"choices[{index}].description for {label!r} must be a non-empty string")
        values.append(choice["value"])
        if description is not None:
            descriptions.append((choice["value"], description.strip()))
    return values, tuple(descriptions)


# "when" operators. A bare value means {"in": [value]}, and a list {"in": list}. "confidence"
# compares the dependency's confidence, not its answer: {"confidence": {"gte": 0.8}}.
COMPARISONS = {"gt": lambda a, b: a > b, "gte": lambda a, b: a >= b,
               "lt": lambda a, b: a < b, "lte": lambda a, b: a <= b}
OPERATORS = ("in", "not_in", "ne", *COMPARISONS, "confidence")


def _numeric(decision: Decision) -> bool:
    """An integer or number field, open or with a numeric enum."""
    if decision.numeric_type is not None or decision.levels:
        return True
    values = [value for value in decision.choices if value is not None]
    return decision.syntax == "Choice" and bool(values) and all(_is_finite_number(v) for v in values)


def _can_answer(decision: Decision, value: Any) -> bool:
    """Whether the field can give this answer, so a condition on it can ever hold."""
    if decision.levels:
        return False  # a score is a weighted average: compare it, do not match it
    if decision.choices:
        return any(_same_value(value, choice) for choice in decision.choices)
    if value is None:
        return decision.nullable
    if decision.numeric_type == "integer":
        return _is_finite_number(value) and float(value).is_integer()
    return decision.numeric_type is not None and _is_finite_number(value)


def _condition(name: str, field: Mapping[str, Any], by_name: Mapping[str, Decision]) -> tuple:
    """A field's "when" as ((field, ((operator, operand), ...)), ...), checked before anything runs.

    The fields it names become dependencies if depends_on leaves them out. Equality tests (a value, a list,
    in, not_in, ne) take answers the field can give: an enum value, true or false, a
    number, or null for a nullable field. gt, gte, lt and lte take numbers, on number
    fields. Open text fields cannot be conditions, except for null. confidence maps gt, gte, lt
    and lte to bounds from 0 to 1, on a field that returns probabilities.
    """
    when = field["when"]
    if not isinstance(when, Mapping) or not when:
        raise SchemaError(f"when for {name!r} must map dependencies to the answers that run it")
    condition = []
    for parent, test in when.items():
        if parent not in by_name:
            raise SchemaError(f"when for {name!r} names {parent!r}, which is not a field")
        source = by_name[parent]
        if not isinstance(test, Mapping):
            test = {"in": test if isinstance(test, list) else [test]}
        if not test:
            raise SchemaError(f"when for {name!r} has no test for {parent!r}")
        tests = []
        for operator, operand in test.items():
            if operator not in OPERATORS:
                raise SchemaError(f"when for {name!r}: unknown operator {operator!r} on {parent!r}; "
                                  f"use one of {list(OPERATORS)}")
            if operator == "confidence":
                tests.append((operator, _confidence_bounds(name, parent, source, operand)))
                continue
            if operator in COMPARISONS:
                if not _numeric(source):
                    raise SchemaError(f"when for {name!r}: {operator} needs a number field, and {parent!r} is not")
                if not _is_finite_number(operand):
                    raise SchemaError(f"when for {name!r}: {operator} on {parent!r} must be a number")
            else:
                if operator in ("in", "not_in"):
                    if not isinstance(operand, list) or not operand:
                        raise SchemaError(f"when for {name!r}: {operator} on {parent!r} must be a non-empty list")
                    operand = tuple(operand)
                for value in operand if operator != "ne" else (operand,):
                    if not _can_answer(source, value):
                        raise SchemaError(f"when for {name!r}: {value!r} is not an answer {parent!r} can give")
            tests.append((operator, operand))
        condition.append((parent, tuple(tests)))
    return tuple(condition)


def _confidence_bounds(name: str, parent: str, source: Any, bounds: Any) -> tuple[tuple[str, float], ...]:
    """A "confidence" test's bounds, as ((operator, bound), ...): only on a field with probabilities."""
    if not getattr(source, "return_probabilities", False):
        raise SchemaError(f"when for {name!r}: confidence on {parent!r} needs return_probabilities on {parent!r}")
    if not isinstance(bounds, Mapping) or not bounds:
        raise SchemaError(f"when for {name!r}: confidence on {parent!r} must map gt, gte, lt or lte to a number")
    tests = []
    for operator, bound in bounds.items():
        if operator not in COMPARISONS:
            raise SchemaError(f"when for {name!r}: confidence on {parent!r} takes {list(COMPARISONS)}, "
                              f"not {operator!r}")
        if not _is_finite_number(bound) or not 0 <= bound <= 1:
            raise SchemaError(f"when for {name!r}: confidence {operator} on {parent!r} must be a number from 0 to 1")
        tests.append((operator, bound))
    return tuple(tests)


def _passes(operator: str, operand: Any, answer: Any) -> bool:
    if operator == "in":
        return any(_same_value(answer, value) for value in operand)
    if operator == "not_in":
        return not any(_same_value(answer, value) for value in operand)
    if operator == "ne":
        return not _same_value(answer, operand)
    # A comparison holds only for a number: null, for one, is not above or below anything.
    return _is_finite_number(answer) and COMPARISONS[operator](answer, operand)


def condition_met(decision: Any, answers: Mapping[str, Any],
                  confidences: Mapping[str, float | None] | None = None) -> bool:
    """Whether a decision with a "when" runs, given its dependencies' answers and, for "confidence"
    tests, their confidences."""
    def holds(parent: str, operator: str, operand: Any) -> bool:
        if operator == "confidence":
            confidence = (confidences or {}).get(parent)
            return confidence is not None and all(COMPARISONS[op](confidence, bound) for op, bound in operand)
        return _passes(operator, operand, answers[parent])
    return all(holds(parent, operator, operand) for parent, tests in decision.when for operator, operand in tests)


def _thinking_settings(name: str, field: Mapping[str, Any]) -> tuple[bool | str | None, int | None]:
    """A field's thinking ("auto" included) and budget, from thinking, thinking_effort and thinking_budget."""
    thinking = field.get("thinking")
    if thinking is not None and type(thinking) is not bool and thinking != "auto":
        raise SchemaError(f'thinking for {name!r} must be a boolean or "auto"')
    budget = field.get("thinking_budget")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise SchemaError(f"thinking_budget for {name!r} must be a positive integer")
    if thinking == "auto" and budget is not None:
        raise SchemaError(f'thinking_budget for {name!r} cannot be set with thinking: "auto", '
                          "which picks the effort itself")
    if "thinking_effort" not in field:
        return thinking, budget
    effort = field["thinking_effort"]
    if not isinstance(effort, str) or effort not in THINKING_EFFORTS:
        raise SchemaError(f"thinking_effort for {name!r} must be one of {list(THINKING_EFFORTS)}")
    if thinking == "auto":
        raise SchemaError(f'thinking_effort for {name!r} cannot be set with thinking: "auto", '
                          "which picks the effort itself")
    if budget is not None:
        raise SchemaError(f"thinking_effort and thinking_budget for {name!r} both set the budget; use one")
    thinks = effort != "none"
    if thinking is not None and thinking != thinks:
        raise SchemaError(f"thinking: {str(thinking).lower()} for {name!r} contradicts thinking_effort {effort!r}")
    return thinks, THINKING_EFFORTS[effort]


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
