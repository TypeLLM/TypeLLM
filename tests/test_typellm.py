import json
import random
import unittest

import httpx

from typellm.runtime import numeric_pattern
from typellm import (
    GenerationTimeout,
    SGLangClient,
    SGLangError,
    SchemaError,
    TypeLLMClient,
    compile_json_schema,
)


def mock_http(handler):
    """A real SGLangClient whose connection pool answers through handler."""
    client = SGLangClient()
    client._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


class BrokenStream(httpx.SyncByteStream):
    """A response body that sends one chunk, then fails with error (if any)."""

    def __init__(self, chunk, error):
        self.chunk, self.error, self.closed = chunk, error, False

    def __iter__(self):
        yield self.chunk
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class FakeSGLang:
    def __init__(self, selected_ids=None, numbers=None):
        self.selected_ids = iter(selected_ids or [32])
        self.numbers = iter(numbers or [])  # replies to number requests; " 7}" when exhausted
        self.prompts = []
        self.cached_prefixes = []
        self.batch_prompts = []
        self.candidate_sets = []
        self.number_prefixes = []
        self.number_patterns = []

    def render_chat(self, messages, *, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def single_token(self, label):
        return ord(label), label

    def generate_numbers(self, prefixes, patterns, max_new_tokens, *, temperature=0, seed=0):
        self.number_prefixes.extend(prefixes)
        self.number_patterns.extend(patterns)
        return [next(self.numbers, " 7}") for _ in prefixes]

    def generate_fields(self, number_prefixes, patterns, number_max_tokens, text_prefixes, max_lengths,
                        *, temperature=0, number_seed=0, text_seed=0, after_key=False, nullable=None):
        numbers = self.generate_numbers(number_prefixes, patterns, number_max_tokens)
        texts = self.generate_texts(text_prefixes, max_lengths) if text_prefixes else []
        return numbers, texts

    def score_candidates(self, prefix, candidate_ids):
        self.prompts.append(prefix)
        self.candidate_sets.append(list(candidate_ids))
        selected = next(self.selected_ids)
        return (
            {token_id: (0.0 if token_id == selected else -10.0) for token_id in candidate_ids},
            {"cached_tokens": 0},
            0.0,
        )

    def cache_prefix(self, prefix):
        self.cached_prefixes.append(prefix)
        return {"cached_tokens": len(prefix)}

    def score_candidates_batch(self, prefixes, candidate_ids):
        self.batch_prompts.append(list(prefixes))
        scored = []
        for ids in candidate_ids:
            selected = next(self.selected_ids)
            scored.append(
                (
                    {
                        token_id: (0.0 if token_id == selected else -10.0)
                        for token_id in ids
                    },
                    {"cached_tokens": 1},
                )
            )
        return scored, 0.0


class RecordingSGLangClient(SGLangClient):
    def __init__(self):
        super().__init__()
        self.requests = []

    def _request(self, path, payload=None, *, allow_text=False):
        self.requests.append((path, payload))
        return [
            {
                "meta_info": {
                    "cached_tokens": 64,
                    "output_token_ids_logprobs": [
                        [[-0.1, 65, "A"], [-2.0, 66, "B"]]
                    ],
                }
            },
            {
                "meta_info": {
                    "cached_tokens": 64,
                    "output_token_ids_logprobs": [
                        [[-3.0, 65, "A"], [-0.2, 66, "B"]]
                    ],
                }
            },
        ]


class FakeChatTokenizer:
    chat_template = "template"
    eos_token_id = 248046
    eos_token = "<|im_end|>"

    def __init__(self):
        self.calls = []

    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered-chat"


class JsonSchemaCompilerTests(unittest.TestCase):
    def test_instructions_take_priority_over_description(self):
        for spec in [{"type": "boolean"}, {"type": "integer"},
                     {"type": "number"}, {"type": "string", "enum": ["a", "b"]}]:
            field = {**spec, "instructions": "Select the value.", "description": "Fallback"}
            [decision] = compile_json_schema({"type": "object", "properties": {"value": field}})
            self.assertEqual(decision.question, "Select the value.")

    def test_question_aliases_are_rejected_even_with_instructions(self):
        for old_key in ["question", "x-question"]:
            for extra in [{}, {"instructions": "New wording"}]:
                field = {"type": "boolean", old_key: "Old wording", **extra}
                with self.subTest(old_key=old_key, extra=extra):
                    with self.assertRaisesRegex(SchemaError, "use instructions"):
                        compile_json_schema({"type": "object", "properties": {"paid": field}})
                    client = TypeLLMClient()
                    for kwargs in [
                        {"questions": {"paid": field}},
                        {"schema": {"properties": [{**field, "type": "bool", "name": "paid"}]}},
                    ]:
                        with self.assertRaisesRegex(SchemaError, "use instructions"):
                            client.generate(context="Paid", **kwargs)

    def test_invalid_instructions_are_rejected(self):
        for value in [None, 1, True, [], {}]:
            with self.subTest(value=value), self.assertRaisesRegex(SchemaError, "instructions"):
                compile_json_schema({"type": "object", "properties": {
                    "x": {"type": "boolean", "instructions": value}
                }})

    def test_legacy_list_accepts_instructions(self):
        client = TypeLLMClient()
        client.sglang = FakeSGLang()
        schema = {"properties": [{"name": "paid", "type": "bool", "instructions": "Is it paid?"}]}
        self.assertEqual(client.compile_schema(schema)[0].question, "Is it paid?")

    def test_description_then_generated_question_fallback(self):
        decisions = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "integer",
                        "enum": [1, 2],
                        "description": "Pick a priority.",
                    },
                    "size": {"type": "number", "enum": [0.1, 0.5, 1.0]},
                },
            }
        )
        self.assertEqual(decisions[0].question, "Pick a priority.")
        self.assertEqual(decisions[1].question, 'Choose the value for "size".')

    def test_boolean(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "reimbursable": {
                        "type": "boolean",
                        "instructions": "Reimburse it?",
                    }
                },
            }
        )
        self.assertEqual(decision.choices, (True, False))
        self.assertEqual(decision.syntax, "Bool")

    def test_number_enum_preserves_float_values(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "scale": {"type": "number", "enum": [0.1, 0.5, 1.0]}
                },
            }
        )
        self.assertEqual(decision.choices, (0.1, 0.5, 1.0))

    def test_open_integer_and_number_compile_with_bounds(self):
        decisions = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "ratio": {"type": "number", "minimum": -1.0},
                },
            }
        )
        self.assertEqual(decisions[0].numeric_type, "integer")
        self.assertEqual(decisions[0].syntax, "Integer")
        self.assertEqual(decisions[0].choices, ())
        self.assertEqual((decisions[0].minimum, decisions[0].maximum), (0, 100))
        self.assertEqual(decisions[1].numeric_type, "number")

    def test_open_numeric_rejects_invalid_bounds(self):
        for field in (
            {"type": "integer", "minimum": "zero"},
            {"type": "number", "minimum": 2, "maximum": 1},
            {"type": "number", "maximum": float("inf")},
        ):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                compile_json_schema(
                    {"type": "object", "properties": {"value": field}}
                )

    def test_integers_beyond_float_range_are_valid_json_numbers(self):
        big = 10**400
        [bounded, choice] = compile_json_schema({"type": "object", "properties": {
            "bounded": {"type": "integer", "maximum": big},
            "choice": {"type": "number", "enum": [big, 0.5]},
        }})
        self.assertEqual(bounded.maximum, big)
        self.assertEqual(choice.choices, (big, 0.5))

    def test_x_score_is_rejected_in_favor_of_number_enum(self):
        with self.assertRaisesRegex(SchemaError, "x-score.*number enum"):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "confidence": {"type": "number", "x-score": True}
                    },
                }
            )

    def test_x_other_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "x-other.*not supported"):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "enum": ["known"],
                            "x-other": True,
                        }
                    },
                }
            )

    def test_enum_is_limited_to_twenty_four_values(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "integer", "enum": list(range(24))}
            },
        }
        [decision] = compile_json_schema(schema)
        self.assertEqual(len(decision.choices), 24)
        schema["properties"]["value"]["enum"].append(24)
        with self.assertRaisesRegex(SchemaError, "maximum is 24"):
            compile_json_schema(schema)

    def test_property_order_is_decision_order(self):
        decisions = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "second": {"type": "boolean"},
                    "first": {"type": "string", "enum": ["x"]},
                },
            }
        )
        self.assertEqual([decision.name for decision in decisions], ["second", "first"])

    def test_invalid_question_duplicate_enum_and_unsupported_type(self):
        with self.assertRaises(SchemaError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string", "enum": ["a"], "instructions": 3}
                    },
                }
            )
        with self.assertRaises(SchemaError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {"x": {"type": "number", "enum": [0.1, 0.1]}},
                }
            )
        with self.assertRaises(NotImplementedError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {"x": {"type": "array"}},
                }
            )

    def test_required_names_must_exist(self):
        with self.assertRaises(SchemaError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {"x": {"type": "boolean"}},
                    "required": ["missing"],
                }
            )


class QuestionsInterfaceTests(unittest.TestCase):
    def test_state_and_context_produce_identical_prompts(self):
        questions = {"paid": {"type": "boolean", "instructions": "Is it paid?"}}
        for text in ["Receipt", ""]:
            clients = [TypeLLMClient(), TypeLLMClient()]
            for client in clients:
                client.sglang = FakeSGLang([ord("A")])
            a = clients[0].generate(state=text, questions=questions)
            b = clients[1].generate(context=text, questions=questions)
            self.assertEqual(a, b)
            self.assertEqual(clients[0].last_prompts, clients[1].last_prompts)

    def test_state_rejects_conflicts_missing_and_invalid_types(self):
        from unittest.mock import Mock
        for kwargs in [{}, {"state": "x", "context": "x"},
                       {"state": "", "context": ""}, {"state": 1},
                       {"state": {}}, {"context": []}]:
            client = TypeLLMClient()
            client.sglang = Mock()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                client.generate(questions={"paid": {"type": "boolean"}}, **kwargs)
            self.assertEqual(client.sglang.mock_calls, [])

    def test_run_schema_supports_state_with_both_input_formats(self):
        from unittest.mock import patch
        from typellm import run_schema
        questions = {"paid": {"type": "boolean"}}
        for kwargs in [{"questions": questions}, {"schema": {"type": "object", "properties": questions}}]:
            with patch("typellm.runtime.SGLangClient", return_value=FakeSGLang([ord("A")])):
                self.assertEqual(run_schema(state="Paid", **kwargs), {"paid": True})
        with self.assertRaises(ValueError):
            run_schema("Paid", state="Paid", questions=questions)

    def test_questions_match_schema(self):
        questions = {
            "expense": {"type": "string", "enum": ["meal", "travel"], "instructions": "Classify."},
            "paid": {"type": "boolean", "instructions": "Is it paid?"},
        }
        clients = [TypeLLMClient(), TypeLLMClient()]
        for client in clients:
            client.sglang = FakeSGLang([ord("B"), ord("A")])
        new = clients[0].generate(context="Receipt", questions=questions)
        old = clients[1].generate(context="Receipt", schema={"type": "object", "properties": questions})
        self.assertEqual(new, {"expense": "travel", "paid": True})
        self.assertEqual(new, old)
        self.assertEqual(clients[0].last_prompts, clients[1].last_prompts)

    def test_questions_numeric_and_reserved_field_names(self):
        client = TypeLLMClient()
        client.sglang = FakeSGLang([ord("A")])
        result = client.generate(context="Seven", questions={
            "type": {"type": "integer", "instructions": "Extract the number."},
            "properties": {"type": "boolean", "instructions": "Is it seven?"},
        })
        self.assertEqual(result, {"type": 7, "properties": True})

    def test_questions_invalid_inputs_fail_before_network(self):
        for kwargs in [{}, {"questions": {}, "schema": {}}, {"questions": {}},
                       {"questions": []}, {"questions": "bad"}, {"questions": {"x": None}}]:
            client = TypeLLMClient()
            with self.subTest(kwargs=kwargs), self.assertRaises(SchemaError):
                client.generate(context="Context", **kwargs)

    def test_run_schema_keeps_positional_schema_and_accepts_questions(self):
        from unittest.mock import patch
        from typellm import run_schema
        questions = {"paid": {"type": "boolean"}}
        with patch("typellm.runtime.SGLangClient", return_value=FakeSGLang([ord("A")])):
            new = run_schema("Context", questions=questions)
        with patch("typellm.runtime.SGLangClient", return_value=FakeSGLang([ord("A")])):
            old = run_schema("Context", {"type": "object", "properties": questions})
        self.assertEqual(new, old)
        self.assertEqual(new, {"paid": True})


class ThinkingTests(unittest.TestCase):
    def test_default_thinking_budget_is_unset(self):
        from unittest.mock import Mock, patch
        from typellm import run_schema
        self.assertIsNone(SGLangClient().thinking_budget)
        self.assertIsNone(TypeLLMClient().sglang.thinking_budget)
        client = SGLangClient()
        client._chat_tokenizer = FakeChatTokenizer()
        client._context_length_cache = 8192
        client._request = Mock(return_value={"text": "Done.</think>"})
        client._finish_thinking("<think>")
        params = client._request.call_args.args[1]["sampling_params"]
        self.assertIn("max_new_tokens", params)
        self.assertGreater(params["max_new_tokens"], 2048)
        self.assertLess(params["max_new_tokens"], 8192)
        self.assertEqual(params["stop"], ["</think>"])
        with patch("typellm.runtime.TypeLLMClient") as factory:
            run_schema(context="x", questions={"flag": {"type": "boolean"}})
            self.assertIsNone(factory.call_args.kwargs["thinking_budget"])

    def make_client(self, response=None):
        from unittest.mock import Mock
        client = SGLangClient(thinking_budget=128)
        client._context_length_cache = 8192
        tokenizer = FakeChatTokenizer()
        tokenizer.apply_chat_template = Mock(return_value="assistant\n<think>\n")
        client._chat_tokenizer = tokenizer
        client._request = Mock(return_value=response or {"text": "Work done. </think>illegal answer"})
        return client

    def test_thinking_stops_before_constrained_answer(self):
        client = self.make_client()
        prompt = client.render_chat([], add_generation_prompt=True, thinking=True)
        self.assertEqual(prompt, "assistant\n<think>\nWork done. </think>\n\n")
        self.assertNotIn("illegal answer", prompt)
        params = client._request.call_args.args[1]["sampling_params"]
        self.assertEqual(params["max_new_tokens"], 128)
        self.assertEqual(params["stop"], ["</think>"])
        self.assertTrue(params["no_stop_trim"])
        self.assertTrue(client._chat_tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_length_stop_forces_closure_and_retains_reasoning(self):
        client = self.make_client({"text": "Partial reasoning", "meta_info": {"finish_reason": {"type": "length"}}})
        prompt = client.render_chat([], add_generation_prompt=True, thinking=True)
        self.assertIn("Partial reasoning", prompt)
        self.assertIn("I will now give the final answer.", prompt)
        self.assertTrue(prompt.endswith("</think>\n\n"))

    def test_context_reserve_limits_thinking_and_rejects_full_input(self):
        client = self.make_client()
        client.thinking_budget = None
        client._context_length_cache = 700
        prefix = "assistant\n<think>\n"
        client._finish_thinking(prefix)
        params = client._request.call_args.args[1]["sampling_params"]
        self.assertGreater(params["max_new_tokens"], 0)
        self.assertLess(params["max_new_tokens"] + len(prefix) + client.answer_reserve_tokens, 700)
        client._request.reset_mock()
        with self.assertRaisesRegex(SGLangError, "no room"):
            client._finish_thinking("x" * 700 + "<think>")
        client._request.assert_not_called()

    def test_context_discovery_and_caching(self):
        from unittest.mock import Mock
        for responses, expected, calls in [
            ([{"context_length": 8192, "server_args": {"context_length": 4096}}], 4096, 1),
            ([{"server_args": {"context_length": None}}, {"data": [{"id": "model", "max_model_len": 16384}]}], 16384, 2),
        ]:
            client = SGLangClient()
            client._request = Mock(side_effect=responses)
            self.assertEqual(client._context_length(), expected)
            self.assertEqual(client._context_length(), expected)
            self.assertEqual(client._request.call_count, calls)

    def test_abort_is_not_forced_even_with_closing_marker(self):
        for text in ("Partial", "Partial</think>"):
            client = self.make_client({"text": text, "meta_info": {"finish_reason": {"type": "abort"}}})
            with self.assertRaisesRegex(SGLangError, "aborted"):
                client.render_chat([], add_generation_prompt=True, thinking=True)

    def test_forced_thinking_keeps_all_final_decoders(self):
        stops = [{"type": "length"}, {"type": "stop", "matched": "<|im_end|>"}]
        for finish in stops:
            thinking = self.make_client({"text": "Partial", "meta_info": {"finish_reason": finish}})
            fake = FakeSGLang([ord("A")])
            render = fake.render_chat
            def render_with_thinking(messages, *, add_generation_prompt):
                prompt = render(messages, add_generation_prompt=add_generation_prompt)
                return thinking._finish_thinking(prompt + "<think>") if add_generation_prompt else prompt
            fake.render_chat = render_with_thinking
            def generate_texts(prefixes, limits, **kwargs):
                self.assertTrue(all(p.endswith('</think>\n\n{"t":') for p in prefixes))
                return ["blue"] * len(prefixes)
            fake.generate_texts = generate_texts
            client = TypeLLMClient()
            client.sglang = fake
            self.assertEqual(client.generate(context="test", questions={
                "n": {"type": "integer"}, "b": {"type": "boolean"}, "t": {"type": "string"},
            }), {"n": 7, "b": True, "t": "blue"})
            self.assertTrue(all(p.endswith('</think>\n\n{"n":') for p in fake.number_prefixes))

    def test_incomplete_or_empty_thinking_returns_no_answer(self):
        from typellm import SGLangError
        for response in [{"text": "not finished"}, {"text": "</think>"}, {"text": None}, []]:
            client = self.make_client()
            client._request.return_value = response
            with self.subTest(response=response), self.assertRaises(SGLangError):
                client.render_chat([], add_generation_prompt=True, thinking=True)

    def test_unsupported_template_does_not_generate(self):
        from typellm import SGLangError
        client = self.make_client()
        client._chat_tokenizer.apply_chat_template.return_value = "<think></think>"
        with self.assertRaisesRegex(SGLangError, "native chat template"):
            client.render_chat([], add_generation_prompt=True, thinking=True)
        client._request.assert_not_called()

    def test_always_thinking_template_reasons_even_when_thinking_is_off(self):
        client = self.make_client()
        prompt = client.render_chat([], add_generation_prompt=True)  # thinking not asked for
        self.assertEqual(prompt, "assistant\n<think>\nWork done. </think>\n\n")
        self.assertFalse(client._chat_tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_history_render_never_runs_thinking(self):
        client = self.make_client()
        client.render_chat([], add_generation_prompt=False)
        client._request.assert_not_called()
        self.assertFalse(client._chat_tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_public_configuration_and_validation(self):
        client = TypeLLMClient(thinking_budget=256)
        self.assertEqual(client.sglang.thinking_budget, 256)
        with self.assertRaises(TypeError):
            TypeLLMClient(thinking=True)  # thinking is chosen per field
        for kwargs in [{"thinking_budget": 0}, {"thinking_budget": True}, {"thinking_budget": 1.5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TypeLLMClient(**kwargs)


class JsonSchemaExecutionTests(unittest.TestCase):
    def test_chat_template_disables_thinking(self):
        client = SGLangClient()
        tokenizer = FakeChatTokenizer()
        client._chat_tokenizer = tokenizer

        rendered = client.render_chat(
            [{"role": "user", "content": "question"}],
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "rendered-chat")
        self.assertEqual(tokenizer.calls[0][1]["enable_thinking"], False)
        self.assertTrue(tokenizer.calls[0][1]["add_generation_prompt"])

    def test_template_bos_is_dropped_only_when_the_tokenizer_adds_its_own(self):
        client = SGLangClient()
        tokenizer = FakeChatTokenizer()
        tokenizer.bos_token, tokenizer.bos_token_id = "<s>", 1
        tokenizer.apply_chat_template = lambda messages, **kwargs: "<s>rendered-chat"
        client._chat_tokenizer = tokenizer
        for encoded, expected in (([1, 120], "rendered-chat"), ([120], "<s>rendered-chat")):
            with self.subTest(encoded=encoded):
                tokenizer.encode = lambda text, encoded=encoded: encoded
                self.assertEqual(client.render_chat([], add_generation_prompt=True), expected)

    def test_manual_choice_is_limited_to_twenty_four_values(self):
        from typellm import Choice

        with self.assertRaisesRegex(ValueError, "maximum is 24"):
            Choice(
                question="Too many?",
                choices={str(index): index for index in range(25)},
            )

    def test_eleven_value_number_enum_uses_a_through_k(self):
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [
                        0.0,
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.0,
                    ],
                },
            },
        }
        client = TypeLLMClient()
        fake = FakeSGLang([ord("K")])
        client.sglang = fake

        [compiled] = client.compile_schema(schema)
        self.assertEqual(list(compiled.choices), list("ABCDEFGHIJK"))
        self.assertEqual(compiled.choices["A"], 0.0)
        self.assertEqual(compiled.choices["K"], 1.0)
        self.assertIn(
            'Answer as {"score": "<label>"}.',
            compiled.opening_text(),
        )
        result = client.generate(context="context", schema=schema)
        self.assertEqual(result, {"score": 1.0})

    def test_semantic_result_and_prior_value_in_later_prompt(self):
        schema = {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "enum": [0.1, 0.5, 1.0],
                    "instructions": "Choose a scale.",
                },
                "enabled": {"type": "boolean", "depends_on": ["scale"]},
            },
            "required": ["scale", "enabled"],
        }
        from tests.test_dependencies import DependencyFake
        client = TypeLLMClient()
        fake = DependencyFake([ord("B"), ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"scale": 0.5, "enabled": True})
        self.assertNotIn("A", result)
        # The dependent field continues from the parent's closed answer.
        self.assertIn('<assistant>{"scale": "B"}</assistant>', fake.batch_prompts[1][0])

    def test_open_integer_is_generated_under_its_pattern(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "instructions": "How many items?",
                },
                "enabled": {"type": "boolean"},
            },
        }
        client = TypeLLMClient()
        fake = FakeSGLang([ord("A")], numbers=[" 42}"])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"count": 42, "enabled": True})
        self.assertEqual(fake.number_patterns, [numeric_pattern("integer", 32)])
        self.assertIn(
            'Type: integer, minimum 0, maximum 100\nInstructions: How many items?\nAnswer as {"count": <integer>}.',
            fake.number_prefixes[0],
        )
        self.assertTrue(fake.number_prefixes[0].endswith('<assistant>{"count":'))
        self.assertIn('<assistant>{"count": 42}</assistant>', client.last_prompts[0])

    def test_open_float_supports_sign_decimal_and_message_termination(self):
        schema = {
            "type": "object",
            "properties": {"temperature": {"type": "number"}},
        }
        client = TypeLLMClient()
        fake = FakeSGLang(numbers=[" -0.75"])  # ends at the end of the message
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"temperature": -0.75})
        self.assertIn(
            'Answer as {"temperature": <number>}. Do not use exponent notation.',
            client.last_prompts[0],
        )
        self.assertIn('<assistant>{"temperature": -0.75}</assistant>', client.last_prompts[0])

    def test_open_integer_beyond_float_range_preserves_value_and_bounds(self):
        big = 10**400
        for value in (big, -big):
            for bounds, error in (
                ({"minimum": value, "maximum": value}, None),
                ({"minimum": value + 1}, "below minimum"),
                ({"maximum": value - 1}, "above maximum"),
            ):
                with self.subTest(value=value, bounds=bounds):
                    client = TypeLLMClient(numeric_max_digits=401)
                    client.sglang = FakeSGLang(numbers=[f" {value}}}"])
                    questions = {"n": {"type": "integer", **bounds}}
                    if error is not None:
                        with self.assertRaisesRegex(ValueError, error):
                            client.generate(context="context", questions=questions)
                    else:
                        result = client.generate(context="context", questions=questions)
                        self.assertEqual(result, {"n": value})
                        self.assertIs(type(result["n"]), int)

    def test_open_number_still_rejects_float_overflow(self):
        for sign in ("", "-"):
            with self.subTest(sign=sign):
                client = TypeLLMClient(numeric_max_digits=401)
                client.sglang = FakeSGLang(numbers=[f" {sign}{10**400}}}"])
                with self.assertRaisesRegex(ValueError, "non-finite number"):
                    client.generate(context="context", questions={"n": {"type": "number"}})

    def test_text_prompt_keeps_non_ascii_field_names_readable(self):
        [compiled] = TypeLLMClient().compile_schema(
            {"type": "object", "properties": {"名称": {"type": "string"}}}
        )
        self.assertIn('Field: "名称"', compiled.opening_text())
        self.assertIn('Answer as {"名称": <string>}.', compiled.opening_text())

    def test_candidate_logprobs_match_shapes_recorded_from_a_real_server(self):
        from typellm import extract_candidate_logprobs
        # Recorded from SGLang 0.5.19 (/generate with token_ids_logprob=[32, 33, 34]).
        meta = {"output_token_ids_logprobs": [
            [[-13.3175, 32, "A"], [-14.4113, 33, "B"], [-14.2238, 34, "C"]]
        ]}
        self.assertEqual(
            extract_candidate_logprobs({"text": "1", "meta_info": meta}, [34, 32]),
            {34: -14.2238, 32: -13.3175},
        )
        # A server that computes no logprobs answers with an empty list.
        for empty in ([], [None], None):
            response = {"text": "1", "meta_info": {"output_token_ids_logprobs": empty}}
            with self.assertRaisesRegex(SGLangError, "not found for every requested"):
                extract_candidate_logprobs(response, [32])

    def test_labels_are_tokenized_without_special_tokens(self):
        from unittest.mock import Mock
        client = SGLangClient(model="model")
        client._request = Mock(side_effect=[{"tokens": [32]}, {"text": "A"}])
        self.assertEqual(client.single_token("A"), (32, "A"))
        path, payload = client._request.call_args_list[0].args
        self.assertEqual(path, "/v1/tokenize")
        self.assertIs(payload["add_special_tokens"], False)

    def test_read_timeout_is_reported_as_sglang_error(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        with self.assertRaisesRegex(SGLangError, "timed out") as caught:
            mock_http(handler)._request("/generate", {"text": "x"})
        self.assertIsInstance(caught.exception.__cause__, httpx.ReadTimeout)

    def test_truncated_response_is_reported_as_sglang_error(self):
        error = httpx.RemoteProtocolError("peer closed connection without sending complete message body")
        stream = BrokenStream(b'{"text":', error)
        client = mock_http(lambda request: httpx.Response(200, stream=stream))
        with self.assertRaisesRegex(SGLangError, "complete message body") as caught:
            client._request("/generate", {"text": "x"})
        self.assertIs(caught.exception.__cause__, error)
        self.assertTrue(stream.closed)

    def test_http_error_preserves_status_when_error_body_read_fails(self):
        for read_error in (httpx.RemoteProtocolError("incomplete body"), httpx.ReadTimeout("timed out")):
            with self.subTest(read_error=read_error):
                stream = BrokenStream(b"partial", read_error)
                client = mock_http(lambda request: httpx.Response(503, stream=stream))
                with self.assertRaisesRegex(
                    SGLangError, "HTTP 503.*Could not read error response"
                ) as caught:
                    client._request("/generate", {"text": "x"})
                self.assertEqual(caught.exception.status, 503)
                self.assertTrue(stream.closed)

    def test_http_error_preserves_body_and_closes_response(self):
        stream = BrokenStream(b"server unavailable", None)
        client = mock_http(lambda request: httpx.Response(503, stream=stream))
        with self.assertRaisesRegex(
            SGLangError, "HTTP 503: server unavailable"
        ) as caught:
            client._request("/generate", {"text": "x"})
        self.assertEqual(caught.exception.status, 503)
        self.assertTrue(stream.closed)

    def test_requests_reuse_one_connection_pool(self):
        client = mock_http(lambda request: httpx.Response(200, json={"ok": True}))
        pool = client._http()
        for _ in range(3):
            self.assertEqual(client._request("/generate", {"text": "x"}), {"ok": True})
        self.assertIs(client._http(), pool)
        client.close()
        self.assertTrue(pool.is_closed)
        self.assertIsNot(client._http(), pool)
    def test_info_endpoints_use_current_sglang_names(self):
        from unittest.mock import Mock
        client = SGLangClient()
        client._request = Mock(side_effect=[{"served_model_name": "model"}, {"context_length": 4096}])
        self.assertEqual(client._tokenizer_model(), "model")
        self.assertEqual(client._context_length(), 4096)
        self.assertEqual(
            [call.args[0] for call in client._request.call_args_list],
            ["/model_info", "/server_info"],
        )

    def test_info_endpoints_fall_back_to_old_names_only_on_404(self):
        paths = []

        def handler(request):
            paths.append(request.url.path)
            if "/get_" not in request.url.path:
                return httpx.Response(404, text="Not Found")
            return httpx.Response(200, json={"context_length": 4096})

        self.assertEqual(mock_http(handler)._context_length(), 4096)
        self.assertEqual(paths, ["/server_info", "/get_server_info"])

        paths.clear()

        def failing(request):
            paths.append(request.url.path)
            return httpx.Response(500, text="boom")

        with self.assertRaisesRegex(SGLangError, "HTTP 500"):
            mock_http(failing)._context_length()
        self.assertEqual(paths, ["/server_info"])

    def test_open_numeric_batch_preserves_schema_order(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
            },
        }
        client = TypeLLMClient()
        fake = FakeSGLang([ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"count": 7, "enabled": True})
        self.assertIn('<assistant>{"count": 7}</assistant>', client.last_prompts[0])
        self.assertIn('<assistant>{"enabled": "A"}</assistant>', client.last_prompts[1])

    def test_batch_prefills_once_and_forks_independent_questions(self):
        schema = {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "enum": [0.1, 0.5, 1.0],
                    "instructions": "Choose a scale.",
                },
                "enabled": {
                    "type": "boolean",
                    "instructions": "Enable it?",
                },
            },
        }
        client = TypeLLMClient()
        fake = FakeSGLang([ord("B"), ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"scale": 0.5, "enabled": True})
        self.assertEqual(fake.cached_prefixes, ["<user>context</user>"])
        self.assertEqual(len(fake.batch_prompts), 1)
        self.assertEqual(len(fake.batch_prompts[0]), 2)
        self.assertIn("Choose a scale.", fake.batch_prompts[0][0])
        self.assertIn("Enable it?", fake.batch_prompts[0][1])
        self.assertNotIn("value=0.5", fake.batch_prompts[0][1])
        self.assertEqual(len(client.last_prompts), 2)

    def test_native_batch_request_uses_per_prompt_candidate_ids(self):
        client = RecordingSGLangClient()

        scored, _ = client.score_candidates_batch(
            ["context + q1", "context + q2"],
            [[65, 66], [65, 66]],
        )

        path, payload = client.requests[0]
        self.assertEqual(path, "/generate")
        self.assertEqual(payload["text"], ["context + q1", "context + q2"])
        self.assertEqual(payload["token_ids_logprob"], [[65, 66], [65, 66]])
        self.assertEqual(scored[0][0], {65: -0.1, 66: -2.0})
        self.assertEqual(scored[1][0], {65: -3.0, 66: -0.2})


class HostedApiTests(unittest.TestCase):
    def client(self, handler, **options):
        client = TypeLLMClient(api_key="k", **options)
        client._transport = httpx.MockTransport(handler)
        return client

    def test_a_call_is_one_request_to_the_hosted_api(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={
                "id": "gen_1", "model": "typellm-latest", "result": {"total": 12.5},
                "thinking": {"total": "The receipt says 12.50."},
                "usage": {"input_tokens": 40, "thinking_tokens": 9}, "elapsed": 0.5,
            })

        client = self.client(handler, seed=7)
        questions = {"total": {"type": "number", "thinking": True}}
        image = "data:image/png;base64,iVBORw0KGgo="
        result = client.generate(context="Total: 12.50", questions=questions, images=[image], timeout=90)

        self.assertEqual(result, {"total": 12.5})
        [request] = seen
        self.assertEqual(str(request.url), "https://api.typellm.ai/v1/generate")
        self.assertEqual(request.headers["authorization"], "Bearer k")
        self.assertEqual(json.loads(request.content), {
            "context": "Total: 12.50", "questions": questions, "images": [image], "timeout": 90,
            # Each call draws its seed from the client's seeded stream.
            "options": {"mode": "argmax", "seed": random.Random(7).randrange(2**32)},
        })
        self.assertEqual((client.last_usage.input_tokens, client.last_usage.thinking_tokens), (40, 9))
        self.assertEqual(client.last_thinking, {"total": "The receipt says 12.50."})

    def test_hosted_temperature_only_applies_to_sampling(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={
                "result": {"a": True}, "usage": {"input_tokens": 1, "thinking_tokens": 0},
            })

        self.client(handler, temperature=0).generate(
            context="x", questions={"a": {"type": "boolean"}})
        self.client(handler, mode="sample", temperature=0.7).generate(
            context="x", questions={"a": {"type": "boolean"}})

        self.assertNotIn("temperature", bodies[0]["options"])
        self.assertEqual(bodies[1]["options"]["temperature"], 0.7)

    def test_constructor_timeout_sets_hosted_limit_with_per_call_override(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={
                "result": {"a": True}, "usage": {"input_tokens": 1, "thinking_tokens": 0},
            })

        questions = {"a": {"type": "boolean"}}
        client = self.client(handler, timeout=240)
        client.generate(context="x", questions=questions)
        client.generate(context="x", questions=questions, timeout=30)
        self.client(handler).generate(context="x", questions=questions)

        self.assertEqual([json.loads(request.content).get("timeout") for request in seen],
                         [240, 30, None])
        self.assertEqual([request.extensions["timeout"]["read"] for request in seen],
                         [300, 90, 180])
        self.assertEqual(TypeLLMClient().sglang.timeout, 120)

    def test_hosted_rejects_invalid_timeouts_before_sending(self):
        sent = []
        client = self.client(lambda request: sent.append(request))
        questions = {"a": {"type": "boolean"}}

        for timeout in (0, -1, True, "5", float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                client.generate(context="x", questions=questions, timeout=timeout)
        with self.assertRaises(ValueError):
            self.client(lambda request: sent.append(request), timeout=0).generate(
                context="x", questions=questions)
        self.assertEqual(sent, [])

    def test_api_errors_keep_their_status(self):
        for status, error in ((401, SGLangError), (429, SGLangError), (504, GenerationTimeout)):
            client = self.client(lambda request: httpx.Response(status, json={"error": {"message": "no"}}))
            with self.assertRaises(error) as caught:
                client.generate(context="x", questions={"a": {"type": "boolean"}})
            self.assertEqual(caught.exception.status, status)
        with self.assertRaises(ValueError):  # only local compilation takes a raw schema
            client.generate(context="x", schema={"type": "object", "properties": {"a": {"type": "boolean"}}})

    def test_hosted_compile_schema_requires_own_server(self):
        client = TypeLLMClient(api_key="k")
        with self.assertRaisesRegex(ValueError, "compile_schema needs your own server"):
            client.compile_schema({"type": "object", "properties": {"a": {"type": "boolean"}}})

    def test_hosted_invalid_responses_fail_clearly(self):
        questions = {"a": {"type": "boolean"}}
        client = TypeLLMClient(api_key="k")
        for response in (httpx.Response(200, text="not JSON"),
                         httpx.Response(200, json={"result": {"a": True}}),
                         httpx.Response(302, headers={"Location": "/elsewhere"})):
            with self.subTest(status=response.status_code):
                client._transport = httpx.MockTransport(lambda request: response)
                with self.assertRaises(SGLangError) as caught:
                    client.generate(context="x", questions=questions)
                self.assertEqual(caught.exception.status, response.status_code)


if __name__ == "__main__":
    unittest.main()
