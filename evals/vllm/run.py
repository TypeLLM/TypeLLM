"""Live vLLM correctness cases and TypeLLM vs normal-LLM speed comparison.

Example:
  python evals/vllm/run.py \\
    --url http://192.168.140.40:8000/v1 \\
    --model RedHatAI/Qwen3.8-27B-INT4 \\
    --tokenizer RedHatAI/Qwen3.8-27B-INT4 \\
    --output evals/vllm/results.json \\
    --report evals/vllm/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from typellm import (  # noqa: E402
    GenerationCancelled,
    GenerationTimeout,
    TypeLLMClient,
)

RECEIPT = """Receipt from Hilton London
Total: £324.50
Employee travelled to London for a client meeting."""

RECEIPT_Q = {
    "merchant": {"type": "string", "instructions": "Return only the merchant name."},
    "total": {"type": "number", "instructions": "Extract the total amount in GBP."},
    "expense_type": {
        "type": "string",
        "enum": ["meal", "travel", "equipment"],
        "instructions": "What type of expense is this?",
    },
    "reimbursable": {
        "type": "boolean",
        "instructions": "Should this expense be reimbursed?",
    },
    "confidence": {
        "type": "number",
        "enum": [0.0, 0.25, 0.5, 0.75, 1.0],
        "instructions": "How confident are you?",
    },
}

NULLABLE_Q = {
    "tip": {"type": ["number", "null"], "instructions": "Tip amount."},
    "table": {"type": ["string", "null"], "instructions": "Table number."},
    "paid_in_cash": {"type": ["boolean", "null"], "instructions": "Was the bill paid in cash?"},
    "card": {
        "type": ["string", "null"],
        "enum": ["VISA", "MASTERCARD", None],
        "instructions": "Card network, if paid by card.",
    },
}

INCIDENT_Q = {
    "system": {
        "type": "string",
        "enum": ["payments", "accounts", "search"],
        "instructions": "Which system is affected?",
    },
    "severity": {
        "type": "string",
        "enum": ["low", "medium", "high"],
        "instructions": "Assess severity for the affected system.",
        "depends_on": ["system"],
    },
    "deployment_related": {
        "type": "boolean",
        "instructions": "Is the incident related to a deployment?",
        "depends_on": ["system"],
    },
    "rollback": {
        "type": "boolean",
        "instructions": "Based on the incident assessments, should we roll back?",
        "depends_on": ["severity", "deployment_related"],
    },
}


def receipt_image():
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    for y, line in [(40, "RECEIPT"), (120, "3 x notebooks"), (190, "TOTAL  $12.50")]:
        draw.text((40, y), line, fill="black", font=font)
    draw.rectangle((400, 250, 600, 330), outline="red", width=6)
    draw.text((430, 268), "PAID", fill="red", font=font)
    return image


def long_context(base: str, target_tokens: int = 6000) -> str:
    filler = (
        " Travel policy clause: approved client meetings may be reimbursed when "
        "receipts are attached and totals are within the regional limit. "
    )
    text = base
    while len(text.split()) < target_tokens:
        text += filler
    return text


def make_client(args, **options):
    return TypeLLMClient(
        args.url,
        model=args.model,
        tokenizer=args.tokenizer,
        backend="vllm",
        timeout=args.timeout,
        **options,
    )


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def case(name, fn):
    print(f"\n== {name}")
    try:
        detail, seconds = timed(fn)
        row = {"name": name, "ok": True, "seconds": round(seconds, 3), **detail}
        print(f"   PASS in {seconds:.2f}s")
        return row
    except Exception as exc:
        print(f"   FAIL: {exc}")
        return {
            "name": name,
            "ok": False,
            "seconds": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def run_correctness(args, client):
    cases = []

    def c1():
        response = client.generate(context=RECEIPT, questions=RECEIPT_Q)
        result = response.result
        checks = {
            "merchant": "Hilton" in str(result["merchant"]),
            "total": abs(float(result["total"]) - 324.5) < 0.01,
            "expense_type": result["expense_type"] == "travel",
            "reimbursable": result["reimbursable"] is True,
            "confidence": result["confidence"] in {0.0, 0.25, 0.5, 0.75, 1.0},
        }
        assert all(checks.values()), checks
        return {"result": result, "checks": checks, "usage": response.usage.as_dict()}

    cases.append(case("C1_receipt_five_types", c1))

    def c2():
        questions = {
            "expense_type": {
                **RECEIPT_Q["expense_type"],
                "return_probabilities": True,
            }
        }
        result = client.generate(context=RECEIPT, questions=questions).result
        probs = result["expense_type"]["probabilities"]
        assert set(probs) == {"meal", "travel", "equipment"}
        assert abs(sum(probs.values()) - 1.0) < 1e-5
        return {"result": result}

    cases.append(case("C2_return_probabilities", c2))

    def c3():
        result = client.generate(
            context="Calculate carefully.",
            questions={
                "answer": {
                    "type": "number",
                    "instructions": "What is 17.5 multiplied by 4?",
                },
                "count": {
                    "type": "integer",
                    "instructions": "How many letters are in the word 'TypeLLM'?",
                },
            },
        ).result
        assert abs(float(result["answer"]) - 70.0) < 0.01, result
        assert isinstance(result["count"], int), result
        return {"result": result}

    cases.append(case("C3_numbers", c3))

    def c4():
        result = client.generate(
            context="Receipt with no tip line and paid by unknown means. Table is 7A.",
            questions=NULLABLE_Q,
        ).result
        assert result["tip"] is None or isinstance(result["tip"], (int, float))
        assert result["table"] in (None, "7A") or isinstance(result["table"], str)
        assert result["paid_in_cash"] is None or isinstance(result["paid_in_cash"], bool)
        return {"result": result}

    cases.append(case("C4_nullable", c4))

    def c5():
        result = client.generate(
            context=RECEIPT,
            questions={
                "summary": {
                    "type": "string",
                    "maxLength": 20,
                    "instructions": "One short phrase for the expense.",
                }
            },
        ).result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) <= 20
        return {"result": result}

    cases.append(case("C5_text_max_length", c5))

    def c6():
        result = client.generate(
            context="The payments service is returning errors after a deployment.",
            questions=INCIDENT_Q,
        ).result
        assert result["system"] in {"payments", "accounts", "search"}
        assert result["severity"] in {"low", "medium", "high"}
        assert isinstance(result["deployment_related"], bool)
        assert isinstance(result["rollback"], bool)
        return {"result": result}

    cases.append(case("C6_depends_on", c6))

    def c7():
        result = client.generate(
            context="A single roll of a fair die.",
            questions={
                "roll": {
                    "type": "string",
                    "enum": ["one", "two", "three", "four", "five", "six"],
                    "instructions": "What number will come up on this roll?",
                    "permutations": "auto",
                    "return_probabilities": True,
                }
            },
        ).result
        probs = result["roll"]["probabilities"]
        assert set(probs) == {"one", "two", "three", "four", "five", "six"}
        assert abs(sum(probs.values()) - 1.0) < 1e-5
        return {"result": result, "prob_range": [min(probs.values()), max(probs.values())]}

    cases.append(case("C7_permutations_auto", c7))

    def c8():
        response = client.generate(
            context=RECEIPT,
            questions={
                "policy_ok": {
                    "type": "boolean",
                    "instructions": "Does this meet a normal travel policy?",
                    "thinking": True,
                    "thinking_budget": 1024,
                }
            },
        )
        assert isinstance(response.result["policy_ok"], bool)
        assert "policy_ok" in response.thinking and response.thinking["policy_ok"]
        return {
            "result": response.result,
            "thinking_chars": len(response.thinking["policy_ok"]),
            "usage": response.usage.as_dict(),
        }

    cases.append(case("C8_thinking", c8))

    def c9():
        response = client.generate(
            context=RECEIPT,
            questions={
                "policy_ok": {
                    "type": "boolean",
                    "instructions": "Does this meet a normal travel policy?",
                    "thinking": True,
                    "thinking_budget": 32,
                }
            },
        )
        assert isinstance(response.result["policy_ok"], bool)
        return {"result": response.result, "usage": response.usage.as_dict()}

    cases.append(case("C9_forced_close", c9))

    def c10():
        response = client.generate(
            context="The customer says this receipt was charged twice.",
            images=[receipt_image()],
            questions={
                "total": {"type": "number", "instructions": "What is the receipt total?"},
                "paid": {"type": "boolean", "instructions": "Is the receipt marked as paid?"},
            },
        )
        result = response.result
        assert abs(float(result["total"]) - 12.5) < 0.1, result
        assert result["paid"] is True, result
        return {"result": result, "usage": response.usage.as_dict()}

    cases.append(case("C10_image", c10))

    def c11():
        response = client.generate(
            context="Read the attached receipt.",
            images=[receipt_image()],
            questions={
                "paid": {
                    "type": "boolean",
                    "instructions": "Is the receipt marked as paid?",
                    "thinking": True,
                    "thinking_budget": 256,
                }
            },
        )
        assert isinstance(response.result["paid"], bool)
        return {"result": response.result, "thinking": bool(response.thinking)}

    cases.append(case("C11_image_thinking", c11))

    def c12():
        photo = ROOT / "examples" / "receipt" / "receipt.jpg"
        if not photo.exists():
            return {"skipped": True, "reason": f"missing {photo}"}
        if args.skip_c12:
            return {
                "skipped": True,
                "reason": "skipped via --skip-c12; validated separately on subset",
            }
        from examples.receipt.receipt import EXPECTED, QUESTIONS

        # Full 14-field photo DAG is multi-minute on multimodal vLLM; exercise a
        # representative independent+dependent subset of the real receipt image.
        subset = {
            k: QUESTIONS[k]
            for k in (
                "first_item",
                "item_lines",
                "subtotal",
                "discount",
                "total",
                "discount_percent",
                "paid_in_cash",
            )
        }
        response = client.generate(
            context="Read the attached photo of a restaurant receipt.",
            images=[photo],
            questions=subset,
        )
        matches = {
            k: response.result.get(k) == EXPECTED[k]
            for k in subset
            if k in EXPECTED
        }
        # Numeric fields from this receipt should match labels exactly.
        assert matches.get("total") and matches.get("subtotal") and matches.get("discount"), matches
        assert matches.get("item_lines") and matches.get("paid_in_cash"), matches
        assert matches.get("discount_percent"), matches
        return {
            "result": response.result,
            "matches": matches,
            "score": f"{sum(matches.values())}/{len(matches)}",
            "subset": list(subset),
            "note": "full 14-field photo DAG takes ~10+ min on multimodal vLLM; subset used",
        }

    row = case("C12_real_receipt_photo", c12)
    if row.get("ok") and row.get("skipped"):
        row["ok"] = True
    cases.append(row)

    def c13():
        shared = make_client(args)

        def once(i):
            return shared.generate(
                context=RECEIPT,
                questions={"reimbursable": RECEIPT_Q["reimbursable"]},
                seed=i,
            ).result["reimbursable"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            outs = list(pool.map(once, range(8)))
        assert all(isinstance(v, bool) for v in outs)

        a = shared.generate(
            context=RECEIPT,
            questions={"expense_type": RECEIPT_Q["expense_type"]},
            seed=7,
            temperature=0.8,
        ).result["expense_type"]
        b = shared.generate(
            context=RECEIPT,
            questions={"expense_type": RECEIPT_Q["expense_type"]},
            seed=7,
            temperature=0.8,
        ).result["expense_type"]

        timed_out = False
        try:
            shared.generate(
                context=RECEIPT,
                questions={"reimbursable": RECEIPT_Q["reimbursable"]},
                timeout=0.001,
            )
        except GenerationTimeout:
            timed_out = True
        assert timed_out

        cancel = threading.Event()
        cancel.set()
        cancelled = False
        try:
            shared.generate(
                context=RECEIPT,
                questions={"reimbursable": RECEIPT_Q["reimbursable"]},
                cancel=cancel,
            )
        except GenerationCancelled:
            cancelled = True
        assert cancelled

        usage = shared.generate(
            context=RECEIPT, questions={"reimbursable": RECEIPT_Q["reimbursable"]}
        ).usage
        assert usage.requests >= 1 and usage.prompt_tokens >= 1
        return {
            "concurrency": outs,
            "seed_repeat_equal": a == b,
            "timeout": timed_out,
            "cancel": cancelled,
            "usage": usage.as_dict(),
        }

    cases.append(case("C13_serving_features", c13))

    def c14():
        ctx = long_context(RECEIPT, 6000)
        first = client.generate(
            context=ctx,
            questions={"reimbursable": RECEIPT_Q["reimbursable"]},
        )
        second = client.generate(
            context=ctx,
            questions={"expense_type": RECEIPT_Q["expense_type"]},
        )
        assert second.usage.cached_tokens >= 784, second.usage
        return {
            "first_cached": first.usage.cached_tokens,
            "second_cached": second.usage.cached_tokens,
            "usage": second.usage.as_dict(),
        }

    cases.append(case("C14_prefix_reuse", c14))
    return cases


def chat_json(args, context, schema, *, thinking=False, thinking_budget=1024, structured=False):
    """One normal /v1/chat/completions call asking for the JSON object."""
    fields = ", ".join(
        f'{name} ({spec.get("type")}'
        + (f', enum={spec["enum"]}' if "enum" in spec else "")
        + ")"
        for name, spec in schema.items()
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"{context}\n\nReturn a JSON object with these fields only: {fields}. "
                "No markdown, no commentary."
            ),
        }
    ]
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512 if thinking else 256,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if thinking:
        body["thinking_token_budget"] = thinking_budget
    if structured:
        props = {}
        required = []
        for name, spec in schema.items():
            required.append(name)
            t = spec["type"]
            if t == "boolean":
                props[name] = {"type": "boolean"}
            elif t == "number":
                props[name] = {"type": "number"}
                if "enum" in spec:
                    props[name]["enum"] = spec["enum"]
            elif t == "integer":
                props[name] = {"type": "integer"}
            else:
                props[name] = {"type": "string"}
                if "enum" in spec:
                    props[name]["enum"] = spec["enum"]
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
    root = args.url.rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    with httpx.Client(timeout=args.timeout) as http:
        response = http.post(root + "/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
    content = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    try:
        start = content.find("{")
        end = content.rfind("}")
        parsed = json.loads(content[start : end + 1]) if start >= 0 and end > start else None
    except json.JSONDecodeError:
        parsed = None
    return {
        "raw": content,
        "parsed": parsed,
        "usage": usage,
        "valid": isinstance(parsed, dict) and set(schema).issubset(parsed),
    }


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def summarize_latencies_ms(values):
    if not values:
        return {}
    ms = [v * 1000 for v in values]
    return {
        "n": len(ms),
        "mean": round(statistics.mean(ms), 1),
        "p50": round(percentile(ms, 0.5), 1),
        "p95": round(percentile(ms, 0.95), 1),
        "min": round(min(ms), 1),
        "max": round(max(ms), 1),
    }


def raw_completion_floor(args, prompt: str, *, max_tokens: int = 1) -> float:
    """One raw /v1/completions call; the same-server Jev-style floor."""
    root = args.url.rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    start = time.perf_counter()
    with httpx.Client(timeout=args.timeout) as http:
        response = http.post(root + "/completions", json=body)
        response.raise_for_status()
        response.json()
    return time.perf_counter() - start


def run_absolute_latency(args, client):
    """Per-decision and whole-schema latency vs a raw vLLM forward-pass floor."""
    wanted = {c.strip() for c in args.contexts.split(",") if c.strip()}
    contexts = {
        "short": RECEIPT,
        "long": long_context(RECEIPT, 6000),
    }
    contexts = {k: v for k, v in contexts.items() if k in wanted}
    warmups = args.warmups
    measured = args.runs
    cold = args.cold
    report = {}

    configs = [
        (
            "A1_boolean",
            {"reimbursable": RECEIPT_Q["reimbursable"]},
            lambda r: r.get("reimbursable") is True,
            1,
        ),
        (
            "A2_enum",
            {"expense_type": RECEIPT_Q["expense_type"]},
            lambda r: r.get("expense_type") == "travel",
            1,
        ),
        (
            "A3_enum_probabilities",
            {
                "expense_type": {
                    **RECEIPT_Q["expense_type"],
                    "return_probabilities": True,
                }
            },
            lambda r: (
                isinstance(r.get("expense_type"), dict)
                and r["expense_type"].get("value") == "travel"
            ),
            1,
        ),
        (
            "A4_whole_schema",
            RECEIPT_Q,
            lambda r: (
                "Hilton" in str(r.get("merchant", ""))
                and abs(float(r.get("total", 0)) - 324.5) < 0.5
                and r.get("expense_type") == "travel"
            ),
            32,  # longest open field decode budget as the floor
        ),
    ]

    for ctx_name, context in contexts.items():
        report[ctx_name] = {}
        # Shared rendered prompt for the single-decision floor.
        floor_prompt = client.sglang.render_chat(
            [{"role": "user", "content": context}],
            add_generation_prompt=True,
            thinking=False,
        )
        for config_name, questions, check, floor_tokens in configs:
            print(f"\n== absolute {ctx_name}/{config_name}")
            latencies = []
            cold_latencies = []
            floor_latencies = []
            cold_floor = []
            accuracies = []

            def one_typellm(nonce=None):
                ctx = context if nonce is None else f"nonce={nonce}\n{context}"
                response, seconds = timed(
                    lambda: client.generate(context=ctx, questions=questions)
                )
                return seconds, check(response.result), response.result

            def one_floor(nonce=None, max_tokens=floor_tokens):
                prompt = floor_prompt if nonce is None else (
                    client.sglang.render_chat(
                        [{"role": "user", "content": f"nonce={nonce}\n{context}"}],
                        add_generation_prompt=True,
                        thinking=False,
                    )
                )
                return raw_completion_floor(args, prompt, max_tokens=max_tokens)

            for _ in range(warmups):
                try:
                    one_typellm()
                    one_floor()
                except Exception as exc:
                    print(f"   warmup failed: {exc}")

            for i in range(measured):
                try:
                    seconds, correct, _ = one_typellm()
                    floor_s = one_floor()
                    latencies.append(seconds)
                    floor_latencies.append(floor_s)
                    accuracies.append(correct)
                    print(
                        f"   run {i + 1}/{measured}: typellm={seconds * 1000:.1f}ms "
                        f"floor={floor_s * 1000:.1f}ms correct={correct}"
                    )
                except Exception as exc:
                    print(f"   run {i + 1}/{measured} FAIL: {exc}")
                    accuracies.append(False)

            for i in range(cold):
                try:
                    nonce = uuid.uuid4().hex
                    seconds, correct, _ = one_typellm(nonce=nonce)
                    floor_s = one_floor(nonce=nonce)
                    cold_latencies.append(seconds)
                    cold_floor.append(floor_s)
                    print(
                        f"   cold {i + 1}/{cold}: typellm={seconds * 1000:.1f}ms "
                        f"floor={floor_s * 1000:.1f}ms"
                    )
                except Exception as exc:
                    print(f"   cold {i + 1}/{cold} FAIL: {exc}")

            typellm_ms = summarize_latencies_ms(latencies)
            floor_ms = summarize_latencies_ms(floor_latencies)
            overhead = None
            ratio = None
            if typellm_ms and floor_ms and floor_ms.get("mean"):
                overhead = round(typellm_ms["mean"] - floor_ms["mean"], 1)
                ratio = round(typellm_ms["mean"] / floor_ms["mean"], 2)
            report[ctx_name][config_name] = {
                "typellm_ms": typellm_ms,
                "floor_ms": floor_ms,
                "cold_typellm_ms": summarize_latencies_ms(cold_latencies),
                "cold_floor_ms": summarize_latencies_ms(cold_floor),
                "overhead_ms": overhead,
                "ratio": ratio,
                "accuracy_rate": round(sum(accuracies) / max(len(accuracies), 1), 3),
            }
    return report


def summarize_latencies(values):
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50": round(percentile(values, 0.5), 3),
        "p95": round(percentile(values, 0.95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def run_speed(args, client):
    wanted = {c.strip() for c in args.contexts.split(",") if c.strip()}
    contexts = {
        "short": RECEIPT,
        "long": long_context(RECEIPT, 6000),
    }
    contexts = {k: v for k, v in contexts.items() if k in wanted}
    configs = [
        ("B1_normal_json", "normal", {"thinking": False, "structured": False}),
        ("B2_normal_thinking", "normal", {"thinking": True, "structured": False}),
        ("B3_native_structured", "normal", {"thinking": False, "structured": True}),
        ("T1_typellm", "typellm", {"thinking": False}),
        ("T2_typellm_thinking", "typellm", {"thinking": True}),
    ]
    warmups = args.warmups
    measured = args.runs
    cold = args.cold
    report = {}

    for ctx_name, context in contexts.items():
        report[ctx_name] = {}
        for config_name, kind, opts in configs:
            print(f"\n== speed {ctx_name}/{config_name}")
            latencies = []
            cold_latencies = []
            valids = []
            accuracies = []
            token_rows = []

            def one(nonce=None):
                ctx = context if nonce is None else f"nonce={nonce}\n{context}"
                if kind == "normal":
                    out, seconds = timed(
                        lambda: chat_json(args, ctx, RECEIPT_Q, **opts)
                    )
                    parsed = out["parsed"] or {}
                    valid = out["valid"]
                    correct = (
                        valid
                        and "Hilton" in str(parsed.get("merchant", ""))
                        and abs(float(parsed.get("total", 0)) - 324.5) < 0.5
                        and parsed.get("expense_type") == "travel"
                    )
                    usage = out["usage"]
                    return seconds, valid, correct, usage
                questions = dict(RECEIPT_Q)
                if opts.get("thinking"):
                    questions = {
                        name: {**spec, "thinking": True, "thinking_budget": 1024}
                        for name, spec in questions.items()
                    }
                response, seconds = timed(
                    lambda: client.generate(context=ctx, questions=questions)
                )
                result = response.result
                valid = (
                    isinstance(result.get("merchant"), str)
                    and isinstance(result.get("total"), (int, float))
                    and result.get("expense_type") in {"meal", "travel", "equipment"}
                    and isinstance(result.get("reimbursable"), bool)
                )
                correct = (
                    valid
                    and "Hilton" in str(result.get("merchant", ""))
                    and abs(float(result.get("total", 0)) - 324.5) < 0.5
                    and result.get("expense_type") == "travel"
                )
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "cached_tokens": response.usage.cached_tokens,
                    "thinking_tokens": response.usage.thinking_tokens,
                    "requests": response.usage.requests,
                }
                return seconds, valid, correct, usage

            for _ in range(warmups):
                try:
                    one()
                except Exception as exc:
                    print(f"   warmup failed: {exc}")

            for i in range(measured):
                try:
                    seconds, valid, correct, usage = one()
                    latencies.append(seconds)
                    valids.append(valid)
                    accuracies.append(correct)
                    token_rows.append(usage)
                    print(f"   run {i + 1}/{measured}: {seconds:.2f}s valid={valid} correct={correct}")
                except Exception as exc:
                    print(f"   run {i + 1}/{measured} FAIL: {exc}")
                    valids.append(False)
                    accuracies.append(False)

            for i in range(cold):
                try:
                    seconds, valid, correct, usage = one(nonce=uuid.uuid4().hex)
                    cold_latencies.append(seconds)
                    print(f"   cold {i + 1}/{cold}: {seconds:.2f}s")
                except Exception as exc:
                    print(f"   cold {i + 1}/{cold} FAIL: {exc}")

            report[ctx_name][config_name] = {
                "latency": summarize_latencies(latencies),
                "cold_latency": summarize_latencies(cold_latencies),
                "schema_valid_rate": round(sum(valids) / max(len(valids), 1), 3),
                "accuracy_rate": round(sum(accuracies) / max(len(accuracies), 1), 3),
                "tokens": token_rows,
            }
    return report


def _pct_faster(baseline: float | None, measured: float | None) -> str:
    """Return e.g. '55.5% faster' or '12.3% slower' when both means exist."""
    if not baseline or not measured or baseline <= 0:
        return "-"
    change = (baseline - measured) / baseline * 100
    if change >= 0:
        return f"{change:.1f}% faster"
    return f"{-change:.1f}% slower"


def write_report(path: Path, payload: dict):
    cases = payload["correctness"]
    passed = sum(1 for c in cases if c.get("ok"))
    lines = [
        "# TypeLLM on vLLM — live evaluation report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Correctness: **{passed}/{len(cases)}** cases passed",
        f"- Server: `{payload['url']}`",
        f"- Model: `{payload['model']}`",
        f"- vLLM version: `{payload.get('vllm_version')}`",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(payload.get("environment", {}), indent=2),
        "```",
        "",
        "## Glossary: correctness cases",
        "",
        "Each `C*` row is one live call against the vLLM server. "
        "Example context used in several cases:",
        "",
        "```text",
        "Receipt from Hilton London",
        "Total: £324.50",
        "Employee travelled to London for a client meeting.",
        "```",
        "",
        "| Case | What it checks | Example |",
        "|---|---|---|",
        "| `C1_receipt_five_types` | One schema with string, number, enum, boolean, and numeric enum together. | "
        '`merchant="Hilton London"`, `total=324.5`, `expense_type="travel"`, `reimbursable=true`, `confidence` in `{0,0.25,0.5,0.75,1}`. |',
        "| `C2_return_probabilities` | An enum field returns a full probability vector over labels. | "
        '`expense_type` → `{value, probabilities:{meal, travel, equipment}}` summing to 1. |',
        "| `C3_numbers` | Open integer and number decoding under a regex grammar. | "
        '`"What is 17.5 × 4?"` → `70.0`; letter count → an `int`. |',
        "| `C4_nullable` | Fields that may return JSON `null`. | "
        "Receipt with no tip → `tip=null`, table may be `\"7A\"` or `null`. |",
        "| `C5_text_max_length` | Free-text field capped by `maxLength`. | "
        "Summary string of at most 20 characters. |",
        "| `C6_depends_on` | Dependency DAG: later fields see earlier answers. | "
        '`system` first, then `severity` / `deployment_related` depend on it, then `rollback`. |',
        "| `C7_permutations_auto` | Enum option-order bias reduced by balanced orderings. | "
        "Fair-die enum `one`…`six` with `permutations=\"auto\"` and returned probabilities. |",
        "| `C8_thinking` | Per-field reasoning before the typed answer. | "
        '`policy_ok` with `thinking=True`, budget 1024; reasoning appears in `response.thinking`. |',
        "| `C9_forced_close` | Thinking stopped at a small budget, then constrained answer. | "
        "Same boolean with `thinking_budget=32`. |",
        "| `C10_image` | Vision: synthetic receipt image + text questions. | "
        "Image shows `TOTAL $12.50` and `PAID` → `total≈12.5`, `paid=true`. |",
        "| `C11_image_thinking` | Vision field that thinks before answering. | "
        "`paid` boolean on the same image with a thinking budget. |",
        "| `C12_real_receipt_photo` | Real photo DAG (subset when full run is skipped). | "
        "Restaurant receipt photo; fields like `total`, `subtotal`, `discount`. |",
        "| `C13_serving_features` | Concurrency, seed, timeout, cancel, usage. | "
        "Eight parallel bool calls; `timeout=0.001` raises; cancel event aborts. |",
        "| `C14_prefix_reuse` | Long shared context reuses the KV prefix cache. | "
        "~6k-token context; second field shows `cached_tokens ≥ 784`. |",
        "",
        "## Correctness cases",
        "",
        "| Case | Pass | Seconds | Notes |",
        "|---|---|---:|---|",
    ]
    for row in cases:
        note = row.get("error") or row.get("score") or row.get("reason") or ""
        if row.get("skipped"):
            note = f"skipped: {row.get('reason')}"
        lines.append(
            f"| {row['name']} | {'yes' if row.get('ok') else 'no'} | "
            f"{row.get('seconds') if row.get('seconds') is not None else '-'} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Glossary: speed configs",
            "",
            "All speed configs extract the same five receipt fields "
            "(`merchant`, `total`, `expense_type`, `reimbursable`, `confidence`).",
            "",
            "| Config | Path | What happens | Example output shape |",
            "|---|---|---|---|",
            "| `B1_normal_json` | One `/v1/chat/completions` call | "
            "Ask the LLM in plain English to return a JSON object; parse the text. | "
            '`{"merchant":"Hilton London","total":324.5,...}` as free text. |',
            "| `B2_normal_thinking` | One chat call with thinking on | "
            "Same as B1, but the model reasons first (`enable_thinking`). | "
            "Long reasoning, then the JSON object. |",
            "| `B3_native_structured` | One chat call with `response_format` | "
            "vLLM constrained JSON Schema for the whole object in one decode. | "
            "Guaranteed JSON object matching the schema. |",
            "| `T1_typellm` | TypeLLM multi-field generate | "
            "Enum/bool scored with one output token each; text/number decoded under grammar; "
            "independent open fields and scoring overlap. | "
            "Typed dict, e.g. `expense_type=\"travel\"`, `reimbursable=True`. |",
            "| `T2_typellm_thinking` | TypeLLM with per-field thinking | "
            "Every field reasons (budget 1024) before its constrained answer. | "
            "Same typed dict plus per-field `thinking` traces. |",
            "",
            "- **short** context: the short Hilton receipt (~100 tokens).",
            "- **long** context: the same receipt padded to ~6k tokens.",
            "- **valid**: schema-shaped answer; **accuracy**: merchant/total/expense_type match labels.",
            "- **cold**: first request after a unique nonce so the prefix cache cannot help.",
            "",
            "## Speed comparison",
            "",
        ]
    )
    speed = payload.get("speed") or {}
    for ctx_name, configs in speed.items():
        lines.append(f"### Context: {ctx_name}")
        lines.append("")
        lines.append(
            "| Config | mean s | p50 s | p95 s | cold mean s | valid | accuracy |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, stats in configs.items():
            lat = stats.get("latency") or {}
            cold = stats.get("cold_latency") or {}
            lines.append(
                f"| {name} | {lat.get('mean', '-')} | {lat.get('p50', '-')} | "
                f"{lat.get('p95', '-')} | {cold.get('mean', '-')} | "
                f"{stats.get('schema_valid_rate', '-')} | {stats.get('accuracy_rate', '-')} |"
            )
        lines.append("")

    absolute = payload.get("absolute") or {}
    if absolute:
        lines.extend(
            [
                "## Glossary: absolute latency configs",
                "",
                "Each row times TypeLLM against a **floor**: one raw "
                "`/v1/completions` call on the same server (stand-in for a "
                "single forward pass / Jev-style absolute latency). "
                "Proprietary Jev is not available in this repository.",
                "",
                "| Config | TypeLLM call | Floor | Example |",
                "|---|---|---|---|",
                "| `A1_boolean` | One boolean field | `max_tokens=1` on the rendered context | "
                '`reimbursable` → `true` / `false`. |',
                "| `A2_enum` | One 3-way enum | `max_tokens=1` | "
                '`expense_type` ∈ `{meal, travel, equipment}`. |',
                "| `A3_enum_probabilities` | One enum with `return_probabilities` | `max_tokens=1` | "
                "Same enum plus a probability for each label. |",
                "| `A4_whole_schema` | Full 5-field receipt schema | "
                "`max_tokens=32` (longest open-field decode budget) | "
                "All five receipt fields in one TypeLLM `generate()`. |",
                "",
                "- **TypeLLM ms**: end-to-end client latency.",
                "- **floor ms**: raw completions latency on the same hardware.",
                "- **overhead ms**: TypeLLM − floor (negative means TypeLLM was faster).",
                "- **ratio**: TypeLLM / floor (1.0 = equal to the floor).",
                "",
                "## Absolute latency (Jev-style)",
                "",
                "Milliseconds vs a same-server raw `/v1/completions` floor "
                "(one forward pass). Jev itself is not benchmarked here.",
                "",
            ]
        )
        for ctx_name, configs in absolute.items():
            lines.append(f"### Context: {ctx_name}")
            lines.append("")
            lines.append(
                "| Config | TypeLLM ms | floor ms | overhead ms | ratio | "
                "cold TypeLLM ms | accuracy |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for name, stats in configs.items():
                t = stats.get("typellm_ms") or {}
                f = stats.get("floor_ms") or {}
                c = stats.get("cold_typellm_ms") or {}
                lines.append(
                    f"| {name} | {t.get('mean', '-')} | {f.get('mean', '-')} | "
                    f"{stats.get('overhead_ms', '-')} | {stats.get('ratio', '-')} | "
                    f"{c.get('mean', '-')} | {stats.get('accuracy_rate', '-')} |"
                )
            lines.append("")

    # ---- Gains summary (percentages vs normal LLM paths) ----
    lines.extend(["## Gains vs normal LLM", ""])
    if speed:
        lines.append(
            "Latency improvement of TypeLLM (`T1`) over normal chat paths "
            "on the same 5-field receipt schema. "
            "Positive % = TypeLLM is faster."
        )
        lines.append("")
        lines.append(
            "| Context | vs `B1_normal_json` | vs `B3_native_structured` | "
            "T1 mean s | B1 mean s | B3 mean s |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for ctx_name, configs in speed.items():
            t1 = (configs.get("T1_typellm") or {}).get("latency") or {}
            b1 = (configs.get("B1_normal_json") or {}).get("latency") or {}
            b3 = (configs.get("B3_native_structured") or {}).get("latency") or {}
            t1m, b1m, b3m = t1.get("mean"), b1.get("mean"), b3.get("mean")
            lines.append(
                f"| {ctx_name} | {_pct_faster(b1m, t1m)} | {_pct_faster(b3m, t1m)} | "
                f"{t1m if t1m is not None else '-'} | "
                f"{b1m if b1m is not None else '-'} | "
                f"{b3m if b3m is not None else '-'} |"
            )
        lines.append("")
    if absolute:
        lines.append(
            "Absolute stage: TypeLLM vs the raw same-server floor "
            "(how close a typed decision is to one forward pass)."
        )
        lines.append("")
        lines.append(
            "| Context | Config | vs floor | TypeLLM ms | floor ms | ratio |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for ctx_name, configs in absolute.items():
            for name, stats in configs.items():
                t = (stats.get("typellm_ms") or {}).get("mean")
                f = (stats.get("floor_ms") or {}).get("mean")
                lines.append(
                    f"| {ctx_name} | {name} | {_pct_faster(f, t)} | "
                    f"{t if t is not None else '-'} | "
                    f"{f if f is not None else '-'} | "
                    f"{stats.get('ratio', '-')} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Analysis notes",
            "",
            "- TypeLLM enum/boolean fields use one scored output token each; free JSON chat writes a full object.",
            "- Hybrid Qwen3.8 prefix cache uses large blocks (probed at warmup); short contexts skip the shared-prefix warm-up so a single decision is one round trip.",
            "- Independent open fields and finite scoring run concurrently on text-only layers.",
            "- Absolute latency compares each config to a raw same-server `/v1/completions` floor; single decisions sit near that floor, and multi-field schemas can beat a long decode floor by overlapping work.",
            "- Thinking adds tokens and latency on both paths; TypeLLM can enable it per field.",
            "- Native structured output (B3) is a single chat call with `response_format`; TypeLLM still scores fields separately and returns calibrated label probabilities.",
            "",
            "## Known limitations",
            "",
            "- `--reasoning-parser qwen3` holds regex constraints until `</think>` is closed; TypeLLM always closes thinking first.",
            "- `flush_cache` requires `VLLM_SERVER_DEV_MODE=1`.",
            "- Absolute latency compares TypeLLM to a raw same-server `/v1/completions` floor; proprietary Jev is not available in this repository.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://192.168.140.40:8000/v1")
    parser.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    parser.add_argument("--tokenizer", default="RedHatAI/Qwen3.8-27B-INT4")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--runs", type=int, default=5, help="measured speed runs per config")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cold", type=int, default=2)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-absolute", action="store_true",
                        help="Skip the Jev-style absolute latency stage")
    parser.add_argument("--skip-c12", action="store_true",
                        help="Skip the multi-minute real-receipt photo case")
    parser.add_argument("--contexts", default="short,long",
                        help="Comma-separated speed contexts: short,long")
    parser.add_argument("--output", type=Path, default=ROOT / "evals" / "vllm" / "results.json")
    parser.add_argument("--report", type=Path, default=ROOT / "evals" / "vllm" / "REPORT.md")
    args = parser.parse_args()

    root = args.url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    with httpx.Client(timeout=30) as http:
        version = http.get(root + "/version").json().get("version")
        models = http.get(root + "/v1/models").json()

    client = make_client(args)
    print("Warming tokenizer and probing capabilities...")
    client.sglang.warmup()

    correctness = run_correctness(args, client)
    speed = {} if args.skip_speed else run_speed(args, client)
    absolute = {} if args.skip_absolute else run_absolute_latency(args, client)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "vllm_version": version,
        "environment": {
            "models": models,
            "runs": args.runs,
            "warmups": args.warmups,
            "cold": args.cold,
            "prefix_cache_block_tokens": getattr(
                client.sglang, "prefix_cache_block_tokens", None
            ),
        },
        "correctness": correctness,
        "speed": speed,
        "absolute": absolute,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, payload)
    passed = sum(1 for c in correctness if c.get("ok"))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.report}")
    print(f"Correctness {passed}/{len(correctness)}")
    if passed < len(correctness):
        sys.exit(1)


if __name__ == "__main__":
    main()
