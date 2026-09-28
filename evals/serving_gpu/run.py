"""Live GPU checks for serving: output parity with main, and the new per-call features.

Phases, each writing JSON to --output:
  parity    fixed cases across every field type, with and without thinking
  perf      sequential latency of a numeric-heavy schema
  features  shared-client concurrency, per-call seed, usage, timeout and cancel
  load      throughput of one shared client at several thread counts

parity and perf run against whichever checkout --root points at, so the same
script compares main with the branch.
"""
import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["parity", "perf", "features", "load"])
parser.add_argument("--root", required=True, help="TypeLLM checkout to import")
parser.add_argument("--output", required=True)
parser.add_argument("--url", default="http://127.0.0.1:30000")
parser.add_argument("--model", default="qwen3.8-27b")
parser.add_argument("--tokenizer", default="RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead")
args = parser.parse_args()
sys.path.insert(0, args.root)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import typellm  # noqa: E402
from typellm import TypeLLMClient  # noqa: E402


def client(**options):
    return TypeLLMClient(args.url, model=args.model, tokenizer=args.tokenizer, **options)


def receipt_image():
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default(size=36)
    for y, line in [(40, "RECEIPT"), (120, "3 x notebooks"), (190, "TOTAL  $12.50")]:
        draw.text((40, y), line, fill="black", font=font)
    draw.rectangle((400, 250, 600, 330), outline="red", width=6)
    draw.text((430, 268), "PAID", fill="red", font=font)
    return image


RECEIPT = """Receipt from Hilton London
Total: £324.50
Employee travelled to London for a client meeting."""

RECEIPT_Q = {
    "merchant": {"type": "string", "instructions": "Return only the merchant name."},
    "total": {"type": "number", "instructions": "Extract the total amount in GBP."},
    "expense_type": {"type": "string", "enum": ["meal", "travel", "equipment"],
                     "instructions": "What type of expense is this?", "return_probabilities": True},
    "reimbursable": {"type": "boolean", "instructions": "Should this expense be reimbursed?"},
    "confidence": {"type": "number", "enum": [0.0, 0.25, 0.5, 0.75, 1.0],
                   "instructions": "How confident are you?"},
}
NULLABLE_Q = {
    "tip": {"type": ["number", "null"], "instructions": "Tip amount."},
    "table": {"type": ["string", "null"], "instructions": "Table number."},
    "paid_in_cash": {"type": ["boolean", "null"], "instructions": "Was the bill paid in cash?"},
    "card": {"type": ["string", "null"], "enum": ["VISA", "MASTERCARD", None],
             "instructions": "Card network, if paid by card."},
}
INCIDENT_Q = {
    "system": {"type": "string", "enum": ["payments", "accounts", "search"],
               "instructions": "Which system is affected?"},
    "severity": {"type": "string", "enum": ["low", "medium", "high"],
                 "instructions": "Assess severity for the affected system.", "depends_on": ["system"]},
    "deployment_related": {"type": "boolean", "instructions": "Is the incident related to a deployment?",
                           "depends_on": ["system"]},
    "rollback": {"type": "boolean", "instructions": "Based on the incident assessments, should we roll back?",
                 "depends_on": ["severity", "deployment_related"]},
}
DIE_Q = {"roll": {"type": "string", "enum": ["one", "two", "three", "four", "five", "six"],
                  "instructions": "What number will come up on this roll?",
                  "permutations": "auto", "return_probabilities": True}}
MATH_Q = {"answer": {"type": "number", "instructions": "What is 17.5 multiplied by 4?"},
          "count": {"type": "integer", "instructions": "How many letters are in the word 'serving'?"}}
IMAGE_Q = {
    "item": {"type": "string", "enum": ["notebooks", "pencils", "coffee"], "instructions": "Which item is on the receipt?"},
    "quantity": {"type": "integer", "instructions": "How many were bought?"},
    "total": {"type": "number", "instructions": "What is the total in dollars?"},
    "paid": {"type": "boolean", "instructions": "Is the receipt stamped PAID?"},
    "check": {"type": "boolean", "instructions": "Is quantity greater than 2 and the receipt paid?",
              "depends_on": ["quantity", "paid"]},
}

def thinks(questions):
    return {name: {**field, "thinking": True} for name, field in questions.items()}


CASES = [
    ("receipt", {}, dict(context=RECEIPT, questions=RECEIPT_Q)),
    ("nullable", {}, dict(context="Table 7A. Paid by card. No tip was added.", questions=NULLABLE_Q)),
    ("incident", {}, dict(context="The payments service is returning errors after a deployment.", questions=INCIDENT_Q)),
    ("die", {}, dict(context="A single roll of a fair die.", questions=DIE_Q)),
    ("math", {}, dict(context="Calculate the requested value accurately.", questions=MATH_Q)),
    ("image", {}, dict(context="Read the attached receipt.", images="receipt", questions=IMAGE_Q)),
    ("receipt+thinking", {"thinking_budget": 512}, dict(context=RECEIPT, questions=thinks(RECEIPT_Q))),
    ("incident+thinking", {"thinking_budget": 512},
     dict(context="The payments service is returning errors after a deployment.", questions=thinks(INCIDENT_Q))),
    ("math+thinking", {"thinking_budget": 512},
     dict(context="Calculate the requested value accurately.", questions=thinks(MATH_Q))),
]


def call(options, request):
    request = dict(request)
    if request.get("images") == "receipt":
        request["images"] = [receipt_image()]
    c = client(**options)
    start = time.perf_counter()
    try:
        result = c.generate(**request).result
        error = None
    except Exception as exc:  # recorded, not raised: parity compares failures too
        result, error = None, f"{type(exc).__name__}: {exc}"
    return result, error, time.perf_counter() - start, c


def parity():
    rows = []
    for name, options, request in CASES:
        result, error, elapsed, _ = call(options, request)
        rows.append({"case": name, "result": result, "error": error, "elapsed": round(elapsed, 3)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return {"rows": rows}


NUMERIC_Q = {f"n{i}": {"type": "number", "instructions": f"What is value {i} on the sheet?"} for i in range(5)}
SHEET = "Sheet: value 0 is 1234.5, value 1 is 88, value 2 is 0.125, value 3 is 42017, value 4 is 3.14159."


def perf():
    c = client()
    c.generate(context=SHEET, questions=NUMERIC_Q).result  # warm tokenizer and prefix cache
    times = []
    for _ in range(10):
        start = time.perf_counter()
        result = c.generate(context=SHEET, questions=NUMERIC_Q).result
        times.append(time.perf_counter() - start)
    return {"result": result, "median": statistics.median(times), "times": times}


def features():
    out = {}
    shared = client()
    shared.sglang.warmup()

    # 1. One shared client, many threads: every answer and prompt stays with its call.
    questions = {"invoice": {"type": "integer", "instructions": "What is the invoice number?"},
                 "quantity": {"type": "integer", "instructions": "How many units were ordered?"},
                 "urgent": {"type": "boolean", "instructions": "Is the order marked urgent?"}}
    cases = [(1000 + n, n % 9 + 1, n % 2 == 0) for n in range(48)]

    def context_for(case):
        number, quantity, urgent = case
        return f"Invoice #{number}. Ordered {quantity} units. {'URGENT order.' if urgent else 'Standard order.'}"

    solo = {}
    for case in cases[:6]:
        solo[case] = shared.generate(context=context_for(case), questions=questions).usage

    def run(case):
        done = shared.generate(context=context_for(case), questions=questions)
        return case, done.result, shared._last_prompts.get(), done.usage

    wrong, leaked, usage_mismatch = [], [], []
    start = time.perf_counter()
    with ThreadPoolExecutor(16) as pool:
        for case, result, prompts, usage in pool.map(run, cases):
            expected = {"invoice": case[0], "quantity": case[1], "urgent": case[2]}
            if result != expected:
                wrong.append({"case": case, "result": result})
            if not prompts or any(f"Invoice #{case[0]}." not in p for p in prompts):
                leaked.append(case)
            if case in solo and (usage.requests, usage.prompt_tokens) != (solo[case].requests, solo[case].prompt_tokens):
                usage_mismatch.append({"case": case, "solo": vars(solo[case]), "concurrent": vars(usage)})
    out["shared_client"] = {"calls": len(cases), "threads": 16, "elapsed": time.perf_counter() - start,
                            "wrong_answers": wrong, "leaked_prompts": leaked, "usage_mismatch": usage_mismatch}

    # 2. A seeded sampled call reproduces. Run serially: SGLang 0.5.19 can crash
    # when a decode batch mixes requests with and without token_ids_logprob
    # (see repro_mixed_logprob.py), and this checks the seed, not concurrency.
    sampled = client(mode="sample", temperature=1.0)
    seed_q = {"roll": {"type": "string", "enum": ["one", "two", "three", "four", "five", "six"],
                       "permutations": 3, "return_probabilities": True},
              "guess": {"type": "number", "instructions": "Guess the average of many rolls."}}
    text_q = {"name": {"type": "string", "instructions": "Invent a name for this die."}}

    def roll(seed):
        return seed, sampled.generate(context="A single roll of a fair die.", questions=seed_q, seed=seed).result

    rolls = [roll(seed) for seed in [7] * 4 + [8] * 4]
    by_seed = {}
    for seed, result in rolls:
        by_seed.setdefault(seed, []).append(json.dumps(result, sort_keys=True))
    out["seed"] = {str(seed): {"distinct_results": len(set(results)), "example": json.loads(results[0])}
                   for seed, results in by_seed.items()}
    names = [sampled.generate(context="A single roll of a fair die.", questions=text_q, seed=seed).result["name"]
             for seed in (7, 7, 8, 8)]
    out["seed"]["text_sequential"] = {"seed7": names[:2], "seed8": names[2:]}

    # 3. Usage: a repeat of the same call should reuse the cached prefix.
    fresh = f"Nonce {time.time_ns()}. " + RECEIPT
    first = vars(shared.generate(context=fresh, questions=RECEIPT_Q).usage)
    out["usage"] = {"first": first, "repeat": vars(shared.generate(context=fresh, questions=RECEIPT_Q).usage)}

    # 4. timeout: an unbounded thinking call stops near its budget, and the server keeps serving.
    thinker = client()
    hard = {"proof": {"type": "boolean", "thinking": True, "instructions":
                      "Think very carefully, checking every case: is 2^89 - 1 prime? Verify with long division."}}
    start = time.perf_counter()
    try:
        timeout_usage = thinker.generate(context="Number theory.", questions=hard, timeout=3).usage
        timeout_error = None
    except Exception as exc:
        timeout_error, timeout_usage = type(exc).__name__, getattr(exc, "usage", None)
    timeout_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    after = shared.generate(context=context_for(cases[0]), questions=questions).result
    out["timeout"] = {"budget": 3, "error": timeout_error, "elapsed": timeout_elapsed,
                      "usage": vars(timeout_usage) if timeout_usage else None,
                      "next_call_ok": after == {"invoice": 1000, "quantity": 1, "urgent": True},
                      "next_call_elapsed": time.perf_counter() - start}

    # 5. cancel: a dependent thinking call stops after its in-flight request.
    cancel = threading.Event()
    budgeted = client()
    budgeted.sglang.thinking_budget = 256  # for every field that thinks
    timer = threading.Timer(1.0, cancel.set)
    timer.start()
    start = time.perf_counter()
    try:
        cancel_usage = budgeted.generate(context="The payments service is returning errors after a deployment.",
                                         questions=thinks(INCIDENT_Q), cancel=cancel).usage
        cancel_error = None
    except Exception as exc:
        cancel_error, cancel_usage = type(exc).__name__, getattr(exc, "usage", None)
    timer.cancel()
    uncancelled = time.perf_counter()
    full = budgeted.generate(context="The payments service is returning errors after a deployment.",
                             questions=thinks(INCIDENT_Q)).usage
    out["cancel"] = {"cancel_at": 1.0, "error": cancel_error, "elapsed": uncancelled - start,
                     "requests_before_stop": cancel_usage.requests if cancel_usage else None,
                     "full_call_elapsed": time.perf_counter() - uncancelled,
                     "full_call_requests": full.requests}
    return out


def wait_healthy(limit=600):
    import urllib.request
    start = time.time()
    while time.time() - start < limit:
        try:
            with urllib.request.urlopen(args.url + "/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def load():
    shared = client()
    shared.sglang.warmup()
    questions = {**{k: v for k, v in RECEIPT_Q.items() if k != "merchant"}}
    rows = []
    for threads in (1, 8, 16, 32):
        calls = max(16, threads * 4)

        def run(n):
            start = time.perf_counter()
            try:
                shared.generate(context=f"Ticket {n}. " + RECEIPT, questions=questions, timeout=120).result
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"[:200]
            return time.perf_counter() - start, None

        start = time.perf_counter()
        with ThreadPoolExecutor(threads) as pool:
            results = list(pool.map(run, range(calls)))
        wall = time.perf_counter() - start
        latencies = sorted(t for t, _ in results if t is not None)
        errors = [e for _, e in results if e is not None]
        row = {"threads": threads, "calls": calls, "ok": len(latencies), "errors": len(errors),
               "first_error": errors[0] if errors else None}
        if latencies:
            row.update(calls_per_s=len(latencies) / wall, p50=latencies[len(latencies) // 2],
                       p95=latencies[max(0, int(len(latencies) * 0.95) - 1)])
        rows.append(row)
        print(json.dumps(row), flush=True)
        if errors:
            wait_healthy()
    return {"rows": rows}


report = {"phase": args.phase, "root": args.root, "typellm": typellm.__file__}
report.update({"parity": parity, "perf": perf, "features": features, "load": load}[args.phase]())
Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=vars))
print(json.dumps(report, ensure_ascii=False, default=vars)[:4000])
