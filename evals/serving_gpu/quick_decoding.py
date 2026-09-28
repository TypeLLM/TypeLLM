"""The 128 numeric eval cases: correct answers, requests, latency; then 16-way throughput."""
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT), str(ROOT / "evals")]
from numeric_eval import matches, schema_for  # noqa: E402
from typellm import TypeLLMClient  # noqa: E402

cases = [json.loads(l) for l in (ROOT / "evals/numeric_eval_cases.jsonl").read_text().splitlines()]


def client():
    return TypeLLMClient("http://127.0.0.1:30000", model="qwen3.8-27b",
                         tokenizer="RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead")


def one(c, case):
    start = time.perf_counter()
    done = c.generate(context=case["context"], schema=schema_for(case))
    return done.result, time.perf_counter() - start, done.usage.requests


report = {}
for decoding in ("grammar",):
    c = client()
    one(c, cases[0])  # warm tokenizer and tables
    rows = []
    for case in cases:
        try:
            result, elapsed, requests = one(c, case)
            error = None
        except Exception as exc:
            result, elapsed, requests, error = None, None, None, f"{type(exc).__name__}: {exc}"[:200]
        rows.append({"id": case["id"], "result": result, "elapsed": elapsed, "requests": requests,
                     "error": error, "correct": error is None and matches(result["answer"], case["expected"])})
    ok = [r for r in rows if not r["error"]]
    summary = {"correct": sum(r["correct"] for r in rows), "errors": len(rows) - len(ok),
               "requests": sum(r["requests"] for r in ok),
               "median_ms": round(1000 * statistics.median(r["elapsed"] for r in ok), 1),
               "total_s": round(sum(r["elapsed"] for r in ok), 2)}

    def work(case):
        return one(c, case)[1]

    start = time.perf_counter()
    with ThreadPoolExecutor(16) as pool:
        latencies = sorted(pool.map(work, cases))
    wall = time.perf_counter() - start
    summary["load16"] = {"calls_per_s": round(len(cases) / wall, 2),
                         "p50_ms": round(1000 * latencies[len(latencies) // 2]),
                         "p95_ms": round(1000 * latencies[int(len(latencies) * 0.95) - 1])}
    print(decoding, json.dumps(summary), flush=True)
    report[decoding] = {"summary": summary, "rows": rows}

Path("quick_decoding.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
