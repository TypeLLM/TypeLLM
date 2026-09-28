import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typellm import TypeLLMClient


SEED = 20260918
EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "numeric_eval_cases.jsonl"
RESULTS_PATH = EVAL_DIR / "numeric_eval_formal_results.jsonl"


def make_cases():
    rng = random.Random(SEED)
    cases = []

    for index in range(24):
        value = rng.randint(10, 9999)
        cases.append(
            {
                "id": f"integer_extraction_{index:03d}",
                "category": "integer_extraction",
                "context": f"Invoice INV-{1000 + index}. The final item count is {value} units.",
                "question": "What is the final item count?",
                "type": "integer",
                "expected": value,
            }
        )

    operations = ["plus", "minus", "multiplied by"]
    for index in range(24):
        operation = operations[index % len(operations)]
        if operation == "plus":
            left, right = rng.randint(10, 900), rng.randint(10, 900)
            expected = left + right
        elif operation == "minus":
            left, right = rng.randint(10, 500), rng.randint(501, 950)
            expected = left - right
        else:
            left, right = rng.randint(2, 99), rng.randint(2, 99)
            expected = left * right
        cases.append(
            {
                "id": f"integer_arithmetic_{index:03d}",
                "category": "integer_arithmetic",
                "context": "Calculate the requested value accurately.",
                "question": f"What is {left} {operation} {right}?",
                "type": "integer",
                "expected": expected,
            }
        )

    for index in range(24):
        whole = rng.randint(-200, 500)
        fraction = rng.choice([1, 2, 4, 5, 8, 25, 50, 75])
        digits = 2 if fraction >= 10 else 1
        value = float(f"{whole}.{fraction:0{digits}d}")
        cases.append(
            {
                "id": f"number_extraction_{index:03d}",
                "category": "number_extraction",
                "context": f"Sensor {index} recorded a calibrated reading of {value} volts.",
                "question": "What calibrated voltage was recorded?",
                "type": "number",
                "expected": value,
            }
        )

    for index in range(24):
        left = rng.randint(-100, 300) / 10
        right = rng.randint(1, 90) / 10
        if index % 2:
            operation = "plus"
            expected = round(left + right, 6)
        else:
            operation = "multiplied by"
            right = rng.choice([0.5, 1.5, 2.0, 2.5, 4.0])
            expected = round(left * right, 6)
        cases.append(
            {
                "id": f"number_arithmetic_{index:03d}",
                "category": "number_arithmetic",
                "context": "Calculate the requested value accurately.",
                "question": f"What is {left} {operation} {right}?",
                "type": "number",
                "expected": expected,
            }
        )

    for index in range(16):
        value = -rng.randint(1, 9999)
        cases.append(
            {
                "id": f"negative_integer_{index:03d}",
                "category": "negative_integer",
                "context": f"The final signed account adjustment is {value} cents.",
                "question": "What is the signed account adjustment in cents?",
                "type": "integer",
                "expected": value,
            }
        )

    for index in range(16):
        left = rng.randint(-80, 80)
        right = rng.randint(-80, 80)
        expected = left + right
        cases.append(
            {
                "id": f"sequential_dependency_{index:03d}",
                "category": "sequential_dependency",
                "context": "Calculate the value, then use the recorded result for the next decision.",
                "question": f"What is {left} plus {right}?",
                "type": "integer",
                "expected": expected,
                "followup": {
                    "question": "Is the previously recorded answer greater than zero?",
                    "expected": expected > 0,
                },
            }
        )
    return cases


def schema_for(case):
    properties = {
        "answer": {
            "type": case["type"],
            "instructions": case["question"],
        }
    }
    if "followup" in case:
        properties["is_positive"] = {
            "type": "boolean",
            "instructions": case["followup"]["question"],
            "depends_on": ["answer"],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


def matches(actual, expected):
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def main():
    cases = make_cases()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    client = TypeLLMClient(
        "http://127.0.0.1:30000",
        model="qwen3.8-27b",
        mode="argmax",
    )
    rows = []
    for position, case in enumerate(cases, 1):
        started = time.perf_counter()
        error = None
        result = None
        try:
            result = client.generate(context=case["context"], schema=schema_for(case)).result
            correct = matches(result["answer"], case["expected"])
            if "followup" in case:
                correct = correct and result["is_positive"] == case["followup"]["expected"]
        except Exception as exc:
            correct = False
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "id": case["id"],
            "category": case["category"],
            "expected": case["expected"],
            "result": result,
            "correct": correct,
            "error": error,
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(json.dumps({"progress": f"{position}/{len(cases)}", **row}), flush=True)

    RESULTS_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    summary = {
        category: {
            "total": len(category_rows),
            "correct": sum(row["correct"] for row in category_rows),
            "accuracy": sum(row["correct"] for row in category_rows) / len(category_rows),
            "mean_seconds": sum(row["seconds"] for row in category_rows) / len(category_rows),
        }
        for category, category_rows in grouped.items()
    }
    summary["overall"] = {
        "total": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_seconds": sum(row["seconds"] for row in rows) / len(rows),
        "errors": sum(row["error"] is not None for row in rows),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
