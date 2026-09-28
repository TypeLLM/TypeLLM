"""Runnable example and CLI for the TypeLLM prototype."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from .benchmark import benchmark_prefix_cache
from .runtime import TypeLLMClient


def example_schema() -> dict[str, Any]:
    return {
        "type": "object",
        # Results come back in this order.
        "properties": {
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
                "description": "Choose the confidence level.",
            },
        },
        "required": ["expense_type", "reimbursable", "confidence"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="typellm",
        description="Type-safe decisions over SGLang /generate"
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
    )
    parser.add_argument("--model", default=os.environ.get("SGLANG_MODEL"))
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("TYPELLM_TOKENIZER"),
        help="Tokenizer path or Hugging Face ID; normally discovered from SGLang",
    )
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 picks the most likely answer; above 0 samples")
    parser.add_argument("--mode", choices=("argmax", "sample"), help="deprecated; use --temperature")
    parser.add_argument("--numeric-max-digits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probabilities", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--context-repeat", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    receipt = (
        "Expense report: An employee bought a train ticket from London to "
        "Cambridge for a required client meeting. The receipt is valid and "
        "the trip was approved by the employee's manager."
    )
    schema = example_schema()
    if args.probabilities:
        for field in schema["properties"].values():
            field["return_probabilities"] = True
    client = TypeLLMClient(
        args.server_url,
        args.model,
        mode=args.mode,
        temperature=args.temperature,
        seed=args.seed,
        numeric_max_digits=args.numeric_max_digits,
        tokenizer=args.tokenizer,
    )
    results = client.generate(
        context=receipt,
        questions=schema["properties"],
        print_final_prompt=True,
    ).result
    print("\n===== DECISIONS =====")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.benchmark:
        long_context = ((receipt + "\n") * args.context_repeat).rstrip()
        decisions = client.compile_schema(schema)
        benchmark = benchmark_prefix_cache(
            long_context, decisions, base_url=args.server_url, model=args.model
        )
        print("\n===== PREFIX-CACHE BENCHMARK =====")
        print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
