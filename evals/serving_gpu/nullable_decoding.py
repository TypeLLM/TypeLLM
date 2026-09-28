"""Nullable fields on a live SGLang server.

Values that start with a merged token (' "$', ' "/', ' "(' ...), missing values,
negative and zero numbers, with nullable booleans and enums as a control. Each
context runs as one batch call and as one call per field.

  python nullable_decoding.py TYPELLM_ROOT [URL [MODEL [TOKENIZER]]]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from typellm import TypeLLMClient  # noqa: E402

URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:30000"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b"
TOKENIZER = sys.argv[4] if len(sys.argv) > 4 else "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"
S, N, I, B = ["string", "null"], ["number", "null"], ["integer", "null"], ["boolean", "null"]

CASES = [
    ("Invoice INV-77 from Northwind Traders. Amount due: $12.50. Tip: none was added. "
     "Pay by card. Customer phone: (555) 123-4567. No coupon code was used.", {
        "amount_text": ({"type": S, "instructions": "The amount due exactly as written, with its currency sign."}, "$12.50"),
        "phone": ({"type": S, "instructions": "The customer phone number exactly as written."}, "(555) 123-4567"),
        "coupon": ({"type": S, "instructions": "The coupon code, if any."}, None),
        "tip": ({"type": N, "instructions": "The tip amount in dollars, if any."}, None),
        "merchant": ({"type": S, "instructions": "The merchant name."}, "Northwind Traders"),
    }),
    ("Install log: the binary was placed at /usr/local/bin/aurora-cli by user ops. "
     "Ticket #A-42 tracks it. No config file was written. Exit code 0.", {
        "binary_path": ({"type": S, "instructions": "The full path of the installed binary."}, "/usr/local/bin/aurora-cli"),
        "ticket": ({"type": S, "instructions": "The ticket reference exactly as written."}, "#A-42"),
        "config_path": ({"type": S, "instructions": "The path of the config file written, if any."}, None),
        "exit_code": ({"type": I, "instructions": "The exit code."}, 0),
        "retries": ({"type": I, "instructions": "How many retries were made, if stated."}, None),
    }),
    ("Account statement for Alice Chen: closing balance -42.75 dollars, overdraft fee 0, "
     "interest rate not stated. Table 7 at the branch. Email not on file.", {
        "balance": ({"type": N, "instructions": "The closing balance in dollars."}, -42.75),
        "overdraft_fee": ({"type": N, "instructions": "The overdraft fee in dollars."}, 0.0),
        "interest_rate": ({"type": N, "instructions": "The interest rate, if stated."}, None),
        "email": ({"type": S, "instructions": "The customer's email, if on file."}, None),
        "name": ({"type": S, "instructions": "The account holder's name."}, "Alice Chen"),
    }),
    ("Delivery note: 3 boxes, weight 12.5 kg, signed by the recipient. Delivery window not given. "
     "Temperature during transport: -3 degrees.", {
        "boxes": ({"type": I, "instructions": "How many boxes?"}, 3),
        "weight": ({"type": N, "instructions": "The weight in kg."}, 12.5),
        "window": ({"type": S, "instructions": "The delivery window, if given."}, None),
        "temperature": ({"type": I, "instructions": "The transport temperature in degrees."}, -3),
        "signed": ({"type": B, "instructions": "Was it signed by the recipient?"}, True),
        "carrier": ({"type": S, "enum": ["DHL", "UPS", None], "instructions": "The carrier, if named."}, None),
    }),
]


def run():
    c = TypeLLMClient(URL, model=MODEL, tokenizer=TOKENIZER)
    rows = []
    for context, fields in CASES:
        questions = {name: spec for name, (spec, _) in fields.items()}
        start = time.perf_counter()
        done = c.generate(context=context, questions=questions)
        batch, batch_s, batch_requests = done.result, time.perf_counter() - start, done.usage.requests
        for name, (spec, expected) in fields.items():
            single = c.generate(context=context, questions={name: spec}).result[name]
            rows.append({"field": name, "expected": expected, "batch": batch[name], "single": single,
                         "batch_ok": batch[name] == expected, "single_ok": single == expected})
        rows[-1]["batch_s"], rows[-1]["batch_requests"] = batch_s, batch_requests
    return rows


report = {"grammar": run()}
for d, rows in report.items():
    print(d, "batch", sum(r["batch_ok"] for r in rows), "/", len(rows),
          "single", sum(r["single_ok"] for r in rows), "/", len(rows),
          "requests", sum(r.get("batch_requests", 0) for r in rows))
print(f"{'field':15} {'expected':28} batch / single")
for r in report["grammar"]:
    mark = "" if r["batch_ok"] and r["single_ok"] else "  <--"
    print(f"{r['field']:15} {json.dumps(r['expected']):28} {json.dumps(r['batch'])} / {json.dumps(r['single'])}{mark}")
Path("nullable_decoding.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
