"""Read a real photographed receipt into typed strings, integers and numbers."""
import argparse
import json
from pathlib import Path
import time

from typellm import TypeLLMClient

HERE = Path(__file__).resolve().parent
RUPIAH = 'in rupiah as a plain number; the receipt uses commas as thousands separators, so 12,500 is 12500'

QUESTIONS = {
    'first_item': {'type': 'string', 'instructions': 'Name of the first line item, exactly as printed, including any misspellings.'},
    'second_item': {'type': 'string', 'instructions': 'Name of the second line item, exactly as printed.'},
    'third_item': {'type': 'string', 'instructions': 'Name of the third line item, exactly as printed.'},
    'item_lines': {'type': 'integer', 'instructions': 'How many line items are listed?'},
    'first_item_quantity': {'type': 'integer', 'instructions': 'Quantity of the first line item.'},
    'subtotal': {'type': 'number', 'instructions': f'Subtotal {RUPIAH}.'},
    'discount': {'type': 'number', 'instructions': f'Discount {RUPIAH}.'},
    'total': {'type': 'number', 'instructions': f'Total {RUPIAH}.'},
    'cash': {'type': 'number', 'instructions': f'Cash tendered {RUPIAH}.'},
    'change': {'type': 'number', 'instructions': f'Change given {RUPIAH}.'},
    'paid_in_cash': {'type': 'boolean', 'instructions': 'Was the bill paid in cash?'},
    # Dependent fields compute from the values read above.
    'discount_percent': {'type': 'number', 'depends_on': ['subtotal', 'discount'],
                         'instructions': 'Discount as a percentage of the subtotal.'},
    'change_is_correct': {'type': 'boolean', 'depends_on': ['total', 'cash', 'change'],
                          'instructions': 'Does change equal cash minus total?'},
    'expense_note': {'type': 'string', 'depends_on': ['first_item', 'second_item', 'third_item', 'total'],
                     'instructions': 'One short sentence for an expense report describing this purchase and its total.'},
}

# Labels from the CORD-v2 test split, receipt 4.
EXPECTED = {
    'first_item': 'ICE BLACKCOFFE', 'second_item': 'AVOCADO COFFEE', 'third_item': 'CHIKEN KATSU FF',
    'item_lines': 3, 'first_item_quantity': 2,
    'subtotal': 194000, 'discount': 19400, 'total': 174600, 'cash': 200000, 'change': 25400,
    'paid_in_cash': True, 'discount_percent': 10, 'change_is_correct': True,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--model', default='qwen3.8-27b')
    parser.add_argument('--tokenizer', default='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE / 'result.json')
    args = parser.parse_args()
    client = TypeLLMClient(args.url, model=args.model, tokenizer=args.tokenizer,
                           timeout=300)
    start = time.monotonic()
    result = client.generate(context='Read the attached photo of a restaurant receipt.',
                             images=[HERE / 'receipt.jpg'],
                             questions={k: {**v, 'thinking': args.thinking} for k, v in QUESTIONS.items()}).result
    seconds = time.monotonic() - start
    checks = {name: result[name] == value for name, value in EXPECTED.items()}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'{sum(checks.values())}/{len(checks)} fields match the labels in {seconds:.1f} s')
    args.output.write_text(json.dumps({
        'model': args.model, 'thinking': args.thinking, 'seconds': seconds,
        'result': result, 'expected': EXPECTED, 'matches': checks,
    }, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
