"""Read a photographed US restaurant receipt into typed fields."""
import argparse
import json
from pathlib import Path
import time

from typellm import TypeLLMClient

HERE = Path(__file__).resolve().parent
DOLLARS = 'in US dollars as a plain number, e.g. 12.50'

QUESTIONS = {
    'store': {'type': 'string', 'instructions': 'Name of the restaurant.'},
    'date': {'type': 'string', 'instructions': 'Date printed on the receipt, exactly as printed.'},
    'first_item': {'type': 'string', 'instructions': 'Name of the first line item, without its quantity.'},
    'second_item': {'type': 'string', 'instructions': 'Name of the second line item, without its quantity.'},
    'third_item': {'type': 'string', 'instructions': 'Name of the third line item, without its quantity.'},
    'table': {'type': 'string', 'instructions': 'Table number.'},
    'item_lines': {'type': 'integer', 'instructions': 'How many line items are listed?'},
    'guests': {'type': 'integer', 'instructions': 'Number of customers on the check.'},
    'subtotal': {'type': 'number', 'instructions': f'Subtotal {DOLLARS}.'},
    'tax': {'type': 'number', 'instructions': f'Sales tax {DOLLARS}.'},
    'total': {'type': 'number', 'instructions': f'Total due {DOLLARS}.'},
    'currency': {'type': 'string', 'enum': ['USD', 'EUR', 'JPY', 'other'], 'instructions': 'Currency of the amounts.'},
    'tips_accepted': {'type': 'boolean', 'instructions': 'Does the restaurant accept tips?'},
    # Dependent fields compute from the values read above.
    'tax_rate_percent': {'type': 'number', 'depends_on': ['subtotal', 'tax'],
                         'instructions': 'Sales tax as a percentage of the subtotal, rounded to two decimals.'},
    'per_person': {'type': 'number', 'depends_on': ['total', 'guests'],
                   'instructions': 'Total due per customer in US dollars, rounded to two decimals.'},
    'expense_note': {'type': 'string',
                     'depends_on': ['store', 'date', 'guests', 'total', 'tips_accepted'],
                     'instructions': 'One short sentence for an expense report describing this meal.'},
}

# WildReceipt labels, plus four fields read by eye (table, guests, currency, tips_accepted).
EXPECTED = {
    'store': 'Sushi Yasuda', 'date': '06/05/13',
    'first_item': 'Edamame', 'second_item': 'Kimo', 'third_item': 'A la Carte Sushi',
    'table': '7A', 'item_lines': 3, 'guests': 2,
    'subtotal': 262.5, 'tax': 23.3, 'total': 285.8, 'currency': 'USD', 'tips_accepted': False,
    'tax_rate_percent': 8.88, 'per_person': 142.9,
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
    print(f'{sum(checks.values())}/{len(checks)} fields match in {seconds:.1f} s')
    args.output.write_text(json.dumps({
        'model': args.model, 'thinking': args.thinking, 'seconds': seconds,
        'result': result, 'expected': EXPECTED, 'matches': checks,
    }, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
