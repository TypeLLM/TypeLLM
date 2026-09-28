"""Live image-input smoke test: the answers exist only in the image, not the text."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont
from typellm import TypeLLMClient

QUESTIONS = {
    'item': {'type': 'string', 'enum': ['notebooks', 'pencils', 'coffee'], 'instructions': 'Which item is on the receipt?'},
    'quantity': {'type': 'integer', 'instructions': 'How many were bought?'},
    'total': {'type': 'number', 'instructions': 'What is the total in dollars?'},
    'paid': {'type': 'boolean', 'instructions': 'Is the receipt stamped PAID?'},
    'check': {'type': 'boolean', 'instructions': 'Is quantity greater than 2 and the receipt paid?',
              'depends_on': ['quantity', 'paid']},
}
EXPECTED = {'item': 'notebooks', 'quantity': 3, 'total': 12.5, 'paid': True, 'check': True}


def receipt() -> Image.Image:
    image = Image.new('RGB', (640, 360), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 36)
    except OSError:
        font = ImageFont.load_default(size=36)
    for y, line in [(40, 'RECEIPT'), (120, '3 x notebooks'), (190, 'TOTAL  $12.50')]:
        draw.text((40, y), line, fill='black', font=font)
    draw.rectangle((400, 250, 600, 330), outline='red', width=6)
    draw.text((430, 268), 'PAID', fill='red', font=font)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--thinking', action='store_true')
    args = parser.parse_args()
    image = receipt()
    rows, passed = [], True
    for fields, questions in [
        ('independent', {k: v for k, v in QUESTIONS.items() if k != 'check'}),
        ('with depends_on', QUESTIONS),
    ]:
        client = TypeLLMClient(args.url, model=args.model)
        values = client.generate(context='Read the attached receipt.', images=[image],
                                 questions={k: {**v, 'thinking': args.thinking} for k, v in questions.items()}).result
        ok = all(values[k] == EXPECTED[k] for k in values)
        passed &= ok
        rows.append({'fields': fields, 'values': values, 'passed': ok})
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({'passed': passed}))
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
