"""Reproducible live SGLang smoke matrix; no model weights are loaded here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typellm import TypeLLMClient


def cases():
    return [
        ('enum', 'The expense was a train ticket.', {'answer': {'type': 'string', 'enum': ['meal', 'travel', 'equipment'], 'instructions': 'Classify the expense.', 'return_probabilities': True}}, {'answer': 'travel'}),
        ('boolean', 'The light is on.', {'answer': {'type': 'boolean', 'instructions': 'Is the light on?', 'return_probabilities': True}}, {'answer': True}),
        ('numeric_enum', 'The rating is 0.5.', {'answer': {'type': 'number', 'enum': [0.0, 0.5, 1.0], 'instructions': 'Extract the rating.', 'return_probabilities': True}}, {'answer': 0.5}),
        ('integer', 'There are 42 apples.', {'answer': {'type': 'integer', 'instructions': 'How many apples?'}}, {'answer': 42}),
        ('negative', 'The temperature is -7 degrees.', {'answer': {'type': 'integer', 'instructions': 'Extract the temperature.'}}, {'answer': -7}),
        ('number', 'The price is 12.5 dollars.', {'answer': {'type': 'number', 'instructions': 'Extract the price.'}}, {'answer': 12.5}),
        ('text', 'The access code is blue.', {'answer': {'type': 'string', 'instructions': 'Return only the access code.'}}, {'answer': 'blue'}),
        ('bounded_text', 'The access code is blue.', {'answer': {'type': 'string', 'instructions': 'Return only the access code.'}}, {'answer': 'blue'}),
        ('mixed', 'There are 3 red apples. They are fresh.', {
            'color': {'type': 'string', 'enum': ['red', 'blue'], 'instructions': 'What color are the apples?', 'return_probabilities': True},
            'count': {'type': 'integer', 'instructions': 'How many apples?'},
            'fresh': {'type': 'boolean', 'instructions': 'Are the apples fresh?'},
            'name': {'type': 'string', 'instructions': 'Name the fruit in one lowercase plural word.'},
        }, {'color': 'red', 'count': 3, 'fresh': True, 'name': 'apples'}),
    ]


def validate(result, questions):
    if set(result) != set(questions):
        return False
    for key, field in questions.items():
        value = result[key]
        if field.get('return_probabilities'):
            if not isinstance(value, dict) or set(value) != {'value', 'probabilities'}:
                return False
            probs = value['probabilities']
            candidates = field.get('enum', [True, False])
            if set(probs) != set(candidates) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values()) or not math.isclose(sum(probs.values()), 1, abs_tol=1e-6):
                return False
            value = value['value']
        kind = field['type']
        valid = {'string': type(value) is str, 'integer': type(value) is int,
                 'number': type(value) in (int, float) and math.isfinite(value) if type(value) in (int, float) else False,
                 'boolean': type(value) is bool}[kind]
        if not valid or ('enum' in field and value not in field['enum']):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--output', required=True)
    parser.add_argument('--thinking-budget', type=int, default=None)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    rows = []
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Warm the shared read-only tokenizer/table once; avoid concurrent cache writes.
    warm = TypeLLMClient(args.url, model=args.model, tokenizer=args.model)
    tokenizer = warm.sglang._get_chat_tokenizer()
    numeric = warm.sglang.numeric_token_pieces()

    def run_one(case, thinking):
        name, context, questions, expected = case
        client = TypeLLMClient(args.url, model=args.model, tokenizer=args.model,
            text_max_tokens=128, timeout=180, seed=42)
        client.sglang.thinking_budget = args.thinking_budget  # for every field that thinks
        client.sglang._chat_tokenizer = tokenizer
        client.sglang._numeric_tokens = numeric
        row = dict(model=args.model, thinking=thinking, thinking_budget=args.thinking_budget,
                   case=name, workers=args.workers)
        reasoning = []
        request = client.sglang._request
        def record_request(path, payload=None, **kwargs):
            response = request(path, payload, **kwargs)
            params = payload.get('sampling_params', {}) if payload else {}
            # A batch of prompts has one sampling-params dict and one response each.
            for item_params, item in zip(params, response) if isinstance(params, list) and isinstance(response, list) else [(params, response)]:
                if isinstance(item_params, dict) and item_params.get('stop') == ['</think>']:
                    meta = item.get('meta_info', {})
                    reasoning.append({k: meta.get(k) for k in ('completion_tokens', 'finish_reason')})
            return response
        client.sglang._request = record_request
        start = time.monotonic()
        try:
            result = client.generate(context=context, questions={k: {**v, 'thinking': True} for k, v in questions.items()} if thinking else questions).result
            values = {k: v['value'] if questions[k].get('return_probabilities') else v for k, v in result.items()}
            row.update(result=result, type_valid=validate(result, questions), correct=values == expected)
        except Exception as exc:
            row.update(error=f'{type(exc).__name__}: {exc}', type_valid=False, correct=False)
        row['seconds'] = round(time.monotonic() - start, 3)
        row['reasoning'] = reasoning
        return row

    with path.open('w') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for thinking in (False, True):
            futures = [pool.submit(run_one, case, thinking) for case in cases()]
            for future in futures:
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {'model': args.model, 'requests': len(rows), 'completed': sum('error' not in r for r in rows),
               'type_valid': sum(r['type_valid'] for r in rows), 'correct': sum(r['correct'] for r in rows)}
    path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
