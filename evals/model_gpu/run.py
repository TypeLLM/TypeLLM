"""Live multi-protocol matrix with exact prompts and transport metadata."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from typellm import TypeLLMClient

spec = importlib.util.spec_from_file_location('small_eval', ROOT / 'evals/qwen35_small/run.py')
small = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small)


def dag_cases():
    return [
        ('dag_diamond', 'Return the requested values, using dependency results.', {
            'final': {'type': 'integer', 'enum': [17, 21], 'depends_on': ['left', 'right'], 'instructions': 'Add left and right.'},
            'root': {'type': 'integer', 'enum': [7, 11], 'instructions': 'Return 7.'},
            'left': {'type': 'integer', 'enum': [8, 12], 'depends_on': ['root'], 'instructions': 'Return root plus 1.'},
            'noise': {'type': 'string', 'enum': ['UNRELATED_SENTINEL'], 'instructions': 'Choose the only candidate.'},
            'right': {'type': 'integer', 'enum': [9, 13], 'depends_on': ['root'], 'instructions': 'Return root plus 2.'},
        }, {'final': 17, 'root': 7, 'left': 8, 'noise': 'UNRELATED_SENTINEL', 'right': 9}),
        ('dag_mixed', 'Follow the instructions.', {
            'number': {'type': 'integer', 'instructions': 'Return 7.', 'depends_on': []},
            'text': {'type': 'string', 'instructions': 'Write number as one lowercase English word.', 'depends_on': ['number']},
            'check': {'type': 'boolean', 'instructions': 'Is text exactly seven?', 'depends_on': ['text']},
        }, {'number': 7, 'text': 'seven', 'check': True}),
        ('dag_enum24', 'Follow the instructions.', {
            'pick': {'type': 'integer', 'enum': list(range(24)), 'instructions': 'Return 23.', 'depends_on': []},
            'check': {'type': 'boolean', 'instructions': 'Is pick equal to 23?', 'depends_on': ['pick']},
        }, {'pick': 23, 'check': True}),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:30001')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thinking-budget', type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    warm = TypeLLMClient(args.url, model=args.model, tokenizer=args.model)
    tokenizer = warm.sglang._get_chat_tokenizer()
    numeric = warm.sglang.numeric_token_pieces()
    manifest = {'model': args.model, 'model_info': warm.sglang._model_info(),
                'thinking_budget': args.thinking_budget,
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'typellm').glob('*.py')}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    with (args.output/'results.jsonl').open('w') as out:
        for thinking in ([False] if 'Ling-mini' in args.model else [False, True]):
            for name, context, questions, expected in [*small.cases(), *dag_cases()]:
                client = TypeLLMClient(args.url, model=args.model, tokenizer=args.model,
                    text_max_tokens=128, timeout=180, seed=42)
                client.sglang.thinking_budget = args.thinking_budget  # for every field that thinks
                client.sglang._chat_tokenizer = tokenizer
                client.sglang._numeric_tokens = numeric
                original = client.sglang._request
                requests = []
                def record(path, payload=None, **kwargs):
                    response = original(path, payload, **kwargs)
                    if path == '/generate':
                        rr = response if isinstance(response, list) else [response]
                        requests.append({'sampling_params': payload.get('sampling_params'),
                                         'meta': [r.get('meta_info', {}) for r in rr]})
                    return response
                client.sglang._request = record
                row = {'model': args.model, 'case': name, 'thinking': thinking}
                start = time.monotonic()
                try:
                    result = client.generate(context=context, questions={k: {**v, 'thinking': True} for k, v in questions.items()} if thinking else questions).result
                    values = {k: v['value'] if questions[k].get('return_probabilities') else v for k,v in result.items()}
                    prompts = dict(zip(questions, client._last_prompts.get())) if name.startswith('dag_') else client._last_prompts.get()
                    row.update(result=result, type_valid=small.validate(result, questions), correct=values == expected, prompts=prompts)
                    if name == 'dag_diamond':
                        row['isolation'] = all('UNRELATED_SENTINEL' not in prompts[k] for k in ('root','left','right','final'))
                    if name.startswith('dag_'):
                        row['prefix_checks'] = []
                        for field, schema in questions.items():
                            parents = schema.get('depends_on', [])
                            if parents:
                                parent = max(parents, key=lambda p:len(prompts[p]))
                                row['prefix_checks'].append({'parent': parent, 'child': field, 'preserved': prompts[field].startswith(prompts[parent])})
                except Exception as exc:
                    row.update(error=f'{type(exc).__name__}: {exc}', type_valid=False, correct=False)
                row.update(seconds=round(time.monotonic()-start,3), requests=requests)
                rows.append(row)
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps({k:v for k,v in row.items() if k not in ('prompts','requests','result')},ensure_ascii=False),flush=True)
    summary = {'model': args.model, 'total':len(rows), 'completed':sum('error' not in r for r in rows),
               'type_valid':sum(r['type_valid'] for r in rows), 'correct':sum(r['correct'] for r in rows)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return 0 if summary['correct']==summary['total'] else 1


if __name__=='__main__':
    raise SystemExit(main())
