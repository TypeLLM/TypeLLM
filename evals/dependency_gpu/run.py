"""Live dependency/prefix smoke test. Does not flush a shared server's cache."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from typellm import TypeLLMClient


def cases():
    return [
        ('diamond', {
            'final': {'type': 'integer', 'depends_on': ['left', 'right'], 'instructions': 'Add the values of left and right.'},
            'root': {'type': 'integer', 'enum': list(range(24)), 'instructions': 'Return 7.', 'return_probabilities': True},
            'left': {'type': 'integer', 'enum': list(range(24)), 'depends_on': ['root'], 'instructions': 'Return root plus 1.'},
            'noise': {'type': 'string', 'enum': ['ISOLATED_BRANCH_SENTINEL'], 'instructions': 'Select the only candidate.'},
            'right': {'type': 'integer', 'enum': list(range(24)), 'depends_on': ['root'], 'instructions': 'Return root plus 2.'},
        }, {'final': 17, 'root': 7, 'left': 8, 'noise': 'ISOLATED_BRANCH_SENTINEL', 'right': 9}),
        ('mixed_chain', {
            'number': {'type': 'integer', 'instructions': 'Return 7.', 'depends_on': []},
            'text': {'type': 'string', 'instructions': 'Write the value of number as an English lowercase word, with no punctuation.', 'depends_on': ['number']},
            'check': {'type': 'boolean', 'instructions': 'Is text exactly seven?', 'depends_on': ['text']},
        }, {'number': 7, 'text': 'seven', 'check': True}),
        ('enum24', {
            'pick': {'type': 'integer', 'enum': list(range(24)), 'instructions': 'Return 23.', 'depends_on': []},
            'check': {'type': 'boolean', 'instructions': 'Is pick equal to 23?', 'depends_on': ['pick']},
        }, {'pick': 23, 'check': True}),
        ('auto_batch', {
            'a': {'type': 'boolean', 'instructions': 'Is 2 plus 2 equal to 4?'},
            'b': {'type': 'boolean', 'instructions': 'Is 2 plus 2 equal to 5?'},
        }, {'a': True, 'b': False}),
    ]


def run(args, name, questions, expected, thinking):
    client = TypeLLMClient(args.url, model=args.model, text_max_tokens=64)
    client.sglang.thinking_budget = args.thinking_budget  # for every field that thinks
    requests = []
    original = client.sglang._request

    def record(path, payload=None, **kwargs):
        start = time.perf_counter()
        response = original(path, payload, **kwargs)
        if path == '/generate':
            responses = response if isinstance(response, list) else [response]
            requests.append({'seconds': time.perf_counter() - start,
                             'batch': isinstance(payload.get('text'), list),
                             'sampling_params': payload.get('sampling_params'),
                             'meta': [r.get('meta_info', {}) for r in responses]})
        return response

    client.sglang._request = record
    start = time.perf_counter()
    row = {'case': name, 'thinking': thinking}
    try:
        context = 'Follow the named dependency results carefully.\n' + ('Background: this is a deterministic dependency test.\n' * 100)
        result = client.generate(context=context, questions={k: {**v, 'thinking': True} for k, v in questions.items()} if thinking else questions).result
        values = {k: v['value'] if isinstance(v, dict) else v for k, v in result.items()}
        prompts = dict(zip(questions, client._last_prompts.get()))
        tokenizer = client.sglang._get_chat_tokenizer()
        edges = []
        for field, spec in questions.items():
            parents = spec.get('depends_on', [])
            if not parents:
                continue
            parent = max(parents, key=lambda p: len(prompts[p]))
            parent_ids = tokenizer.encode(prompts[parent], add_special_tokens=False)
            child_ids = tokenizer.encode(prompts[field], add_special_tokens=False)
            common = 0
            for a, b in zip(parent_ids, child_ids):
                if a != b:
                    break
                common += 1
            edges.append({'parent': parent, 'child': field,
                          'string_prefix': prompts[field].startswith(prompts[parent]),
                          'parent_tokens': len(parent_ids), 'shared_tokens': common})
        isolation = name != 'diamond' or all('ISOLATED_BRANCH_SENTINEL' not in prompts[k] for k in ('root', 'left', 'right', 'final'))
        prefix_ok = all(e['string_prefix'] and e['shared_tokens'] >= e['parent_tokens'] - 1 for e in edges)
        row.update(result=result, correct=values == expected, isolation=isolation,
                   edges=edges, prefix_ok=prefix_ok, prompts=prompts,
                   passed=values == expected and isolation and prefix_ok)
    except Exception as exc:
        row.update(passed=False, error=f'{type(exc).__name__}: {exc}')
    row.update(seconds=time.perf_counter() - start, requests=requests)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--model', default=None)
    parser.add_argument('--thinking-budget', type=int, default=512)
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    client = TypeLLMClient(args.url, model=args.model)
    manifest = {'model': client.sglang._model_info(), 'source_sha256': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('typellm*.py')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    with (output / 'results.jsonl').open('w') as f:
        for thinking in (False, True):
            selected_cases = cases()
            if args.diagnostics:
                name, questions, expected = cases()[0]
                for field, candidates in {'root': [7, 11], 'left': [8, 12], 'right': [9, 13], 'final': [17, 21]}.items():
                    questions[field]['enum'] = candidates
                selected_cases = [('diamond', questions, expected)]
                for target in (9, 17):
                    selected_cases.append((f'standalone24_{target}', {
                        'answer': {'type': 'integer', 'enum': list(range(24)),
                                   'instructions': f'Return {target}.'},
                    }, {'answer': target}))
            for name, questions, expected in selected_cases:
                row = run(args, name, questions, expected, thinking)
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                print(json.dumps({k: v for k, v in row.items() if k not in ('requests', 'prompts')}, ensure_ascii=False), flush=True)
    summary = {'passed': sum(r['passed'] for r in rows), 'total': len(rows),
               'cache_note': 'Warm shared server; no cache flush. Metadata demonstrates reuse, not a cold-cache benchmark.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return 0 if all(r['passed'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
