"""120-case thinking-on regression using the current production numeric prompt."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from _numeric_eval import make_cases, schema_for, matches
from _thinking_eval import ThinkingClient, valid
from typellm import TypeLLMClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remaining', action='store_true', help='Run the eight cases omitted from the 120-case subset')
    args = parser.parse_args()
    out = Path('evals/thinking_numeric8_20260919' if args.remaining else 'evals/thinking_numeric120_20260919')
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    cases = []
    for case in make_cases():
        limit = 16 if case['category'] in ('negative_integer', 'sequential_dependency') else 22
        if counts[case['category']] < limit:
            counts[case['category']] += 1
            cases.append(case)
    assert len(cases) == 120
    if args.remaining:
        selected_ids = {c['id'] for c in cases}
        cases = [c for c in make_cases() if c['id'] not in selected_ids]
        counts = Counter(c['category'] for c in cases)
        assert len(cases) == 8
    (out/'cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases))
    client = TypeLLMClient('http://127.0.0.1:30000',model='qwen3.8-27b',tokenizer='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead',timeout=180)
    client.sglang = ThinkingClient(model='qwen3.8-27b',tokenizer='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead',timeout=180)
    client.sglang.budget = 1024
    metadata = {'cases':len(cases), 'category_counts':dict(counts), 'thinking':True,
                'budget_per_field':1024, 'temperature':.6,'top_p':.95,'top_k':20,
                'decision_mode':'argmax','execution':'sequential',
                'selection':('remaining eight cases' if args.remaining else 'first 22 of each 24-case category; all 16 of each remaining category'),
                'runtime_sha256':hashlib.sha256(Path('typellm_runtime.py').read_bytes()).hexdigest(),
                'model':client.sglang._model_info()}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    with (out/'results.jsonl').open('w') as f:
        for i,case in enumerate(cases,1):
            client.sglang.traces.clear()
            started=time.perf_counter();result=None;error=None;safe=None;correct=False
            schema=schema_for(case)
            try:
                result=client.generate(context=case['context'],schema=schema).result
                safe=valid(result,schema)
                correct=matches(result['answer'],case['expected'])
                if 'followup' in case:
                    correct=correct and result['is_positive']==case['followup']['expected']
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            row={'id':case['id'],'category':case['category'],'expected':case['expected'],
                 'result':result,'correct':correct,'type_safe':safe,'error':error,
                 'seconds':time.perf_counter()-started,'thinking_traces':list(client.sglang.traces)}
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
            if i%10==0 or error or not correct:
                print(json.dumps({'progress':f'{i}/{len(cases)}','correct_so_far':sum(r['correct'] for r in rows),
                                  'last':row}),flush=True)
    def summarize(rr):
        traces=[t for r in rr for t in r['thinking_traces']]
        return {'cases':len(rr),'correct':sum(r['correct'] for r in rr),
                'accuracy':sum(r['correct'] for r in rr)/len(rr),
                'returned':sum(r['error'] is None for r in rr),
                'errors':sum(r['error'] is not None for r in rr),
                'type_violations':sum(r['type_safe'] is False for r in rr),
                'returned_fields':sum(len(r['result'] or {}) for r in rr),
                'mean_seconds':sum(r['seconds'] for r in rr)/len(rr),
                'total_seconds':sum(r['seconds'] for r in rr),
                'thinking_calls':len(traces),'thinking_tokens':sum(t.get('completion_tokens') or 0 for t in traces),
                'unclosed_thinking':sum(not t['closed'] for t in traces)}
    summary={'overall':summarize(rows),'categories':{c:summarize([r for r in rows if r['category']==c]) for c in counts}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY '+json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
