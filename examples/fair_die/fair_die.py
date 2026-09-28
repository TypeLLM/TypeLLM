"""Compare one ordering, a balanced set of six, eight sampled orderings, and all 720."""
import argparse
import json
import math
from pathlib import Path
import time

from typellm import TypeLLMClient

LABELS = ['one', 'two', 'three', 'four', 'five', 'six']
QUESTION = 'what number will come up on a single roll of a fair six-sided die?'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--model', default='qwen3.8-27b')
    parser.add_argument('--tokenizer', default='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('result.json'))
    args = parser.parse_args()
    client = TypeLLMClient(args.url, model=args.model, tokenizer=args.tokenizer,
                           seed=42, timeout=300)
    runs = {}
    for budget in (1, 'auto', 8, 'all'):
        start = time.monotonic()
        answer = client.generate(
            context='A single roll of a fair die.',
            questions={'roll': {
                'type': 'string', 'enum': LABELS, 'instructions': QUESTION,
                'permutations': budget, 'return_probabilities': True,
            }},
        ).result['roll']
        probs = answer['probabilities']
        assert set(probs) == set(LABELS)
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
        assert math.isclose(sum(probs.values()), 1, abs_tol=1e-9)
        assert answer['value'] == max(probs, key=probs.get)
        kl = sum(math.log((1/6)/p)/6 for p in probs.values()) if all(probs.values()) else None
        runs[str(budget)] = {'answer': answer, 'kl_uniform_to_prediction': kl,
                            'seconds': time.monotonic() - start}
        print(json.dumps({'permutations': budget, **runs[str(budget)]}), flush=True)
    record = {'question': QUESTION, 'context': 'A single roll of a fair die.',
              'model': args.model, 'tokenizer': args.tokenizer, 'seed': 42,
              'thinking': False, 'mode': 'argmax',
              'temperature': 1.0, 'ground_truth': dict.fromkeys(LABELS, 1/6),
              'runs': runs}
    args.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
