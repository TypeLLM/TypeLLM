# A fair die with permutation averaging

Ask Qwen the same fair-die question with the native per-field `permutations` API.
The expected probability of each outcome is 1/6. The example compares one
ordering, the balanced set of six from `"auto"`, eight sampled distinct
orderings, and all 720 orderings.

```python
from typellm import TypeLLMClient

client = TypeLLMClient("http://127.0.0.1:30000", seed=42)
result = client.generate(
    context="A single roll of a fair die.",
    questions={"roll": {
        "type": "string",
        "enum": ["one", "two", "three", "four", "five", "six"],
        "instructions": "what number will come up on a single roll of a fair six-sided die?",
        "permutations": "auto",
        "return_probabilities": True,
    }},
).result
print(result["roll"])
```

Only explicit enum questions support `permutations`. The client handles
reordering, batch scoring, probability alignment, and averaging; the return
format remains `{value, probabilities}`. `1` preserves the original ordering,
`"auto"` evaluates six balanced orderings, `8` samples eight distinct
permutations, and `"all"` evaluates all 720.

## Run

Install TypeLLM from this checkout (`pip install -e .`) and serve
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` through SGLang with the served name
`qwen3.8-27b`. From the repository root:

```bash
python examples/fair_die/fair_die.py --url http://127.0.0.1:30000
```

Use `--model`, `--tokenizer`, and `--output` to override the defaults.
The script writes `result.json`, including probabilities, KL(P || Q), model
configuration, and elapsed time. P is uniform; lower KL is better. A `null` KL
means the prediction contains a zero probability, giving infinite divergence.

The 8-permutation result is one seeded sample, not an average over many subsets.
This example tests one known distribution; it does not establish calibration
across tasks. Each run uses one `generate` call, which can issue multiple backend
requests.

## Recorded GPU result

Run on September 26, 2026, using the model and settings above.

| Permutations | one | two | three | four | five | six | KL(P ∥ Q) | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66.19% | 0.94% | 13.03% | 14.77% | 1.76% | 3.30% | 0.95417 | 1.43 s |
| auto (6) | 35.39% | 13.02% | 12.46% | 12.38% | 10.43% | 16.32% | 0.09534 | 0.13 s |
| 8 | 40.28% | 15.19% | 22.71% | 6.31% | 5.23% | 10.27% | 0.25249 | 0.14 s |
| 720 (all) | 33.43% | 11.96% | 14.97% | 13.35% | 11.49% | 14.81% | 0.07595 | 7.39 s |
| Ground truth | 16.67% | 16.67% | 16.67% | 16.67% | 16.67% | 16.67% | 0 | |

The first run's time includes loading the tokenizer.

See [result.json](result.json) for full precision. This run uses a single field named `roll` for every ordering. Earlier experiments used separate `order_N` fields, so their prompts and numerical results differ. Six balanced orderings get close to the 720-order mean at a small fraction of the cost; both are much closer to uniform, but still favor `one`.
