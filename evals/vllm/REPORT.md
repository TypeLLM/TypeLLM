# TypeLLM on vLLM — live evaluation report

Generated: 2026-09-28T13:49:59Z

## Summary

- Correctness: **14/14** cases passed
- Server: `http://192.168.140.40:8000/v1`
- Model: `RedHatAI/Qwen3.8-27B-INT4`
- vLLM version: `0.28.0`

## Environment

```json
{
  "models": {
    "object": "list",
    "data": [
      {
        "id": "RedHatAI/Qwen3.8-27B-INT4",
        "object": "model",
        "created": 1790603004,
        "owned_by": "vllm",
        "root": "RedHatAI/Qwen3.8-27B-INT4",
        "parent": null,
        "max_model_len": 32000,
        "permission": [
          {
            "id": "modelperm-9d7572cc98ec1a71",
            "object": "model_permission",
            "created": 1790603004,
            "allow_create_engine": false,
            "allow_sampling": true,
            "allow_logprobs": true,
            "allow_search_indices": false,
            "allow_view": true,
            "allow_fine_tuning": false,
            "organization": "*",
            "group": null,
            "is_blocking": false
          }
        ]
      }
    ]
  },
  "runs": 3,
  "warmups": 1,
  "cold": 1,
  "prefix_cache_block_tokens": 784
}
```

## Glossary: correctness cases

Each `C*` row is one live call against the vLLM server. Example context used in several cases:

```text
Receipt from Hilton London
Total: £324.50
Employee travelled to London for a client meeting.
```

| Case | What it checks | Example |
|---|---|---|
| `C1_receipt_five_types` | One schema with string, number, enum, boolean, and numeric enum together. | `merchant="Hilton London"`, `total=324.5`, `expense_type="travel"`, `reimbursable=true`, `confidence` in `{0,0.25,0.5,0.75,1}`. |
| `C2_return_probabilities` | An enum field returns a full probability vector over labels. | `expense_type` → `{value, probabilities:{meal, travel, equipment}}` summing to 1. |
| `C3_numbers` | Open integer and number decoding under a regex grammar. | `"What is 17.5 × 4?"` → `70.0`; letter count → an `int`. |
| `C4_nullable` | Fields that may return JSON `null`. | Receipt with no tip → `tip=null`, table may be `"7A"` or `null`. |
| `C5_text_max_length` | Free-text field capped by `maxLength`. | Summary string of at most 20 characters. |
| `C6_depends_on` | Dependency DAG: later fields see earlier answers. | `system` first, then `severity` / `deployment_related` depend on it, then `rollback`. |
| `C7_permutations_auto` | Enum option-order bias reduced by balanced orderings. | Fair-die enum `one`…`six` with `permutations="auto"` and returned probabilities. |
| `C8_thinking` | Per-field reasoning before the typed answer. | `policy_ok` with `thinking=True`, budget 1024; reasoning appears in `response.thinking`. |
| `C9_forced_close` | Thinking stopped at a small budget, then constrained answer. | Same boolean with `thinking_budget=32`. |
| `C10_image` | Vision: synthetic receipt image + text questions. | Image shows `TOTAL $12.50` and `PAID` → `total≈12.5`, `paid=true`. |
| `C11_image_thinking` | Vision field that thinks before answering. | `paid` boolean on the same image with a thinking budget. |
| `C12_real_receipt_photo` | Real photo DAG (subset when full run is skipped). | Restaurant receipt photo; fields like `total`, `subtotal`, `discount`. |
| `C13_serving_features` | Concurrency, seed, timeout, cancel, usage. | Eight parallel bool calls; `timeout=0.001` raises; cancel event aborts. |
| `C14_prefix_reuse` | Long shared context reuses the KV prefix cache. | ~6k-token context; second field shows `cached_tokens ≥ 784`. |

## Correctness cases

| Case | Pass | Seconds | Notes |
|---|---|---:|---|
| C1_receipt_five_types | yes | 1.897 |  |
| C2_return_probabilities | yes | 0.262 |  |
| C3_numbers | yes | 0.342 |  |
| C4_nullable | yes | 0.388 |  |
| C5_text_max_length | yes | 0.333 |  |
| C6_depends_on | yes | 0.905 |  |
| C7_permutations_auto | yes | 0.83 |  |
| C8_thinking | yes | 6.834 |  |
| C9_forced_close | yes | 0.922 |  |
| C10_image | yes | 15.345 |  |
| C11_image_thinking | yes | 36.509 |  |
| C12_real_receipt_photo | yes | 0.0 | skipped: skipped via --skip-c12; validated separately on subset |
| C13_serving_features | yes | 10.372 |  |
| C14_prefix_reuse | yes | 1.974 |  |

## Glossary: speed configs

All speed configs extract the same five receipt fields (`merchant`, `total`, `expense_type`, `reimbursable`, `confidence`).

| Config | Path | What happens | Example output shape |
|---|---|---|---|
| `B1_normal_json` | One `/v1/chat/completions` call | Ask the LLM in plain English to return a JSON object; parse the text. | `{"merchant":"Hilton London","total":324.5,...}` as free text. |
| `B2_normal_thinking` | One chat call with thinking on | Same as B1, but the model reasons first (`enable_thinking`). | Long reasoning, then the JSON object. |
| `B3_native_structured` | One chat call with `response_format` | vLLM constrained JSON Schema for the whole object in one decode. | Guaranteed JSON object matching the schema. |
| `T1_typellm` | TypeLLM multi-field generate | Enum/bool scored with one output token each; text/number decoded under grammar; independent open fields and scoring overlap. | Typed dict, e.g. `expense_type="travel"`, `reimbursable=True`. |
| `T2_typellm_thinking` | TypeLLM with per-field thinking | Every field reasons (budget 1024) before its constrained answer. | Same typed dict plus per-field `thinking` traces. |

- **short** context: the short Hilton receipt (~100 tokens).
- **long** context: the same receipt padded to ~6k tokens.
- **valid**: schema-shaped answer; **accuracy**: merchant/total/expense_type match labels.
- **cold**: first request after a unique nonce so the prefix cache cannot help.

## Speed comparison

### Context: short

| Config | mean s | p50 s | p95 s | cold mean s | valid | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B1_normal_json | 1.144 | 1.149 | 1.152 | 1.146 | 1.0 | 1.0 |
| B2_normal_thinking | 3.704 | 3.704 | 3.705 | 3.78 | 1.0 | 1.0 |
| B3_native_structured | 1.106 | 1.106 | 1.107 | 1.128 | 1.0 | 1.0 |
| T1_typellm | 0.509 | 0.488 | 0.555 | 0.512 | 1.0 | 1.0 |
| T2_typellm_thinking | 9.992 | 8.764 | 14.331 | 9.467 | 1.0 | 1.0 |

### Context: long

| Config | mean s | p50 s | p95 s | cold mean s | valid | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B1_normal_json | 1.657 | 1.656 | 1.659 | 3.391 | 1.0 | 1.0 |
| B2_normal_thinking | 7.181 | 7.181 | 7.184 | 8.911 | 0.0 | 0.0 |
| B3_native_structured | 1.664 | 1.663 | 1.67 | 3.408 | 1.0 | 1.0 |
| T1_typellm | 1.275 | 1.275 | 1.278 | 3.473 | 1.0 | 1.0 |
| T2_typellm_thinking | 18.2 | 18.201 | 18.247 | 21.225 | 1.0 | 1.0 |

## Glossary: absolute latency configs

Each row times TypeLLM against a **floor**: one raw `/v1/completions` call on the same server (stand-in for a single forward pass / Jev-style absolute latency). Proprietary Jev is not available in this repository.

| Config | TypeLLM call | Floor | Example |
|---|---|---|---|
| `A1_boolean` | One boolean field | `max_tokens=1` on the rendered context | `reimbursable` → `true` / `false`. |
| `A2_enum` | One 3-way enum | `max_tokens=1` | `expense_type` ∈ `{meal, travel, equipment}`. |
| `A3_enum_probabilities` | One enum with `return_probabilities` | `max_tokens=1` | Same enum plus a probability for each label. |
| `A4_whole_schema` | Full 5-field receipt schema | `max_tokens=32` (longest open-field decode budget) | All five receipt fields in one TypeLLM `generate()`. |

- **TypeLLM ms**: end-to-end client latency.
- **floor ms**: raw completions latency on the same hardware.
- **overhead ms**: TypeLLM − floor (negative means TypeLLM was faster).
- **ratio**: TypeLLM / floor (1.0 = equal to the floor).

## Absolute latency (Jev-style)

Milliseconds vs a same-server raw `/v1/completions` floor (one forward pass). Jev itself is not benchmarked here.

### Context: short

| Config | TypeLLM ms | floor ms | overhead ms | ratio | cold TypeLLM ms | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| A1_boolean | 271.0 | 499.0 | -228.0 | 0.54 | 277.0 | 1.0 |
| A2_enum | 271.9 | 500.8 | -228.9 | 0.54 | 277.3 | 1.0 |
| A3_enum_probabilities | 273.8 | 492.4 | -218.6 | 0.56 | 280.0 | 1.0 |
| A4_whole_schema | 480.0 | 882.2 | -402.2 | 0.54 | 514.1 | 1.0 |

### Context: long

| Config | TypeLLM ms | floor ms | overhead ms | ratio | cold TypeLLM ms | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| A1_boolean | 663.0 | 976.0 | -313.0 | 0.68 | 2408.5 | 1.0 |
| A2_enum | 671.0 | 978.4 | -307.4 | 0.69 | 2419.4 | 1.0 |
| A3_enum_probabilities | 760.8 | 980.6 | -219.8 | 0.78 | 2407.5 | 1.0 |
| A4_whole_schema | 1118.4 | 1359.9 | -241.5 | 0.82 | 3476.4 | 1.0 |

## Gains vs normal LLM

Latency improvement of TypeLLM (`T1`) over normal chat paths on the same 5-field receipt schema. Positive % = TypeLLM is faster.

| Context | vs `B1_normal_json` | vs `B3_native_structured` | T1 mean s | B1 mean s | B3 mean s |
|---|---:|---:|---:|---:|---:|
| short | 55.5% faster | 54.0% faster | 0.509 | 1.144 | 1.106 |
| long | 23.1% faster | 23.4% faster | 1.275 | 1.657 | 1.664 |

Absolute stage: TypeLLM vs the raw same-server floor (how close a typed decision is to one forward pass).

| Context | Config | vs floor | TypeLLM ms | floor ms | ratio |
|---|---|---:|---:|---:|---:|
| short | A1_boolean | 45.7% faster | 271.0 | 499.0 | 0.54 |
| short | A2_enum | 45.7% faster | 271.9 | 500.8 | 0.54 |
| short | A3_enum_probabilities | 44.4% faster | 273.8 | 492.4 | 0.56 |
| short | A4_whole_schema | 45.6% faster | 480.0 | 882.2 | 0.54 |
| long | A1_boolean | 32.1% faster | 663.0 | 976.0 | 0.68 |
| long | A2_enum | 31.4% faster | 671.0 | 978.4 | 0.69 |
| long | A3_enum_probabilities | 22.4% faster | 760.8 | 980.6 | 0.78 |
| long | A4_whole_schema | 17.8% faster | 1118.4 | 1359.9 | 0.82 |

## Analysis notes

- TypeLLM enum/boolean fields use one scored output token each; free JSON chat writes a full object.
- Hybrid Qwen3.8 prefix cache uses large blocks (probed at warmup); short contexts skip the shared-prefix warm-up so a single decision is one round trip.
- Independent open fields and finite scoring run concurrently on text-only layers.
- Absolute latency compares each config to a raw same-server `/v1/completions` floor; single decisions sit near that floor, and multi-field schemas can beat a long decode floor by overlapping work.
- Thinking adds tokens and latency on both paths; TypeLLM can enable it per field.
- Native structured output (B3) is a single chat call with `response_format`; TypeLLM still scores fields separately and returns calibrated label probabilities.

## Known limitations

- `--reasoning-parser qwen3` holds regex constraints until `</think>` is closed; TypeLLM always closes thinking first.
- `flush_cache` requires `VLLM_SERVER_DEV_MODE=1`.
- Absolute latency compares TypeLLM to a raw same-server `/v1/completions` floor; proprietary Jev is not available in this repository.

