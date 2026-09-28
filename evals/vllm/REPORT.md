# TypeLLM on vLLM — live evaluation report

Generated: 2026-09-28T12:48:42Z

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
        "created": 1790599376,
        "owned_by": "vllm",
        "root": "RedHatAI/Qwen3.8-27B-INT4",
        "parent": null,
        "max_model_len": 32000,
        "permission": [
          {
            "id": "modelperm-a317f594ea9324c1",
            "object": "model_permission",
            "created": 1790599376,
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
  "cold": 1
}
```

## Correctness cases

| Case | Pass | Seconds | Notes |
|---|---|---:|---|
| C1_receipt_five_types | yes | 2.648 |  |
| C2_return_probabilities | yes | 0.533 |  |
| C3_numbers | yes | 0.36 |  |
| C4_nullable | yes | 0.929 |  |
| C5_text_max_length | yes | 0.34 |  |
| C6_depends_on | yes | 1.719 |  |
| C7_permutations_auto | yes | 1.278 |  |
| C8_thinking | yes | 13.253 |  |
| C9_forced_close | yes | 1.199 |  |
| C10_image | yes | 24.589 |  |
| C11_image_thinking | yes | 23.59 |  |
| C12_real_receipt_photo | yes | 321.094 | 6/7 |
| C13_serving_features | yes | 4.266 |  |
| C14_prefix_reuse | yes | 3.945 |  |

## Speed comparison

### Context: short

| Config | mean s | p50 s | p95 s | cold mean s | valid | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B1_normal_json | 1.143 | 1.144 | 1.144 | 1.147 | 1.0 | 1.0 |
| B2_normal_thinking | 3.721 | 3.717 | 3.729 | 5.43 | 1.0 | 1.0 |
| B3_native_structured | 1.122 | 1.122 | 1.123 | 1.127 | 1.0 | 1.0 |
| T1_typellm | 1.011 | 1.013 | 1.017 | 1.031 | 1.0 | 1.0 |
| T2_typellm_thinking | 9.347 | 9.836 | 13.795 | 9.069 | 1.0 | 1.0 |

### Context: long

| Config | mean s | p50 s | p95 s | cold mean s | valid | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B1_normal_json | 1.656 | 1.657 | 1.657 | 3.391 | 1.0 | 1.0 |
| B2_normal_thinking | 7.179 | 7.177 | 7.186 | 8.933 | 0.0 | 0.0 |
| B3_native_structured | 1.667 | 1.664 | 1.674 | 3.385 | 1.0 | 1.0 |
| T1_typellm | 2.007 | 1.951 | 2.233 | 4.552 | 1.0 | 1.0 |
| T2_typellm_thinking | 17.094 | 17.748 | 17.968 | 20.433 | 1.0 | 1.0 |

## Analysis notes

- On the short receipt context, TypeLLM without thinking (**T1, 1.01 s**) was slightly faster than a normal JSON chat call (**B1, 1.14 s**) and than native structured output (**B3, 1.12 s**), with 100% schema-valid and accurate answers in all three.
- On the long (~6k-token) context, a single normal JSON call (**B1, 1.66 s**) beat TypeLLM (**T1, 2.01 s**) because TypeLLM issues multiple field requests; prefix-cache reuse still keeps it close.
- Thinking adds large variance: TypeLLM per-field thinking (**T2**) is slower than one-shot chat thinking (**B2**), but stays schema-valid. Long-context B2 dropped to 0% valid JSON in this run when reasoning consumed the answer.
- TypeLLM enum/boolean fields use one scored output token each; free JSON chat writes a full object.
- Hybrid Qwen3.8 prefix cache uses large blocks (784 tokens here), so short contexts show little reuse; C14 confirmed cache hits on long contexts.
- Multimodal fields are much slower (~20-40 s each on this server) because each field goes through `/inference/v1/generate` with image features.
- Native structured output (B3) is a single chat call with `response_format`; TypeLLM still scores fields separately and returns calibrated label probabilities.

## Known limitations

- `--reasoning-parser qwen3` holds regex constraints until `</think>` is closed; TypeLLM always closes thinking first.
- `flush_cache` requires `VLLM_SERVER_DEV_MODE=1`.
- Absolute Jev latency is out of scope; this report compares TypeLLM-on-vLLM with normal vLLM calls on the same server.
- C12 used a 7-field subset of the real receipt photo (6/7 exact label matches; `first_item` spelling differed). The full 14-field multimodal DAG is multi-minute on this GPU.

