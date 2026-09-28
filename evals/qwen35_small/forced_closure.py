"""Live GPU regression for truncated reasoning; run independent requests serially."""
import argparse
import json
import time
from pathlib import Path

from run import TypeLLMClient, cases, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    warm = TypeLLMClient('http://127.0.0.1:30000', model=args.model, tokenizer=args.model)
    tokenizer = warm.sglang._get_chat_tokenizer()
    numeric = warm.sglang.numeric_token_pieces()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for budget, selected in ((32, cases()), (None, [cases()[0]])):
            for name, context, questions, expected in selected:
                client = TypeLLMClient('http://127.0.0.1:30000', model=args.model,
                    tokenizer=args.model, text_max_tokens=128, timeout=300, seed=42)
                backend = client.sglang
                backend.thinking_budget = budget  # for every field that thinks
                backend._chat_tokenizer = tokenizer
                backend._numeric_tokens = numeric
                row = dict(model=args.model, case=name,
                           thinking_budget=budget, reasoning=[], closures=[])
                request = backend._request
                complete = backend._complete_thinking

                def record_request(endpoint, payload=None, **kwargs):
                    response = request(endpoint, payload, **kwargs)
                    params = payload.get('sampling_params', {}) if payload else {}
                    # A batch of prompts has one sampling-params dict and one response each.
                    for item_params, item in (zip(params, response)
                                              if isinstance(params, list) and isinstance(response, list)
                                              else [(params, response)]):
                        if isinstance(item_params, dict) and item_params.get('stop') == ['</think>']:
                            meta = item.get('meta_info', {})
                            row['reasoning'].append(dict(
                                max_new_tokens=item_params['max_new_tokens'],
                                completion_tokens=meta.get('completion_tokens'),
                                finish_reason=meta.get('finish_reason')))
                    return response

                # Single and batched thinking both close each prompt here, one at a time.
                def record_complete(prefix, response, image_tokens):
                    completed = complete(prefix, response, image_tokens)
                    used = len(tokenizer.encode(completed, add_special_tokens=False))
                    row['closures'].append(dict(
                        forced=completed.endswith('\n\nI will now give the final answer.\n</think>\n\n'),
                        context_length=backend._context_length(), prompt_tokens=used,
                        answer_reserve=backend.answer_reserve_tokens,
                        reserve_valid=used + backend.answer_reserve_tokens <= backend._context_length()))
                    return completed

                backend._request = record_request
                backend._complete_thinking = record_complete
                started = time.monotonic()
                try:
                    result = client.generate(context=context, questions={k: {**v, 'thinking': True} for k, v in questions.items()}).result
                    values = {k: v['value'] if questions[k].get('return_probabilities') else v
                              for k, v in result.items()}
                    row.update(result=result, type_valid=validate(result, questions), correct=values == expected)
                except Exception as exc:
                    row.update(error=f'{type(exc).__name__}: {exc}', type_valid=False, correct=False)
                row['seconds'] = round(time.monotonic() - started, 3)
                encoded = json.dumps(row, ensure_ascii=False)
                stream.write(encoded + '\n')
                stream.flush()
                print(encoded, flush=True)


if __name__ == '__main__':
    main()
