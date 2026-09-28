"""Inspect thinking-to-numeric boundary on the failed invoice case."""
import json
import random
from pathlib import Path
from _thinking_eval import ThinkingClient
from typellm_sglang import SGLangClient
from typellm_runtime import Choice, _decode_numeric

class Probe(ThinkingClient):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls=[]
        self.capture=False
    def _request(self, path, payload=None, **kw):
        if payload and payload.get('return_logprob'):
            payload={**payload, 'top_logprobs_num':20}
        r=super()._request(path,payload,**kw)
        if self.capture and payload and payload.get('return_logprob'):
            self.calls.append({'prefix_tail':payload['text'][-90:], 'response':r})
        return r

c=Probe(model='qwen3.8-27b',tokenizer='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead',timeout=180)
d=Choice(question='What is the invoice total?',choices={},name='total',syntax='Number',numeric_type='number')
# Match the actual compiler's syntax instead of assuming it.
from typellm import TypeLLMClient
from _thinking_eval import cases
base=TypeLLMClient('http://127.0.0.1:30000',model='qwen3.8-27b',tokenizer='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead')
case=next(x for x in cases() if x['id']=='batch_mixed')
d=base.compile_schema(case['schema'])[-1]
messages=[{'role':'user','content':case['context']},{'role':'user','content':d.opening_text()}]
p=Path('evals/thinking_diagnosis.jsonl');p.parent.mkdir(exist_ok=True)
with p.open('w') as out:
 for trial in range(6):
    c.capture=False
    try: prefix=c.render_chat(messages,add_generation_prompt=True)
    except Exception as e:
        out.write(json.dumps({'trial':trial,'error':str(e)})+'\n');out.flush();continue
    row={'trial':trial,'thinking_prefix':prefix,'trace':c.traces[-1],'variants':[]}
    for name,suffix in [('two_newlines','\n\n'),('one_newline','\n'),('no_newline','')]:
        prompt=prefix.rsplit('</think>',1)[0]+'</think>'+suffix
        free=c._request('/generate',{'text':prompt,'sampling_params':{'max_new_tokens':48,'temperature':0}})
        c.calls=[];c.capture=True
        value,_,text=_decode_numeric(c,prompt,d,'argmax',1,random.Random(0),32)
        c.capture=False
        row['variants'].append({'name':name,'free_answer':free.get('text'),'value':value,'numeric_text':text,'steps':list(c.calls)})
    out.write(json.dumps(row)+'\n');out.flush()
    print(json.dumps({'trial':trial,'variants':[{k:v for k,v in x.items() if k!='steps'} for x in row['variants']]}),flush=True)
