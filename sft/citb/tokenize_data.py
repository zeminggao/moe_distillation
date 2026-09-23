import os
import json, hashlib, random
from pathlib import Path
from transformers import AutoTokenizer
from official_ni_collator import DataCollatorForNI
ROOT=Path(os.environ['CITB_WORK_DIR']).resolve()
tok=AutoTokenizer.from_pretrained(os.environ['TEACHER_MODEL'],trust_remote_code=True,use_fast=False)
c=DataCollatorForNI(tokenizer=tok,max_source_length=1024,max_target_length=128,add_task_name=False,add_task_definition=True,num_pos_examples=2,num_neg_examples=0,add_explanation=False,tk_instruct=False,text_only=True)
manifest=json.loads((ROOT/'data/manifest.json').read_text());manifest['tokenized']={}
random.seed(42)
for name in manifest['counts']:
    f=ROOT/'data'/f'{name}.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest['sha256'][name]
    rows=json.loads(f.read_text());out=[]
    for i,r in enumerate(rows):
        text=c([r])['inputs'][0]
        prompt=tok(text,max_length=1024,truncation=True)['input_ids']
        refs=r['Instance']['output'];assert refs
        targets=[tok.encode(s,add_special_tokens=False)[:127]+[tok.eos_token_id] for s in refs]
        out.append({'sample_id':r['Task']+':'+r['Instance']['id'],'task':r['Task'],'categories':r['Categories'],'prompt':prompt,'targets':targets,'references':refs})
        if i%1000==0:print(name,i,flush=True)
    dest=ROOT/'data'/f'{name}_tokens.json';dest.write_text(json.dumps(out,ensure_ascii=False))
    manifest['tokenized'][name]={'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'instances':len(out),'max_prompt':max(len(x['prompt']) for x in out),'max_target':max(len(t) for x in out for t in x['targets'])}
(ROOT/'data/tokenization_manifest.json').write_text(json.dumps(manifest,indent=2))
print('TOKENIZATION COMPLETE',flush=True)
