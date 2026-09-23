"""CPU-only preparation. Explicit asset config; no model training and no test mixing."""
import argparse, json, random
from pathlib import Path
from transformers import AutoTokenizer
from core import read,write,sha,model_fingerprint

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='suite.json');ap.add_argument('--dataset',choices=['dolly','citb'],required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    suite=read(a.config); c=suite['datasets'][a.dataset]; out=Path(a.output)
    assert not (out/'manifest.json').exists(),'Do not overwrite an existing fixed data split'
    tok=AutoTokenizer.from_pretrained(c['student'],use_fast=False)
    tt=AutoTokenizer.from_pretrained(c['teacher'],use_fast=False,trust_remote_code=True)
    assert tok.get_vocab()==tt.get_vocab()
    manifest={'dataset':a.dataset,'config':c,'sources':{},'excluded':[],'format_version':1}
    manifest['teacher_sha256']=model_fingerprint(c['teacher'])
    manifest['student_sha256']=model_fingerprint(c['student'])
    def row(key,prompt,refs,**extra):
        assert 0<len(prompt)<=c['prompt_cap'] and refs
        response=tok.encode(refs[0],add_special_tokens=False)[:c['response_cap']-1]+[tok.eos_token_id]
        return dict(source_id=key,input_ids=prompt+response,prompt_len=len(prompt),output=refs,**extra)
    splits={}
    if a.dataset=='dolly':
        for split in ['train','valid']:
            splits[split]=read(c[split]);manifest['sources'][split]=sha(c[split])
        manifest['note']='Existing Dolly preprocessing retained byte-for-byte at token level; gold sequences may use total512 budget.'
    else:
        from official_ni_collator import DataCollatorForNI
        collator=DataCollatorForNI(tokenizer=tok,max_source_length=c['prompt_cap'],max_target_length=c['response_cap'],add_task_name=False,add_task_definition=True,num_pos_examples=2,num_neg_examples=0,add_explanation=False,tk_instruct=False,text_only=True)
        random.seed(42)
        for source,split in [('train','train'),('dev','valid'),('initial_test','initial_test'),('cl_test','cl_test'),('official_test','official_test')]:
            path=Path(c['raw_dir'])/(source+'.json');manifest['sources'][source]=sha(path);splits[split]=[]
            for r in read(path):
                text=collator([r])['inputs'][0];p=tok(text,max_length=c['prompt_cap'],truncation=True).input_ids
                splits[split].append(row(r['Task']+':'+r['Instance']['id'],p,r['Instance']['output'],task=r['Task'],categories=r['Categories']))
        train_keys={r['source_id'] for r in splits['train']}
        for split in splits:
            if split!='train': assert not train_keys & {r['source_id'] for r in splits[split]},split
        manifest['note']='Official fixed CITB split and collator retained; first gold reference fixed for history/probe only. GKD learns generated responses; evaluation uses all references.'
    manifest['counts']={k:len(v) for k,v in splits.items()}
    for name,rows in splits.items():write(out/(name+'.json'),rows)
    manifest['hashes']={name:sha(out/(name+'.json')) for name in splits}
    write(out/'manifest.json',manifest);print(json.dumps(manifest['counts']))
if __name__=='__main__':main()
