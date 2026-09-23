import os
import json,hashlib,types,sys
from pathlib import Path
from transformers import AutoTokenizer
root=Path(os.environ['DOLLY_WORK_DIR']).resolve()
root.mkdir(parents=True,exist_ok=True)
model=os.environ['TOKENIZER_MODEL']
tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,use_fast=False)
# Execute the unmodified official Encoder class; omit unrelated mmap-builder imports.
source=(Path(os.environ['MINILLM_SOURCE_DIR'])/'tools/process_data_dolly.py').read_text()
ns={"json":json,"AutoTokenizer":AutoTokenizer}
exec(source[source.index('class Encoder'):source.index('def main():')],ns)
Encoder=ns['Encoder'];enc=Encoder(types.SimpleNamespace(model_type='llama',max_prompt_length=256,model_path=model));Encoder.tokenizer=tok
lines=(Path(os.environ['DOLLY_RAW_JSONL'])).read_text().splitlines()
manifest={'source':'https://huggingface.co/datasets/MiniLLM/dolly','raw_sha256':hashlib.sha256((Path(os.environ['DOLLY_RAW_JSONL'])).read_bytes()).hexdigest(),'source_count':len(lines),'dev_num_before_filter':1000,'split_rule':'official raw first 1000 valid, remainder train','splits':{}}
for name,indexes in [('valid',range(1000)),('train',range(1000,len(lines)))]:
    rows=[];excluded=[]
    for idx in indexes:
        row,prompt,pids,rids,_=enc.encode(lines[idx])
        if row is None:excluded.append(idx);continue
        full=pids+rids
        assert len(full[:512])>len(pids)
        rows.append({'source_id':idx,'input_ids':full[:512],'prompt_len':len(pids),'prompt':prompt,'output':row['output'],'truncated':len(full)>512})
    path=root/(name+'.json');path.write_text(json.dumps(rows,ensure_ascii=False))
    manifest['splits'][name]={'retained':len(rows),'excluded_prompt_over_256':excluded,'truncated_response':sum(x['truncated'] for x in rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(root/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
