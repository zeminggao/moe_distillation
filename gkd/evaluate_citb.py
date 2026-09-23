"""Fixed final student, all three held-out CITB sets, greedy128 and full references."""
import argparse,re,string
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from rouge_score import rouge_scorer
from core import read,write

def normalize(s):return ' '.join(s.lower().translate(str.maketrans('','',string.punctuation)).split())
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=4);a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.checkpoint,use_fast=False);tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,torch_dtype=torch.bfloat16).cuda().eval()
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    for split in ['initial_test','cl_test','official_test']:
        path=Path(a.output)/(split+'.json');assert not path.exists();rows=read(Path(a.data)/(split+'.json'));records=[]
        for start in range(0,len(rows),a.batch_size):
            part=rows[start:start+a.batch_size];ids=torch.full((len(part),1024),tok.eos_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
            for j,r in enumerate(part):
                prompt=r['input_ids'][:r['prompt_len']];assert len(prompt)<=1024
                ids[j,-len(prompt):]=torch.tensor(prompt,device='cuda');mask[j,-len(prompt):]=1
            with torch.no_grad():seq=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
            for j,r in enumerate(part):
                pred=tok.decode(seq[j,1024:],skip_special_tokens=True).strip();refs=r['output']
                records.append(dict(source_id=r['source_id'],task=r['task'],prediction=pred,references=refs,rougeL=100*max(scorer.score(x,pred)['rougeL'].fmeasure for x in refs),exact_match=100*max(normalize(x)==normalize(pred) for x in refs)))
        tasks={t:[r for r in records if r['task']==t] for t in {r['task'] for r in records}}
        by_task={t:{m:sum(r[m] for r in rs)/len(rs) for m in ['rougeL','exact_match']} for t,rs in tasks.items()}
        write(path,dict(samples=len(records),rougeL=sum(r['rougeL'] for r in records)/len(records),exact_match=sum(r['exact_match'] for r in records)/len(records),task_macro_rougeL=sum(v['rougeL'] for v in by_task.values())/len(by_task),by_task=by_task,records=records))
if __name__=='__main__':main()
