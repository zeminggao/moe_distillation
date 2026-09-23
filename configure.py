"""Generate independent run configurations from explicitly supplied local assets."""
import argparse,hashlib,json
from pathlib import Path
METHODS=('top2','top8','strawman','kid','ka','sar')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--objective',choices=['gkd','taid'],required=True);p.add_argument('--dataset',choices=['dolly','citb'],required=True);p.add_argument('--method',choices=METHODS,required=True)
 for n in ['teacher','student','data','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();data=Path(a.data).resolve();out=Path(a.output).resolve();teacher=Path(a.teacher).resolve();student=Path(a.student).resolve()
 assert not out.exists(),'Use a new output directory; never overwrite an existing run'
 for model in [teacher,student]:assert (model/'config.json').is_file(),model
 rows=json.loads((data/'train.json').read_text());valid=json.loads((data/'valid.json').read_text());assert rows and valid
 if a.method in ('kid','strawman'):
  history=data/'history';m=json.loads((history/'manifest.json').read_text());assert m['teacher']==str(teacher) and m['train_sha256']==sha(data/'train.json');assert (history/'m80_expert_bitmask.npy').is_file()
 P,R=(256,256) if a.dataset=='dolly' else (1024,128)
 c=dict(dataset=a.dataset,method=a.method,teacher=str(teacher),student=str(student),global_batch=384,seed=42,prompt_cap=P,response_cap=R)
 out.mkdir(parents=True)
 if a.objective=='gkd':
  c.update(data=str(data),world_size=6,epochs=10,lr=2e-5,temperature=1.,train_microbatch=4,generation_microbatch=4,eval_microbatch=4,router_microbatch=1,probe_interval=10,kid_full_batch_teacher=a.method=='kid',eval='rouge_sampling' if a.dataset=='dolly' else 'rouge_greedy')
 else:
  epochs=5 if a.dataset=='dolly' else 10
  assets={'data':{name:sha(data/(name+'.json')) for name in ['train','valid']}}
  (out/'assets.json').write_text(json.dumps(assets,indent=2))
  c.update(objective='TAID',world_size=2,epochs=epochs,scheduler_epochs=epochs,lr=1e-4,student_lr=1e-4,weight_decay=.01,gradient_clip=1.,microbatch=4,router_lr=2e-5,router_microbatch=1,t_start=.2,t_end=1.,alpha=5e-4,beta=.99,eval_seed=10,eval_microbatch=4,eval_sampling=a.dataset=='dolly',job_root=str(out),train=str(data/'train.json'),valid=str(data/'valid.json'),valid_count=len(valid),history=str(data/'history/m80_expert_bitmask.npy'),assets_manifest=str(out/'assets.json'),tests={n:str(data/(n+'.json')) for n in ['initial_test','cl_test','official_test'] if (data/(n+'.json')).exists()})
 (out/'config.json').write_text(json.dumps(c,indent=2));print(out/'config.json')
if __name__=='__main__':main()
