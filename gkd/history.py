"""Six-card train-only history. Construct exactly one table shared by Strawman/KID."""
import argparse,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from core import read,write,sha
from benchmark_gkd import load

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--teacher',required=True);p.add_argument('--microbatch',type=int,default=1);a=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==6
    torch.cuda.set_device(rank);torch.set_num_threads(4);dist.init_process_group('nccl')
    out=Path(a.data)/'history'
    if rank==0:out.mkdir(exist_ok=False)
    dist.barrier();start=time.time();teacher=load(a.teacher,torch.bfloat16).cuda().eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(a.teacher,use_fast=False,trust_remote_code=True);special=set(tok.all_special_ids)
    rows=read(Path(a.data)/'train.json');layers=teacher.model.layers;counts=torch.zeros((teacher.config.vocab_size,len(layers),8),device='cuda',dtype=torch.int64);state={};targets=0
    def hook(layer):
        def collect(module,args,result):
            picked=result['topK_indices'][state['positions']];assert picked.shape[-1]==2
            ids=state['targets'][:,None].expand_as(picked)
            counts[:,layer,:].index_put_((ids.reshape(-1),picked.reshape(-1)),torch.ones(picked.numel(),device='cuda',dtype=torch.int64),accumulate=True)
        return collect
    handles=[l.mlp.gate.register_forward_hook(hook(i)) for i,l in enumerate(layers)]
    local=rows[rank::world]
    for offset in range(0,len(local),a.microbatch):
        part=local[offset:offset+a.microbatch];length=max(len(r['input_ids']) for r in part)
        ids=torch.full((len(part),length),tok.eos_token_id,device='cuda',dtype=torch.long);attn=torch.zeros_like(ids);pos=[];ys=[]
        for j,row in enumerate(part):
            seq=row['input_ids'];ids[j,:len(seq)]=torch.tensor(seq,device='cuda');attn[j,:len(seq)]=1
            for k in range(row['prompt_len'],len(seq)):
                if seq[k] not in special:pos.append(j*length+k-1);ys.append(seq[k])
        state.update(positions=torch.tensor(pos,device='cuda',dtype=torch.long),targets=torch.tensor(ys,device='cuda',dtype=torch.long));targets+=len(ys)
        with torch.no_grad():teacher.model(input_ids=ids,attention_mask=attn,use_cache=False)
    dist.all_reduce(counts);n=torch.tensor(targets,device='cuda');dist.all_reduce(n)
    if rank==0:
        arr=counts.cpu().numpy();tot=arr.sum(-1);assert np.all(tot.sum(0)==2*int(n))
        order=np.argsort(-arr,axis=-1,kind='stable');cum=np.take_along_axis(arr,order,-1).cumsum(-1)
        need=np.where(tot>0,(cum*5<tot[...,None]*4).sum(-1)+1,0)
        bits=((1<<order)*(np.arange(8)<need[...,None])).sum(-1).astype(np.uint8)
        np.save(out/'native_top2_counts.npy',arr);np.save(out/'m80_expert_bitmask.npy',bits)
        write(out/'manifest.json',dict(teacher=str(Path(a.teacher).resolve()),train_sha256=sha(Path(a.data)/'train.json'),samples=len(rows),targets=int(n),coverage=.8,tie_break='expert_id',seconds=time.time()-start,source='training gold response targets; prediction position target-1; special excluded',counts_sha256=sha(out/'native_top2_counts.npy')))
    dist.destroy_process_group()
if __name__=='__main__':main()
