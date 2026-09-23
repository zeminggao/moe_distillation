"""Six-rank GKD; memory-bounded generation/backward, one global optimizer step.
Use suite.py to prepare/preflight; this module never selects a different batch silently.
"""
import os, math, time, random, argparse, contextlib, shutil
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoTokenizer
from core import read,write,sha,save_steps,weighted_scale
from benchmark_gkd import load
from expert_methods import ExpertMethod,sequence_kl
from sar_update import update_router
from dynamic_experts import DynamicExperts,HistoryLookup
from progress_controller import NativeKLProgress

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',dest='run',required=True);ap.add_argument('--steps',type=int,default=0);ap.add_argument('--stress',action='store_true');a=ap.parse_args()
    root=Path(a.run);c=read(root/'config.json');rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==c['world_size']==6 and c['global_batch']%world==0
    torch.cuda.set_device(rank);torch.set_num_threads(4);dist.init_process_group('nccl')
    random.seed(c['seed']+rank);torch.manual_seed(c['seed']+rank)
    out=root/('bench_stress' if a.stress else 'bench_real') if a.steps else root/'run'
    resume=os.environ.get('RESUME_CHECKPOINT') if not a.steps else None
    if resume:assert c['method'] in ('top2','top8','sar'),'Resume requires controller state restoration'
    if rank==0:
        out.mkdir(exist_ok=bool(resume))
        write(out/'config.json',dict(c,benchmark_steps=a.steps,stress=a.stress))
    dist.barrier()
    def emit(kind,**kw):
        if rank==0:
            import json
            event=dict(kind=kind,time=time.time(),**kw)
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            write(out/'status.json',event);print(event,flush=True)
    tok=AutoTokenizer.from_pretrained(c['student'],use_fast=False);tok.pad_token=tok.eos_token
    assert tok.get_vocab()==AutoTokenizer.from_pretrained(c['teacher'],use_fast=False,trust_remote_code=True).get_vocab()
    teacher=load(c['teacher'],torch.bfloat16).cuda().eval().requires_grad_(False)
    if c['method']=='kid':
        import types
        for layer in teacher.model.layers:
            original=layer.self_attn.forward
            def chunk_attention(attn,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,output_attentions=False,use_cache=False,_original=original,**kw):
                assert not use_cache and past_key_value is None and not output_attentions
                n=hidden_states.shape[0]
                def part(x,start):
                    return x[start:start+4] if torch.is_tensor(x) and x.ndim and x.shape[0]==n else x
                pieces=[]
                for start in range(0,n,4):
                    result=_original(hidden_states[start:start+4],attention_mask=part(attention_mask,start),position_ids=part(position_ids,start),past_key_value=None,output_attentions=False,use_cache=False,**{k:part(v,start) for k,v in kw.items()})
                    pieces.append(result[0]);assert result[1] is None and result[2] is None
                return torch.cat(pieces,dim=0),None,None
            layer.self_attn.forward=types.MethodType(chunk_attention,layer.self_attn)
    method=c['method'];policy=ExpertMethod(teacher,method) if method in ('top8','ka','sar') else None
    control=DynamicExperts(teacher) if method in ('strawman','kid') else None
    lookup=HistoryLookup(Path(c['data'])/'history/m80_expert_bitmask.npy') if control else None
    if control:
        manifest=read(Path(c['data'])/'history/manifest.json')
        assert manifest['train_sha256']==sha(Path(c['data'])/'train.json')
        assert manifest['teacher']==str(Path(c['teacher']).resolve())
    student=load(resume or c['student'],torch.float32).cuda();student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ddp=DDP(student,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
    opt=ZeroRedundancyOptimizer(ddp.parameters(),optimizer_class=torch.optim.AdamW,lr=c['lr'],weight_decay=0,parameters_as_bucket_view=True,foreach=False)
    data=read(Path(c['data'])/'train.json');valid=read(Path(c['data'])/'valid.json')
    batch=c['global_batch'];per=math.ceil(len(data)/batch);total=per*c['epochs'];run_epochs=c.get('stop_after_epochs',c['epochs']);assert 0<run_epochs<=c['epochs'];inner=2 if method=='ka' else 1
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total*inner)
    ropt=torch.optim.AdamW(policy.router_parameters,lr=c['lr'],weight_decay=0,foreach=False) if method=='sar' else None
    rsched=torch.optim.lr_scheduler.CosineAnnealingLR(ropt,total) if ropt else None
    progress=NativeKLProgress(interval=c['probe_interval'],warmup_windows=4,beta=.9,alpha=.5,cutoff=.05) if method=='kid' else None
    P,R=c['prompt_cap'],c['response_cap'];L=P+R
    micro=c['train_microbatch'];generation_micro=c['generation_microbatch']
    assert min(micro,generation_micro)>0
    # Layer/rank-local extra budget must retain its original sample grouping.
    if method=='kid':
        assert c.get('kid_full_batch_teacher'), 'KID must preserve whole-rank teacher selection'
    def prompts(rows):
        ids=torch.full((len(rows),P),tok.eos_token_id,device='cuda',dtype=torch.long);attn=torch.zeros_like(ids)
        for j,row in enumerate(rows):
            p=row['input_ids'][:row['prompt_len']];assert 0<len(p)<=P
            ids[j,-len(p):]=torch.tensor(p,device='cuda');attn[j,-len(p):]=1
        return ids,attn
    def generate(rows,stress=False):
        ids=torch.full((len(rows),L),tok.eos_token_id,device='cuda',dtype=torch.long);attn=torch.zeros_like(ids);mask=torch.zeros((len(rows),L-1),device='cuda',dtype=torch.bool)
        student.eval()
        for start in range(0,len(rows),generation_micro):
            part=rows[start:start+generation_micro];p,pa=prompts(part)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                seq=student.generate(input_ids=p,attention_mask=pa,max_new_tokens=R,min_new_tokens=R if stress else 0,do_sample=True,temperature=1.,top_p=1.,top_k=0,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
            for j,row in enumerate(part):
                response=seq[j,P:];end=(response==tok.eos_token_id).nonzero();length=int(end[0])+1 if end.numel() else len(response)
                full=torch.cat([torch.tensor(row['input_ids'][:row['prompt_len']],device='cuda'),response[:length]])
                ids[start+j,:len(full)]=full;attn[start+j,:len(full)]=1;mask[start+j,row['prompt_len']-1:len(full)-1]=True
        return ids,attn,mask
    # Same fixed 4096 gold-target probe and controller as Dolly; variable sequence length.
    probe=[]
    if progress:
        order=list(range(len(data)));random.Random(100045).shuffle(order);remaining=4096;selected=[]
        control.enabled=False
        for index in order:
            row=data[index];positions=list(range(row['prompt_len']-1,len(row['input_ids'])-1))[:remaining]
            if positions:selected.append((index,positions));remaining-=len(positions)
            if not remaining:break
        assert not remaining
        for j,(index,positions) in enumerate(selected):
            if j%world!=rank:continue
            ids=torch.tensor([data[index]['input_ids']],device='cuda');attn=torch.ones_like(ids);pos=torch.tensor(positions,device='cuda')
            with torch.no_grad():q=teacher(input_ids=ids,attention_mask=attn,use_cache=False).logits[0,pos].to('cpu',dtype=torch.float16)
            probe.append((ids,attn,pos,q))
        control.enabled=True
        if rank==0:write(out/'probe_manifest.json',{'seed':100045,'tokens':4096,'positions':selected})
    def probe_eval(step):
        state=torch.cuda.get_rng_state();cpu=torch.get_rng_state();student.eval();value=torch.zeros(2,device='cuda',dtype=torch.float64);start=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for ids,attn,pos,q in probe:
                logits=student(input_ids=ids,attention_mask=attn,use_cache=False).logits[0,pos].float()
                value[0]+=torch.nn.functional.kl_div(logits.log_softmax(-1),q.cuda().float().softmax(-1),reduction='sum');value[1]+=len(pos)
        dist.all_reduce(value);assert int(value[1])==4096
        update=progress.observe(step,(value[0]/value[1]).item());control.multiplier=progress.multiplier
        torch.cuda.set_rng_state(state);torch.set_rng_state(cpu);student.train();torch.cuda.synchronize()
        duration=torch.tensor(time.perf_counter()-start,device='cuda');dist.all_reduce(duration,op=dist.ReduceOp.MAX)
        emit('native_kl_probe',seconds=duration.item(),**update)
    if progress:probe_eval(0)
    # FP32 model (4 bytes/parameter) plus the sharded Adam moments (8 bytes).
    # SAR's router optimizer is replicated across ranks and saved on each rank.
    checkpoint_bytes=sum(p.numel()*12 for p in student.parameters())
    checkpoint_bytes+=sum(b.numel()*b.element_size() for b in student.buffers())
    if ropt:checkpoint_bytes+=sum(p.numel()*(8*world+p.element_size()) for p in policy.router_parameters)
    checkpoint_required_free=math.ceil(checkpoint_bytes*1.10)+2*2**30
    if rank==0:write(out/'storage_budget.json',dict(estimated_checkpoint_bytes=checkpoint_bytes,required_free_bytes=checkpoint_required_free))
    def save(step):
        wait_start=time.time()
        while shutil.disk_usage(root).free<checkpoint_required_free:
            if (root/'archive_error.json').exists():raise RuntimeError('Archive failed; stop before exhausting storage')
            if time.time()-wait_start>1800:raise TimeoutError('Checkpoint archive backpressure exceeded 30 minutes')
            time.sleep(5)
        path=out/f'checkpoint-{step:04d}';path.mkdir(exist_ok=True);dist.barrier()
        torch.save(dict(optimizer=opt.optim.state_dict(),scheduler=sched.state_dict(),router_optimizer=ropt.state_dict() if ropt else None,router_scheduler=rsched.state_dict() if rsched else None,progress=progress.state_dict() if progress else None,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),step=step,rank=rank,world_size=world),path/f'training_state_rank{rank}.pt')
        if rank==0:
            student.save_pretrained(path,safe_serialization=True);tok.save_pretrained(path)
            if ropt:torch.save(policy.router_state(),path/'teacher_router.pt')
            write(path/'metadata.json',dict(step=step,epoch=step/per,method=method,teacher=c['teacher'],config=c))
        dist.barrier()
        if rank==0:write(path/'.ready',{f.name:sha(f) for f in path.iterdir() if f.is_file() and not f.name.startswith('.')})
        dist.barrier();emit('checkpoint_saved',step=step,path=str(path))
    def evaluate(step):
        if c['eval'] not in ('rouge_sampling','rouge_greedy'): raise ValueError('Unsupported evaluation')
        from rouge_score import rouge_scorer
        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
        state=torch.cuda.get_rng_state();cpu=torch.get_rng_state();prng=random.getstate();torch.manual_seed(10+rank);random.seed(10+rank)
        student.eval();records=[];start=time.perf_counter();local=valid[rank::world]
        for offset in range(0,len(local),c['eval_microbatch']):
            rows=local[offset:offset+c['eval_microbatch']];ids,attn=prompts(rows)
            opts=dict(do_sample=True,temperature=1.,top_p=1.,top_k=0) if c['eval']=='rouge_sampling' else dict(do_sample=False)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                seq=student.generate(input_ids=ids,attention_mask=attn,max_new_tokens=R,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,**opts)
            for j,row in enumerate(rows):
                text=tok.decode(seq[j,P:],skip_special_tokens=True).strip();refs=row['output'];refs=[refs] if isinstance(refs,str) else refs
                records.append(dict(source_id=row['source_id'],task=row.get('task'),prediction=text,references=refs,rougeL=max(scorer.score(r,text)['rougeL'].fmeasure for r in refs)))
        dest=out/'eval'/f'step-{step:04d}';write(dest/f'rank{rank}.json',records)
        scores=torch.tensor([sum(r['rougeL'] for r in records),len(records)],device='cuda',dtype=torch.float64);dist.all_reduce(scores)
        assert int(scores[1])==len(valid)
        emit('evaluation',step=step,rougeL=100*(scores[0]/scores[1]).item(),samples=len(valid),seconds=time.perf_counter()-start)
        torch.cuda.set_rng_state(state);torch.set_rng_state(cpu);random.setstate(prng);student.train()
    emit('initialized',total_steps=per*run_epochs,student_updates=per*run_epochs*inner,scheduler_steps=total*inner)
    checkpoints=save_steps(len(data),batch,run_epochs);step=0
    resumed_step=0
    if resume:
        state=torch.load(Path(resume)/f'training_state_rank{rank}.pt',map_location='cpu',weights_only=False)
        assert state['rank']==rank and state['world_size']==world
        opt.optim.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        if method=='sar':
            assert state['router_optimizer'] is not None and state['router_scheduler'] is not None
            policy.load_router_state(torch.load(Path(resume)/'teacher_router.pt',map_location='cpu',weights_only=True))
            ropt.load_state_dict(state['router_optimizer']);rsched.load_state_dict(state['router_scheduler'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);random.setstate(state['python_rng'])
        resumed_step=state['step'];emit('resumed',step=resumed_step,checkpoint=resume)
    for epoch in range(run_epochs):
        order=list(range(len(data)));random.Random(c['seed']+epoch).shuffle(order)
        for start in range(0,len(order),batch):
            step+=1
            if step<=resumed_step:continue
            ix=order[start:start+batch];rows=[data[i] for i in ix[rank::world]]
            assert rows,'Dataset tail must have at least one sample per rank'
            dist.barrier();torch.cuda.synchronize();begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            ids,attn,mask=generate(rows,a.stress)
            router_stats={}
            if ropt:
                router_stats=update_router(policy,student,ropt,ids,attn,mask,len(ix),c['router_microbatch']);rsched.step()
            kid_targets=None
            if method=='kid':
                # One full-rank teacher forward preserves every layer's global candidate budget.
                future=lookup.submit(ids,mask)
                bits,lookup_stats,copy_events,pinned=lookup.finish(future);control.set_batch(bits,mask)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    full_q=teacher(input_ids=ids,attention_mask=attn,use_cache=False).logits[:,:-1]
                kid_targets=full_q.to('cpu');del full_q
                control.bits=None;control.response_mask=None;del bits,pinned,future
            loss_sum=torch.zeros((),device='cuda')
            for augmentation in range(inner):
                opt.zero_grad(set_to_none=True);student.train()
                for off in range(0,len(rows),micro):
                    sl=slice(off,off+micro);n=len(ids[sl]);sync=contextlib.nullcontext() if off+micro>=len(rows) else ddp.no_sync()
                    # no_sync must cover forward AND backward; DDP averages once.
                    with sync:
                        if control and method!='kid':
                            future=lookup.submit(ids[sl],mask[sl])
                            with torch.autocast('cuda',dtype=torch.bfloat16):p=ddp(input_ids=ids[sl],attention_mask=attn[sl],use_cache=False).logits[:,:-1]
                            bits,lookup_stats,copy_events,pinned=lookup.finish(future);control.set_batch(bits,mask[sl])
                        else:
                            if policy:policy.set_batch(attn[sl])
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            q=kid_targets[sl].to('cuda') if method=='kid' else teacher(input_ids=ids[sl],attention_mask=attn[sl],use_cache=False).logits[:,:-1]
                        if not control or method=='kid':
                            with torch.autocast('cuda',dtype=torch.bfloat16):p=ddp(input_ids=ids[sl],attention_mask=attn[sl],use_cache=False).logits[:,:-1]
                        loss=sequence_kl(p,q,mask[sl]);assert torch.isfinite(loss)
                        (loss*weighted_scale(n,len(ix),world)).backward();loss_sum+=loss.detach()*n/inner
                        del p,q,loss
                        if control and method!='kid':control.bits=None;control.response_mask=None;del bits,pinned,future
                norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.);assert torch.isfinite(norm)
                opt.step();sched.step()
            del kid_targets
            counts=mask.sum(-1);last=ids.gather(1,(attn.sum(-1)-1)[:,None]).squeeze(1)
            statistics=torch.stack([loss_sum,counts.sum(),((counts==R)&(last!=tok.eos_token_id)).sum()]).double();dist.all_reduce(statistics)
            torch.cuda.synchronize();perf=torch.tensor([time.perf_counter()-begin,torch.cuda.max_memory_reserved()/2**30],device='cuda');dist.all_reduce(perf,op=dist.ReduceOp.MAX)
            emit('train',step=step,student_optimizer_step=step*inner,epoch=epoch+(start+len(ix))/len(data),seconds=perf[0].item(),peak_reserved_GiB=perf[1].item(),loss=statistics[0].item()/len(ix),response_tokens=int(statistics[1]),cap_without_eos_fraction=statistics[2].item()/len(ix),batch_samples=len(ix),progress_multiplier=progress.multiplier if progress else None,**router_stats)
            del ids,attn,mask,counts
            if progress and step%c['probe_interval']==0:probe_eval(step)
            if a.steps and step>=a.steps:
                if rank==0:write(out/'summary.json',dict(completed=True,warmup_steps=1,steps=step))
                dist.destroy_process_group();return
            if step in checkpoints:save(step);evaluate(step)
    if rank==0:write(out/'complete.json',dict(steps=step,epochs=run_epochs,scheduler_epochs=c['epochs'],checkpoint_count=len(checkpoints)))
    dist.destroy_process_group()
if __name__=='__main__':main()
