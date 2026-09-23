"""Two GPU KA/SAR plus Dolly TAID. Explicit preflight/train/resume only."""
import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from common import read, write, sha, schedule, normalize, collate, parameter_groups, verify_checkpoint
from taid import Controller, loss_sum
from expert_methods import ExpertMethod
from sar_update import update_router
from history_policy import HistoricalPolicy, FixedProbe, logical_rows, response_mask

ROOT=Path(__file__).resolve().parent


def load(path, dtype):
    cfg=AutoConfig.from_pretrained(path,trust_remote_code=True)
    if cfg.model_type=='llama_moe':
        cfg.rope_scaling=None
        cfg._attn_implementation='eager'
        assert cfg.num_selects==2, 'This experiment only permits native Top-2'
    else:
        cfg._attn_implementation='sdpa'
    model=AutoModelForCausalLM.from_pretrained(path,config=cfg,trust_remote_code=True,
                                             torch_dtype=dtype,low_cpu_mem_usage=True)
    model.config.use_cache=False
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(ROOT/'config.json'))
    ap.add_argument('--mode',choices=['preflight','train'],required=True)
    ap.add_argument('--stress',action='store_true')
    ap.add_argument('--resume')
    a=ap.parse_args();c=read(a.config)
    method=c['method'];assert method in ('top2','top8','strawman','kid','ka','sar')
    inner=2 if method=='ka' else 1
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==c['world_size']==2 and c['global_batch']==384
    assert c['epochs'] in (5,10) and c['lr']==1e-4
    job_root=Path(c['job_root']);job_root.mkdir(parents=True,exist_ok=True)
    assert not (a.stress and a.mode=='train')
    assert not (a.resume and a.mode!='train')
    local_batch=c['global_batch']//world;micro=c['microbatch']
    assert local_batch%micro==0
    torch.cuda.set_device(rank);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=45))
    random.seed(c['seed']+rank);torch.manual_seed(c['seed']+rank)
    torch.backends.cuda.matmul.allow_tf32=True
    os.chdir(ROOT)
    # Fixed source assets; includes teacher identity and exact raw data hashes.
    assets=read(c['assets_manifest'])
    for key in ['train','valid']:
        assert sha(c[key])==assets['data'][key],key
    signature={'config':sha(a.config),'assets':sha(c['assets_manifest']),
               'code':{p.name:sha(p) for p in ROOT.glob('*.py')}}
    out=job_root/(('preflight_stress' if a.stress else 'preflight_real') if a.mode=='preflight' else 'run')
    if rank==0:
        if a.mode=='train':
            for name in ['preflight_real','preflight_stress']:
                pre=read(job_root/name/'summary.json')
                assert pre['signature']==signature and pre['completed'], 'Preflight stale or missing'
        if a.resume:
            assert out.is_dir() and not (out/'complete.json').exists()
            complete_paths=sorted(p for p in out.glob('checkpoint-*') if (p/'.ready.json').exists())
            assert complete_paths and Path(a.resume).resolve()==complete_paths[-1].resolve(), 'Resume latest complete checkpoint only'
            checkpoint_step=int(complete_paths[-1].name.split('-')[-1])
            for partial in out.glob('checkpoint-*'):
                if not (partial/'.ready.json').exists():
                    assert not partial.is_symlink() and partial.resolve().parent==out.resolve()
                    partial.rename(out/('failed_partial_'+partial.name+'_'+str(time.time_ns())))
            events=out/'events.jsonl'
            if events.exists():
                shutil.copy2(events,out/f'events_before_resume_{time.time_ns()}.jsonl')
                lines=[line for line in events.read_text().splitlines() if json.loads(line).get('step',0)<=checkpoint_step]
                events.write_text('\n'.join(lines)+'\n')
        else:
            assert not out.exists(), 'Use a new run directory or explicit --resume'
            out.mkdir(parents=True)
        write(out/'signature.json',signature)
    dist.barrier()
    def emit(kind,**kw):
        if rank==0:
            event=dict(kind=kind,time=time.time(),**kw)
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            write(out/'status.json',event);print(json.dumps(event),flush=True)
    tok=AutoTokenizer.from_pretrained(c['student'],use_fast=False);tok.pad_token=tok.eos_token
    tt=AutoTokenizer.from_pretrained(c['teacher'],use_fast=False,trust_remote_code=True)
    assert tok.get_vocab()==tt.get_vocab() and tok.eos_token_id==tt.eos_token_id
    data,cut=normalize(read(c['train']),c['prompt_cap'],c['response_cap'],tok.eos_token_id)
    valid,_=normalize(read(c['valid']),c['prompt_cap'],c['response_cap'],tok.eos_token_id)
    assert not {r['source_id'] for r in data}&{r['source_id'] for r in valid}
    per,saves=schedule(len(data),c['global_batch'],c['epochs']);total=per*c['epochs']
    assert len(saves)==2*c['epochs'] and len(valid)==c['valid_count']
    scheduler_total=per*c['scheduler_epochs']
    teacher=load(c['teacher'],torch.bfloat16).cuda().eval().requires_grad_(False)
    # Inspect runtime gates, not just the model config.
    gates=[m for m in teacher.modules() if hasattr(m,'num_selects')]
    assert gates and all(m.num_selects==2 for m in gates)
    policy=HistoricalPolicy(teacher,c['history'],tok.all_special_ids,kid=method=='kid') if method in ('strawman','kid') else ExpertMethod(teacher,method)
    resume_path=Path(a.resume).resolve() if a.resume else None
    if resume_path:
        assert resume_path.parent==out.resolve()
        if rank==0:verify_checkpoint(resume_path)
        dist.barrier()
    # FP32 master weights/moments; BF16 autocast forward. DDP replaces official
    # DeepSpeed ZeRO-2 storage, preserving master-weight optimizer precision.
    student=load(str(resume_path or c['student']),torch.float32).cuda()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ddp=DDP(student,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
    opt=torch.optim.AdamW(parameter_groups(student),lr=c['lr'],weight_decay=c['weight_decay'],foreach=False)
    sched=get_scheduler('cosine',opt,num_warmup_steps=0,num_training_steps=scheduler_total*inner)
    controller=Controller(t_start=c['t_start'],t_end=c['t_end'],alpha=c['alpha'],beta=c['beta'],t=c['t_start'])
    ropt=torch.optim.AdamW(policy.router_parameters,lr=c['router_lr'],weight_decay=0,foreach=False) if method=='sar' else None
    rsched=get_scheduler('cosine',ropt,num_warmup_steps=0,num_training_steps=scheduler_total) if ropt else None
    completed=0;pure_seconds=0.
    # Latest-only local retention; no checkpoint uploads during training.
    archiver=None
    # Standalone release retains checkpoints locally; no remote credential required.
    latest=resume_path
    if resume_path:
        state=torch.load(resume_path/'optimizer.pt',map_location='cpu',weights_only=False)
        assert state['signature']==signature and state['total_steps']==total
        opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        if ropt:
            policy.load_router_state(state['router'])
            ropt.load_state_dict(state['router_optimizer']);rsched.load_state_dict(state['router_scheduler'])
        controller=Controller(**state['controller'])
        if isinstance(policy,HistoricalPolicy):policy.load_state_dict(state['progress'])
        completed=state['step'];pure_seconds=state['pure_seconds']
        rng=torch.load(resume_path/f'rng_rank{rank}.pt',map_location='cpu',weights_only=False)
        assert rng['step']==completed and rng['world']==world
        random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
        emit('resumed',step=completed,checkpoint=str(resume_path))
    emit('initialized',steps_per_epoch=per,total_steps=total,checkpoints=saves,
         rows=len(data),dropped_per_epoch=len(data)%c['global_batch'],response_truncated=cut,
         method=method,student_updates=total*inner,microbatch=micro,accumulation=local_batch//micro,global_batch=c['global_batch'],runtime_gates=len(gates))

    def evaluate(step, evaluation_rows=None, split='validation'):
        eval_rows=valid if evaluation_rows is None else evaluation_rows
        from rouge_score import rouge_scorer
        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
        cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state();py=random.getstate()
        student.eval();start=time.perf_counter();records=[]
        # Retain six logical evaluation shards and their seeds, even on two GPUs.
        # This preserves old Dolly sample grouping and batch4 sampling protocol.
        for shard in range(rank,6,world):
            torch.manual_seed(c['eval_seed']+shard);random.seed(c['eval_seed']+shard)
            rows=eval_rows[shard::6]
            for offset in range(0,len(rows),c['eval_microbatch']):
                part=rows[offset:offset+c['eval_microbatch']];P=c['prompt_cap']
                ids=torch.full((len(part),P),tok.eos_token_id,device='cuda',dtype=torch.long)
                attn=torch.zeros_like(ids)
                for j,r in enumerate(part):
                    p=r['input_ids'][:r['prompt_len']];ids[j,-len(p):]=torch.tensor(p,device='cuda');attn[j,-len(p):]=1
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    seq=student.generate(input_ids=ids,attention_mask=attn,max_new_tokens=c['response_cap'],
                        do_sample=c['eval_sampling'],use_cache=True,
                        **(dict(temperature=1.,top_p=1.,top_k=0) if c['eval_sampling'] else {}),
                        pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
                for j,r in enumerate(part):
                    pred=tok.decode(seq[j,P:],skip_special_tokens=True).strip()
                    refs=r['output'];refs=[refs] if isinstance(refs,str) else refs
                    records.append(dict(source_id=r['source_id'],prediction=pred,references=refs,
                                        rougeL=max(scorer.score(ref,pred)['rougeL'].fmeasure for ref in refs)))
        write(out/('evaluation' if split=='validation' else 'test_'+split)/f'step{step:04d}_rank{rank}.json',records)
        gathered=[None]*world;dist.all_gather_object(gathered,records)
        if rank==0:
            all_rows=sum(gathered,[])
            assert {r['source_id'] for r in all_rows}=={r['source_id'] for r in eval_rows} and len(all_rows)==len(eval_rows)
            write(out/('evaluation' if split=='validation' else 'test_'+split)/f'step{step:04d}.json',dict(step=step,rougeL=100*sum(r['rougeL'] for r in all_rows)/len(all_rows),
                  pure_train_seconds=pure_seconds,samples=len(all_rows),seconds=time.perf_counter()-start))
        dist.barrier();torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);random.setstate(py);student.train()
        emit('evaluation_complete',step=step,seconds=time.perf_counter()-start)

    def save(step):
        nonlocal latest
        dist.barrier();start=time.perf_counter();dest=out/f'checkpoint-{step:04d}'
        if rank==0:
            need=sum(p.numel()*p.element_size() for p in student.parameters())*3*1.1+2*2**30
            assert shutil.disk_usage(out).free>need, 'Insufficient save headroom; preserve latest checkpoint and stop'
            assert not dest.exists();dest.mkdir()
            student.save_pretrained(dest,safe_serialization=True);tok.save_pretrained(dest)
            torch.save(dict(step=step,total_steps=total,signature=signature,optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),controller=controller.state_dict(),pure_seconds=pure_seconds,
                progress=policy.state_dict() if isinstance(policy,HistoricalPolicy) else None,
                router=policy.router_state() if ropt else None,router_optimizer=ropt.state_dict() if ropt else None,router_scheduler=rsched.state_dict() if ropt else None),dest/'optimizer.pt')
        dist.barrier()
        torch.save(dict(step=step,world=world,python=random.getstate(),cpu=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state()),dest/f'rng_rank{rank}.pt')
        dist.barrier()
        if rank==0:
            write(dest/'training_summary.json',dict(step=step,pure_train_seconds=pure_seconds))
            write(dest/'.ready.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
            verify_checkpoint(dest)
            # Keep all model snapshots; only the newest retains recovery state.
            for old in sorted(out.glob('checkpoint-*')):
                if old==dest:continue
                assert old.is_dir() and not old.is_symlink() and old.resolve().parent==out.resolve()
                manifest=read(old/'.ready.json')
                states=[p for p in old.iterdir() if p.name=='optimizer.pt' or (p.name.startswith('rng_rank') and p.suffix=='.pt')]
                kept={n:h for n,h in manifest.items() if n not in {p.name for p in states}}
                assert any(n.endswith('.safetensors') for n in kept)
                write(old/'.ready.json',kept)
                for path in states:path.unlink()
                if states:
                    with (out/'cleanup.jsonl').open('a') as f:
                        f.write(json.dumps(dict(checkpoint=str(old),removed=[p.name for p in states],recovery=str(dest),time=time.time()))+'\n')
        latest=dest
        dist.barrier();emit('checkpoint',step=step,seconds=time.perf_counter()-start)

    def retain_best_and_latest(step):
        if rank==0:
            scores=[read(p) for p in (out/'evaluation').glob('step*.json') if '_rank' not in p.name]
            best=min(scores,key=lambda e:(-e['rougeL'],e['step']))
            best_path=out/f"checkpoint-{best['step']:04d}"
            verify_checkpoint(best_path)
            write(out/'best.json',dict(step=best['step'],rougeL=best['rougeL'],path=str(best_path)))
            for old in out.glob('checkpoint-*'):
                if old in (best_path,latest):continue
                assert old.resolve().parent==out.resolve() and not old.is_symlink()
                verify_checkpoint(old)
                shutil.rmtree(old)
                emit('checkpoint_removed',step=step,path=str(old),reason='neither_best_nor_latest')
        dist.barrier()

    probe=None
    if method=='kid':
        before=(torch.get_rng_state(),torch.cuda.get_rng_state(),random.getstate())
        probe_start=time.perf_counter()
        probe=FixedProbe(teacher,policy,data,tok.all_special_ids,rank,world)
        if rank==0:write(out/'probe_manifest.json',dict(seed=100045,positions=probe.selected,source='fixed training gold-response positions',tokens=4096))
        torch.set_rng_state(before[0]);torch.cuda.set_rng_state(before[1]);random.setstate(before[2])
        emit('probe_setup',seconds=time.perf_counter()-probe_start)
    def probe_eval(step):
        before=(torch.get_rng_state(),torch.cuda.get_rng_state(),random.getstate());start=time.perf_counter()
        student.eval();value=probe.score(student);dist.all_reduce(value);assert int(value[1])==4096
        update=policy.progress.observe(step,float(value[0]/value[1]));policy.control.multiplier=policy.progress.multiplier
        torch.set_rng_state(before[0]);torch.cuda.set_rng_state(before[1]);random.setstate(before[2]);student.train()
        torch.cuda.synchronize();emit('native_kl_probe',seconds=time.perf_counter()-start,**update)
    if method=='kid' and not resume_path:probe_eval(0)
    bench=[];student.train()
    if resume_path and completed in saves:
        if not (out/'evaluation'/f'step{completed:04d}.json').exists():evaluate(completed)
        retain_best_and_latest(completed)
    for epoch in range(c['epochs']):
        order=list(range(len(data)));random.Random(c['seed']+epoch).shuffle(order)
        for b in range(per):
            step=epoch*per+b+1
            if step<=completed:continue
            indices=order[b*c['global_batch']:(b+1)*c['global_batch']]
            global_rows=[data[i] for i in indices]
            rows=logical_rows(global_rows,rank) if method=='kid' else global_rows[rank::world]
            torch.cuda.synchronize();dist.barrier();begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            tokens=sum((c['prompt_cap']+c['response_cap']-1 if a.stress else len(r['input_ids'])-1) for r in rows)
            count=torch.tensor(tokens,device='cuda',dtype=torch.float64);dist.all_reduce(count)
            router_stats={}
            if ropt:
                ids,attn,mask=collate(rows,tok.eos_token_id,'cuda',a.stress,c['prompt_cap'],c['response_cap'])
                router_stats=update_router(policy,student,ropt,ids,attn,mask,int(count),c['router_microbatch'])
                rsched.step();del ids,attn,mask
            kid_targets=None
            activation_hist=torch.zeros(9,device='cuda',dtype=torch.int64) if method=='kid' else None
            activation_extra=torch.zeros(2,device='cuda',dtype=torch.int64) if method=='kid' else None
            activation_multiplier=policy.control.multiplier if method=='kid' else None
            if method=='kid':
                # Retain six logical rank-64 selections, assigned three to each physical GPU.
                length=max(len(r['input_ids']) for r in rows) if not a.stress else c['prompt_cap']+c['response_cap']
                kid_targets=torch.zeros((len(rows),length-1,teacher.config.vocab_size),dtype=torch.bfloat16)
                for off in range(0,len(rows),64):
                    part=rows[off:off+64]
                    ids,attn,_=collate(part,tok.eos_token_id,'cuda',a.stress,c['prompt_cap'],c['response_cap'])
                    rm=(attn[:,1:].bool() if a.stress else response_mask(part,ids.shape[1],ids.device))
                    policy.set_tokens(ids,rm)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        hidden=teacher.model(input_ids=ids,attention_mask=attn,use_cache=False).last_hidden_state
                        for j in range(0,len(part),micro):
                            q=teacher.lm_head(hidden[j:j+micro,:-1])
                            kid_targets[off+j:off+j+len(q),:q.shape[1]]=q.cpu()
                    activation_hist.add_(policy.control.k_hist)
                    activation_extra.add_(policy.control.extra_counts)
                    del hidden,q;policy.clear()
            update_records=[]
            for augmentation in range(inner):
                opt.zero_grad(set_to_none=True);loss_total=torch.zeros((),device='cuda',dtype=torch.float64)
                t_used=controller.t
                for start in range(0,local_batch,micro):
                    part=rows[start:start+micro]
                    ids,attn,mask=collate(part,tok.eos_token_id,'cuda',a.stress,c['prompt_cap'],c['response_cap'])
                    if isinstance(policy,HistoricalPolicy):
                        if method=='strawman':policy.set_tokens(ids,response_mask(part,ids.shape[1],ids.device))
                    else:policy.set_batch(attn)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        if method=='kid':
                            target=kid_targets[start:start+len(part),:ids.shape[1]-1].to('cuda')
                        else:target=teacher(input_ids=ids,attention_mask=attn,use_cache=False).logits[:,:-1]
                    if isinstance(policy,HistoricalPolicy):policy.clear()
                    sync=ddp.no_sync() if start+micro<local_batch else contextlib.nullcontext()
                    with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=ddp(input_ids=ids,attention_mask=attn,use_cache=False).logits[:,:-1]
                        value=loss_sum(logits,target,mask,t_used)
                        loss_total+=value.detach().double()
                        (value*world/count).backward()
                    del logits,target,value
                dist.all_reduce(loss_total)
                loss=float(loss_total/count)
                norm=torch.nn.utils.clip_grad_norm_(student.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                assert math.isfinite(loss)
                lr=opt.param_groups[0]['lr'];opt.step();sched.step()
                # Match official Lightning global_step index (zero-based during forward).
                controller.update(loss,(step-1)*inner+augmentation,scheduler_total*inner)
                update_records.append(dict(loss=loss,t_used=t_used,t_next=controller.t,student_update=(step-1)*inner+augmentation+1))
            del kid_targets
            torch.cuda.synchronize();elapsed=torch.tensor(time.perf_counter()-begin,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            seconds=float(elapsed);pure_seconds+=seconds
            peak=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device='cuda');dist.all_reduce(peak,op=dist.ReduceOp.MAX)
            record=dict(step=step,student_optimizer_step=step*inner,updates=update_records,router_stats=router_stats,loss=loss,t_used=t_used,t_next=controller.t,lr=lr,grad_norm=float(norm),
                        seconds=seconds,pure_train_seconds=pure_seconds,peak_GiB=float(peak),tokens=int(count))
            emit('train',**record)
            if method=='kid':
                stat_start=time.perf_counter()
                dist.all_reduce(activation_hist);dist.all_reduce(activation_extra)
                h=activation_hist.tolist();n=sum(h)
                assert n>0 and sum(h[:2])==0
                emit('expert_activation',step=step,epoch=epoch+1,stage_epoch=epoch+1,
                     progress_multiplier=activation_multiplier,taid_t=t_used,
                     expert_token_layer_counts=h,response_token_layer_count=n,
                     mean_active_experts=sum(k*v for k,v in enumerate(h))/n,
                     extra_candidate_pairs=int(activation_extra[0]),extra_selected_pairs=int(activation_extra[1]),
                     statistics_seconds=time.perf_counter()-stat_start,
                     scope='response target non-special token x layer; pooled six logical rank64 batches')
            if a.mode=='preflight':
                bench.append(record)
                if len(bench)>=(2 if a.stress else 3):
                    if rank==0:write(out/'summary.json',dict(completed=True,signature=signature,records=bench,
                        seconds_excluding_warmup=sum(x['seconds'] for x in bench[1:])/(len(bench)-1)))
                    dist.destroy_process_group();return
            else:
                if method=='kid' and step%10==0:probe_eval(step)
                if step in saves:
                    save(step);evaluate(step);retain_best_and_latest(step)
    for split,path in c.get('tests',{}).items():
        test_rows,_=normalize(read(path),c['prompt_cap'],c['response_cap'],tok.eos_token_id)
        evaluate(total,test_rows,split=split)
    if rank==0:
        import subprocess,sys
        write(out/'complete.json' ,dict(config_sha256=sha(a.config),dataset=c['dataset'],method=method,steps=total,epochs=c['epochs'],scheduler_epochs=c['scheduler_epochs'],checkpoints=saves,pure_train_seconds=pure_seconds,validation_complete=True,tests=list(c.get('tests',{})),archive_pending=False))
        write(out/'archive_status.json',dict(stage='local_only',time=time.time()))
    dist.destroy_process_group()


if __name__=='__main__':main()
