import os,json,time,math,random,argparse,shutil,hashlib,gc,string
from pathlib import Path
from functools import partial
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, FullStateDictConfig, FullOptimStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
from rouge_score import rouge_scorer
ROOT=Path(os.environ['CITB_WORK_DIR']).resolve();MODEL=Path(os.environ['TEACHER_MODEL']).resolve()

def collate(rows,pad):
    pairs=[(r['prompt'],random.choice(r['targets'])) for r in rows]
    n=max(len(p)+len(t)-1 for p,t in pairs)
    ids=torch.full((len(rows),n),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
    for j,(p,t) in enumerate(pairs):
        seq=p+t;length=len(seq)-1
        ids[j,:length]=torch.tensor(seq[:-1],device='cuda');mask[j,:length]=1
        labels[j,:length]=torch.tensor(seq[1:],device='cuda');labels[j,:len(p)-1]=-100
    return ids,mask,labels

def metrics(records):
    scorer=rouge_scorer.RougeScorer(['rouge1','rougeL'],use_stemmer=True)
    def norm(s):return ' '.join(''.join(c for c in s.lower() if c not in string.punctuation).split())
    vals=[];groups={}
    for r in records:
        scores=[scorer.score(ref,r['prediction']) for ref in r['references']]
        v={'exact_match':float(any(norm(r['prediction'])==norm(ref) for ref in r['references'])),**{k:max(s[k].fmeasure for s in scores) for k in ['rouge1','rougeL']}}
        vals.append(v)
        for g in ['task:'+r['task'],*['category:'+c for c in r['categories']]]:groups.setdefault(g,[]).append(v)
    def avg(xs):return {k:round(100*sum(v[k] for v in xs)/len(xs),4) for k in ['exact_match','rouge1','rougeL']}
    result={'overall':avg(vals),'groups':{k:avg(v) for k,v in groups.items()},'instances':len(records)}
    tasks=[v for k,v in result['groups'].items() if k.startswith('task:')]
    result['task_macro']={k:sum(v[k] for v in tasks)/len(tasks) for k in tasks[0]}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--resume');p.add_argument('--resume-test',action='store_true');a=p.parse_args()
    dist.init_process_group('nccl',timeout=timedelta(hours=3));rank=dist.get_rank();world=dist.get_world_size();assert world==2
    torch.cuda.set_device(rank);torch.set_num_threads(8)
    random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
    out=ROOT/('smoke' if a.smoke else 'run');out.mkdir(exist_ok=True)
    def emit(x):
        if rank==0:
            x['time']=time.time()
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
            (out/'status.json').write_text(json.dumps(x));print(json.dumps(x),flush=True)
    data={s:json.loads((ROOT/'data'/f'{s}_tokens.json').read_text()) for s in ['train','dev']}
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False)
    cfg=AutoConfig.from_pretrained(MODEL,trust_remote_code=True);cfg._attn_implementation='eager';cfg.rope_scaling=None
    source=Path(a.resume) if a.resume else MODEL
    raw=AutoModelForCausalLM.from_pretrained(source,config=cfg,trust_remote_code=True,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    assert raw.config.num_selects==2 and all(p.requires_grad for p in raw.parameters())
    raw.config.use_cache=False;raw.model.gradient_checkpointing=True
    engine=FSDP(raw,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={type(raw.model.layers[0])}),mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16),sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,device_id=rank,sync_module_states=True)
    optimizer=torch.optim.AdamW(engine.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    sampler=DistributedSampler(data['train'],num_replicas=world,rank=rank,shuffle=True,seed=42,drop_last=False)
    steps_epoch=math.ceil(len(sampler)/8)
    config={'setting':'CITB long-stream MULTI_TASK with decoder-only MoE adaptation','seed':42,'epochs_max':15,'learning_rate':5e-5,'scheduler':'constant','warmup':0,'weight_decay':0,'micro':8,'world_size':2,'global_batch':16,'accum':1,'steps_per_epoch':steps_epoch,'total_steps_max':15*steps_epoch,'eval_steps':500,'save_steps':500,'patience_evaluations':3,'max_prompt':1024,'max_response_including_eos':128,'positive_examples':2,'loss':'target-token CE only, matching official SFT; native MoE balance statistic logged but not added','routing':'native Top-2, all parameters trainable','eval':'greedy, max_new_tokens 128, multi-reference EM/ROUGE1/ROUGEL, micro 32','selection':'max overall dev rougeL; baseline reported separately','checkpoint':'FP32 model + full Adam/scheduler/RNG/cursor states; SHA256 archival; best/latest retention','data':json.loads((ROOT/'data/tokenization_manifest.json').read_text())}
    config['checkpoint']='Best validation ROUGE-L model weights only; temporary candidate for evaluation; no Adam/RNG recovery checkpoints or archival'
    config['archive_enabled']=False
    if rank==0:(out/'config.json').write_text(json.dumps(config,indent=2))
    step=0;start_epoch=0;start_offset=0;best=-float('inf');best_path=None;bad=0;baseline=None
    if a.resume:
        state=torch.load(Path(a.resume)/'trainer_state.pt',map_location='cpu',weights_only=False)
        fullopt=torch.load(Path(a.resume)/'optimizer.pt',map_location='cpu',mmap=True,weights_only=False) if rank==0 else None
        optimizer.load_state_dict(FSDP.scatter_full_optim_state_dict(fullopt,engine));del fullopt
        scheduler.load_state_dict(state['scheduler']);step=state['step'];start_epoch=state['next_epoch'];start_offset=state['next_offset'];best=state['best'];best_path=state['best_path'];bad=state['bad'];baseline=state['baseline']
        rng=state['rng'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
        emit({'kind':'resumed','step':step,'next_epoch':start_epoch,'next_offset':start_offset,'optimizer_state_count':len(optimizer.state),'lr':optimizer.param_groups[0]['lr']})
        assert len(optimizer.state)>0 and optimizer.param_groups[0]['lr']==5e-5
        # Finish restoration before releasing the one-time switch snapshot's Adam state.
        gc.collect();torch.cuda.synchronize();dist.barrier()
        if rank==0:
            optfile=Path(a.resume)/'optimizer.pt'
            assert Path(a.resume).resolve().parent==out.resolve()
            if optfile.exists():optfile.unlink()
            for old in out.glob('checkpoint-*'):
                if old.name!=best_path:
                    assert old.resolve().parent==out.resolve();shutil.rmtree(old)
            retained=out/best_path
            for filename in ['optimizer.pt','trainer_state.pt']:
                if (retained/filename).exists():(retained/filename).unlink()
            hashes=json.loads((retained/'sha256.json').read_text())
            hashes={k:v for k,v in hashes.items() if (retained/k).exists()}
            (retained/'sha256.json').write_text(json.dumps(hashes,indent=2))
            (retained/'.ready').write_text('best model weights only, no archival\n')
            selection={'best_checkpoint':best_path,'best_rougeL':best,'last_evaluated_step':step,'baseline_rougeL':baseline,'bad_evaluations':bad,'step':step,'retention':'best_only','archive_enabled':False}
            tmp=out/'selection.json.tmp';tmp.write_text(json.dumps(selection));tmp.replace(out/'selection.json')
        dist.barrier()
    emit({'kind':'initialized','step':step,'config':{k:v for k,v in config.items() if k!='data'}})

    def evaluate(model_path,split,label,limit=None):
        optimizer.zero_grad(set_to_none=True);engine.eval();gc.collect();torch.cuda.empty_cache()
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
        rows=data.get(split)
        if rows is None:rows=json.loads((ROOT/'data'/f'{split}_tokens.json').read_text())
        if limit:rows=rows[:limit]
        subset=rows[rank::world]
        evalmodel=AutoModelForCausalLM.from_pretrained(model_path,config=cfg,trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda').eval()
        evalmodel.config.use_cache=True;evalmodel.model.gradient_checkpointing=False
        records=[];begin=time.time()
        with torch.no_grad():
            for off in range(0,len(subset),32):
                batch=subset[off:off+32];width=max(len(r['prompt']) for r in batch)
                ids=torch.full((len(batch),width),tok.eos_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
                for j,r in enumerate(batch):ids[j,-len(r['prompt']):]=torch.tensor(r['prompt'],device='cuda');mask[j,-len(r['prompt']):]=1
                generated=evalmodel.generate(input_ids=ids,attention_mask=mask,do_sample=False,num_beams=1,max_new_tokens=128,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,use_cache=True)
                predictions=tok.batch_decode(generated[:,width:],skip_special_tokens=True)
                records.extend({k:r[k] for k in ['sample_id','task','categories','references']}|{'prediction':pred} for r,pred in zip(batch,predictions))
                del ids,mask,generated
                if off%320==0:print(json.dumps({'kind':'generation_progress','rank':rank,'label':label,'done':min(off+32,len(subset)),'total':len(subset)}),flush=True)
        del evalmodel;gc.collect();torch.cuda.empty_cache()
        shards=[None]*world;dist.all_gather_object(shards,records)
        result=None
        if rank==0:
            allrows=[r for part in shards for r in part];assert len(allrows)==len(rows)
            result=metrics(allrows)
            (out/f'{label}_metrics.json').write_text(json.dumps(result,indent=2))
            with (out/f'{label}_predictions.jsonl').open('w') as f:
                for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        obj=[result];dist.broadcast_object_list(obj,src=0);result=obj[0]
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state(rng[3]);engine.train()
        emit({'kind':'evaluation','label':label,'split':split,'step':step,'seconds':time.time()-begin,**result['overall'],'instances':len(rows)})
        return result['overall']['rougeL']

    def save(next_epoch,next_offset):
        path=out/f'checkpoint-{step}'
        if rank==0:
            assert shutil.disk_usage(out).free>29_000_000_000,'Need space for one temporary evaluation candidate'
            path.mkdir(exist_ok=False)
        dist.barrier()
        with FSDP.state_dict_type(engine,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            modelstate=engine.state_dict()
        if rank==0:
            assert all(torch.isfinite(v).all() for v in modelstate.values() if v.is_floating_point())
            torch.save(modelstate,path/'pytorch_model.bin')
            for f in MODEL.iterdir():
                if f.suffix in ('.json','.py','.model') and f.name!='pytorch_model.bin.index.json':shutil.copy2(f,path/f.name)
            engine.module.config.save_pretrained(path)
        del modelstate;gc.collect();dist.barrier()
        return path

    def publish(path,next_epoch,next_offset):
        if rank==0:
            if path.name==best_path:
                hashes={}
                for f in path.iterdir():
                    h=hashlib.sha256()
                    with f.open('rb') as stream:
                        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
                    hashes[f.name]=h.hexdigest()
                (path/'sha256.json').write_text(json.dumps(hashes,indent=2));(path/'.ready').write_text('best model weights only, no archival\n')
            selection={'best_checkpoint':best_path,'best_rougeL':best,'last_evaluated_step':step,'baseline_rougeL':baseline,'bad_evaluations':bad,'step':step,'retention':'best_only','archive_enabled':False}
            tmp=out/'selection.json.tmp';tmp.write_text(json.dumps(selection));tmp.replace(out/'selection.json')
            for old in out.glob('checkpoint-*'):
                if old.name!=best_path:
                    assert old.resolve().parent==out.resolve();shutil.rmtree(old)
        dist.barrier()

    if baseline is None:baseline=evaluate(MODEL,'dev','baseline',64 if a.smoke else None)
    # from_pretrained defaults to eval; resume bypasses baseline's train-mode restoration.
    engine.train();gc.collect();torch.cuda.empty_cache()
    stop=False
    for epoch in range(start_epoch,15):
        sampler.set_epoch(epoch);indices=list(iter(sampler))
        if a.smoke:
            longest=sorted(range(len(data['train'])),key=lambda i:len(data['train'][i]['prompt'])+max(map(len,data['train'][i]['targets'])),reverse=True)[:16]
            indices=longest[rank::world]+indices
        for off in range(start_offset if epoch==start_epoch else 0,len(indices),8):
            rows=[data['train'][i] for i in indices[off:off+8]];begin=time.time();optimizer.zero_grad(set_to_none=True)
            ids,mask,labels=collate(rows,tok.eos_token_id)
            pred=engine(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
            loss=F.cross_entropy(pred.logits.float().flatten(0,1),labels.flatten(),ignore_index=-100)
            assert torch.isfinite(loss);loss.backward();norm=engine.clip_grad_norm_(1.);assert torch.isfinite(norm)
            optimizer.step();scheduler.step();step+=1;torch.cuda.synchronize()
            vals=torch.tensor([loss.item(),float(pred.balance_loss) if pred.balance_loss is not None else 0.,torch.cuda.max_memory_allocated()/2**30],device='cuda');dist.all_reduce(vals);vals/=world
            emit({'kind':'train','step':step,'epoch':epoch+min(off+8,len(indices))/len(indices),'loss':vals[0].item(),'native_balance_statistic':vals[1].item(),'mean_peak_GiB':vals[2].item(),'lr':optimizer.param_groups[0]['lr'],'grad_norm':float(norm),'seconds':time.time()-begin})
            del ids,mask,labels,pred,loss;optimizer.zero_grad(set_to_none=True)
            next_epoch=epoch+(off+8>=len(indices));next_offset=0 if next_epoch>epoch else off+8
            if a.resume_test:
                if rank==0:(out/'resume_verified.json').write_text(json.dumps({'resumed_from':a.resume,'step_after_update':step,'finite_update':True}))
                dist.destroy_process_group();return
            if step%500==0 or (a.smoke and step==3):
                path=save(next_epoch,next_offset)
                score=evaluate(path,'dev',f'step{step}',64 if a.smoke else None)
                if score>best:best=score;best_path=path.name;bad=0
                else:bad+=1
                publish(path,next_epoch,next_offset)
                if a.smoke:
                    if rank==0:(out/'complete.json').write_text(json.dumps({'step':step,'checkpoint':path.name,'rougeL':score}))
                    dist.destroy_process_group();return
                if bad>=3:stop=True;break
        start_offset=0
        if stop:break
    # Official selection uses periodic validation checkpoints; final partial interval does not replace best.
    assert best_path is not None
    for split in ['initial_test','cl_test','official_test']:evaluate(out/best_path,split,'best_'+split)
    if rank==0:(out/'complete.json').write_text(json.dumps({'steps':step,'best_checkpoint':best_path,'best_rougeL':best,'baseline_rougeL':baseline,'early_stopped':stop}))
    dist.destroy_process_group()
if __name__=='__main__':main()
