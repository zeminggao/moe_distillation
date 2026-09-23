"""Frozen-teacher historical M80 union native Top-2; genuinely ragged dispatch."""
import time
import types
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

class HistoryLookup:
    def __init__(self, path):
        self.table = np.load(path)
        assert self.table.dtype == np.uint8 and self.table.ndim == 2
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.device = torch.cuda.current_device()
        self.copy_stream = torch.cuda.Stream()

    def submit(self, ids, mask):
        # target at p+1 controls the teacher distribution at p, not the route at p+1.
        cpu_ids = torch.empty(ids.shape, dtype=ids.dtype, pin_memory=True)
        cpu_mask = torch.empty(mask.shape, dtype=mask.dtype, pin_memory=True)
        ready = torch.cuda.Event(); begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        ready.record()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(ready); begin.record()
            cpu_ids.copy_(ids, non_blocking=True); cpu_mask.copy_(mask, non_blocking=True); end.record()
            ids.record_stream(self.copy_stream); mask.record_stream(self.copy_stream)
        def work():
            torch.cuda.set_device(self.device)
            end.synchronize(); start=time.perf_counter()
            result=torch.zeros((*ids.shape,self.table.shape[1]),dtype=torch.uint8,pin_memory=True)
            result.numpy()[:,:-1,:] = self.table[cpu_ids.numpy()[:,1:]] * cpu_mask.numpy()[...,None]
            return result, {'cpu_lookup_ms':1000*(time.perf_counter()-start),'lookup_d2h_ms':begin.elapsed_time(end)}
        return self.pool.submit(work)

    def finish(self, future):
        start=time.perf_counter(); cpu,stats=future.result()
        stats['lookup_host_wait_ms']=1000*(time.perf_counter()-start)
        begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        begin.record(); gpu=cpu.to('cuda',non_blocking=True); end.record()
        return gpu,stats,(begin,end),cpu

def retain_extras(extra, probabilities, multiplier):
    """Layer/rank-local floor(m*C) budget; no GPU scalar readback."""
    if multiplier == 1.0:return extra
    if multiplier == 0.0:return torch.zeros_like(extra)
    values=probabilities.float().masked_fill(~extra,float('-inf')).reshape(-1)
    order=torch.argsort(values,descending=True,stable=True)
    budget=torch.floor(extra.sum().to(torch.float64)*multiplier).to(torch.int64)
    keep=torch.arange(values.numel(),device=values.device)<budget
    return torch.zeros_like(extra).reshape(-1).scatter_(0,order,keep).reshape_as(extra)

class DynamicExperts:
    def __init__(self, teacher):
        self.bits=None
        self.multiplier=1.0
        self.extra_counts=torch.zeros(2,dtype=torch.int64,device='cuda')
        self.k_hist=torch.zeros(9,dtype=torch.int64,device='cuda')
        self.response_mask=None
        self.enabled=True
        for layer_id,layer in enumerate(teacher.model.layers):
            gate=layer.mlp.gate; calc=layer.mlp.calculator
            assert gate.num_experts==8 and gate.num_selects==2 and gate.use_softmax
            assert calc.__class__.__name__=='UniversalCalculator'
            old_gate=gate.forward; old_calc=calc.forward
            output_class=old_calc.__func__.__globals__['CalculatorOutput']
            def gate_forward(g,x,layer_id=layer_id,old=old_gate):
                if not self.enabled:return old(x)
                if self.multiplier==0.0:
                    if self.response_mask is not None:self.k_hist[2].add_(self.response_mask.sum())
                    return old(x)
                assert not g.training and self.bits is not None
                logits=g.gate_network(x)
                native=logits.topk(3,dim=1).indices[:,:2]
                all_ids=torch.arange(8,device=x.device).expand(x.shape[0],8)
                hist=self.bits[:,:,layer_id].reshape(-1).to(torch.int64)
                extra=((hist[:,None] >> all_ids)&1).bool()
                extra.scatter_(1,native,False)
                self.extra_counts[0].add_(extra.sum())
                if self.multiplier!=1.0:
                    extra=retain_extras(extra,logits.float().softmax(-1),self.multiplier)
                self.extra_counts[1].add_(extra.sum())
                indices=torch.cat((native,all_ids),1)
                valid=torch.cat((torch.ones_like(native,dtype=torch.bool),extra),1)
                selected_logits=logits.gather(1,indices).float().masked_fill(~valid,float('-inf'))
                scores=torch.softmax(selected_logits,dim=1).to(logits.dtype)
                filtered=torch.zeros_like(logits).scatter_add_(1,indices,scores)
                importance=filtered.sum(0); load=(filtered>0).sum(0)
                balance=(g.cv_squared(importance)+g.cv_squared(load))*g.balance_loss_weight if g.use_balance else logits.new_tensor(0.)
                if self.response_mask is not None:
                    k=valid.sum(1)[self.response_mask.reshape(-1)]
                    self.k_hist.add_(torch.bincount(k,minlength=9))
                return {'topK_indices':indices.masked_fill(~valid,-1),'topK_scores':scores,'balance_loss':balance,'load':load,'importance':importance}
            def calc_forward(c,x,topK_indices,topK_scores,old=old_calc,output_class=output_class,**kwargs):
                if not self.enabled or self.multiplier==0.0: return old(x,topK_indices,topK_scores,**kwargs)
                rows=torch.arange(x.shape[0],device=x.device)[:,None].expand_as(topK_indices)
                valid=topK_indices>=0
                experts=topK_indices[valid]; scores=topK_scores[valid]; rows=rows[valid]
                order=experts.argsort(); rows=rows[order]; scores=scores[order]
                sizes=experts.bincount(minlength=8).tolist()
                pieces=x.index_select(0,rows).split(sizes)
                outputs=torch.cat([c.experts(pieces[e],e) for e in range(8) if sizes[e]],0)
                if c.multiply_gate_scores:
                    if c.mlp_norm is None: outputs=outputs*(scores[:,None]*c.score_scale_factor)
                    else: outputs=c.mlp_norm(outputs*scores[:,None])
                y=outputs.new_zeros((x.shape[0],outputs.shape[1])).index_add_(0,rows,outputs)
                return output_class(hidden_states=y,num_dropped_tokens=torch.tensor(-1.))
            gate.forward=types.MethodType(gate_forward,gate)
            calc.forward=types.MethodType(calc_forward,calc)

    def set_batch(self,bits,mask=None):
        self.bits=bits; self.k_hist.zero_(); self.extra_counts.zero_()
        if mask is None:self.response_mask=None
        else:
            self.response_mask=torch.nn.functional.pad(mask,(0,1),value=False)
