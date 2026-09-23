"""SAR with legacy-noise replay and selected-token LM heads.

The two-pass global balance objective and optimizer update are unchanged.
Only teacher routers receive gradients. Full-vocabulary logits are never made
in the statistics pass; the KL heads are checkpointed in 128-token chunks.
"""
import time
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from expert_methods import cv_squared


def sum_all(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def selected_head_kl(teacher_head, student_head, hidden, reference, weights, chunk=128):
    def term(h, ref, w):
        with torch.no_grad():
            logq = student_head(ref).float().log_softmax(-1)
        logp = teacher_head(h).float().log_softmax(-1)
        return ((logp.exp() * (logp-logq)).sum(-1) * w).sum()
    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(hidden), chunk):
        loss = loss + checkpoint(term, hidden[start:start+chunk], reference[start:start+chunk],
                                 weights[start:start+chunk], use_reentrant=False)
    return loss


def update_router(policy, student, optimizer, ids, attention, mask, global_samples,
                  microbatch=1):
    teacher = policy.teacher
    optimizer.zero_grad(set_to_none=True)
    old_training = student.training
    student.eval()
    # Keep legacy backbone shapes and noise call ordering. Only the LM heads
    # are restricted to loss-bearing positions. BF16 padding trim is not enabled.
    batches=[]
    for start in range(0,len(ids),microbatch):
        sl=slice(start,min(start+microbatch,len(ids)))
        batches.append((sl,ids[sl],attention[sl],mask[sl]))
    def set_batch(attn):
        policy.set_batch(attn)
    policy.sums = torch.zeros((len(teacher.model.layers), 8), device=ids.device)
    policy.collect = True
    states = []
    torch.cuda.synchronize();start_time=time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for sl, tokens, attn, valid in batches:
            states.append(torch.cuda.get_rng_state())
            set_batch(attn)
            result = teacher.model(input_ids=tokens, attention_mask=attn, use_cache=False)
            del result
    final_rng = torch.cuda.get_rng_state()
    torch.cuda.synchronize();statistics_seconds=time.perf_counter()-start_time
    policy.collect = False
    sums = sum_all(policy.sums).detach().requires_grad_(True)
    balance = .01 * sum(cv_squared(row) for row in sums)
    policy.balance_coefficients = torch.autograd.grad(balance, sums)[0].detach()
    loss_total = torch.zeros((), device=ids.device)
    torch.cuda.synchronize();start_time=time.perf_counter()
    for j, (sl, tokens, attn, valid) in enumerate(batches):
        torch.cuda.set_rng_state(states[j])
        set_batch(attn)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            reference = student.model(input_ids=tokens, attention_mask=attn, use_cache=False).last_hidden_state[:, :-1][valid].detach()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = teacher.model(input_ids=tokens, attention_mask=attn, use_cache=False)
            hidden = output.last_hidden_state[:, :-1][valid]
            counts = valid.sum(-1)
            assert bool((counts > 0).all()), 'Empty response'
            weights = ((1/counts.float())[:, None].expand_as(valid)[valid] / global_samples)
            weighted = selected_head_kl(teacher.lm_head,student.lm_head,hidden,reference,weights)
            loss = weighted + output.balance_loss
        loss.backward()
        loss_total += weighted.detach()
        del output, reference, hidden, loss, weighted, weights
    torch.cuda.set_rng_state(final_rng)
    policy.balance_coefficients = None
    policy.set_batch(attention)
    torch.cuda.synchronize();backward_seconds=time.perf_counter()-start_time
    for parameter in policy.router_parameters:
        if parameter.grad is None:
            raise RuntimeError('A router parameter received no gradient')
        sum_all(parameter.grad)
    norm = torch.nn.utils.clip_grad_norm_(policy.router_parameters, 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite SAR router gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    student.train(old_training)
    return dict(router_kl=sum_all(loss_total).item(), router_balance=balance.item(),
                router_grad_norm=norm.item(),router_statistics_seconds=statistics_seconds,
                router_backward_seconds=backward_seconds,
                router_padded_positions_before=len(ids)*ids.shape[1],
                router_padded_positions_after=sum(x[1].numel() for x in batches))
