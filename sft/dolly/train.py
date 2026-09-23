#!/usr/bin/env python3
"""Full-parameter, native-routing Dolly SFT for the LLaMA-MoE teacher."""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--valid-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=-1, help="100 for sanity; -1 for all epochs")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-steps", type=int, default=20)
    p.add_argument("--diagnostic-samples", type=int, default=16)
    return p.parse_args()


def rank0():
    return dist.get_rank() == 0


def emit(path, record):
    if rank0():
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DollyDataset(Dataset):
    def __init__(self, path, tok, max_length, max_prompt_length):
        self.rows = json.load(open(path, encoding="utf-8"))
        self.items = []
        self.stats = {"prompt_raw": [], "response_raw": [], "total_raw": [],
                      "total_truncated": 0, "response_truncated": 0, "prompt_truncated": 0}
        for row in self.rows:
            prompt = row["prompt"]
            answer = row["output"].strip()
            raw_prompt = tok(prompt, add_special_tokens=True).input_ids
            raw_response = tok(answer + tok.eos_token, add_special_tokens=False).input_ids
            prompt_ids = raw_prompt[:max_prompt_length]
            room = max_length - len(prompt_ids)
            if room < 1:
                raise ValueError("No room for a response token")
            response_ids = raw_response[:room]
            ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
            if not response_ids or len(ids) > max_length or labels[len(prompt_ids)] == -100:
                raise AssertionError("Invalid response-only mask")
            self.items.append({"input_ids": ids, "labels": labels,
                               "prompt_len": len(prompt_ids), "response_len": len(response_ids)})
            self.stats["prompt_raw"].append(len(raw_prompt))
            self.stats["response_raw"].append(len(raw_response))
            self.stats["total_raw"].append(len(raw_prompt) + len(raw_response))
            self.stats["prompt_truncated"] += len(raw_prompt) > len(prompt_ids)
            self.stats["response_truncated"] += len(raw_response) > len(response_ids)
            self.stats["total_truncated"] += len(raw_prompt) + len(raw_response) > max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(rows, pad_id):
    n = max(len(x["input_ids"]) for x in rows)
    ids = [x["input_ids"] + [pad_id] * (n - len(x["input_ids"])) for x in rows]
    labels = [x["labels"] + [-100] * (n - len(x["labels"])) for x in rows]
    attention = [[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in rows]
    if any(any(v != -100 for v in label[:x["prompt_len"]]) for label, x in zip(labels, rows)):
        raise AssertionError("Prompt labels were not masked")
    if any(any(v != -100 for v in label[len(x["input_ids"]):]) for label, x in zip(labels, rows)):
        raise AssertionError("Padding labels were not masked")
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long)}


def describe_lengths(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
            "max": max(values)}


def make_report(args, tok, train, valid, total_params, trainable_params, decay_count, no_decay_count, world, total_steps, eval_steps):
    report = {
        "model": args.model, "train_file": args.train_file, "valid_file": args.valid_file,
        "train_sha256": sha256(args.train_file), "valid_sha256": sha256(args.valid_file),
        "train_samples": len(train), "valid_samples": len(valid), "test_samples": None,
        "total_parameters": total_params, "trainable_parameters": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params,
        "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
        "betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": args.max_grad_norm,
        "decay_group_parameters": decay_count, "no_decay_group_parameters": no_decay_count,
        "decay_group_weight_decay": args.weight_decay, "no_decay_group_weight_decay": 0.0,
        "scheduler": "cosine", "warmup_steps": 0, "planned_epochs": args.epochs,
        "planned_optimizer_steps": total_steps, "eval_every_optimizer_steps": eval_steps,
        "micro_batch_per_gpu": args.micro_batch, "gpu_count": world,
        "gradient_accumulation": args.grad_accum,
        "effective_global_batch": args.micro_batch * world * args.grad_accum,
        "max_sequence_length": args.max_length, "max_prompt_length": args.max_prompt_length,
        "precision": "BF16 forward/reduce, FP32 optimizer parameters",
        "distributed_strategy": "PyTorch FSDP FULL_SHARD, transformer-layer auto-wrap, use_orig_params",
        "routing": "native top-2, unchanged", "native_gate_balance_coefficient": 0.01,
        "seed": args.seed, "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__,
        "training_command_argv": sys.argv, "source_sha256": sha256(__file__),
        "git_commit": None,
        "train_lengths": {k: describe_lengths(train.stats[k]) for k in ("prompt_raw", "response_raw", "total_raw")},
        "train_truncation_rates": {k: train.stats[k] / len(train) for k in ("prompt_truncated", "total_truncated", "response_truncated")},
    }
    if rank0():
        with (Path(args.output_dir) / "config.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        for filename in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
            source = Path(args.model) / filename
            if source.exists():
                shutil.copy2(source, Path(args.output_dir) / f"source_{filename}")
        for i in (0, len(train) // 2, len(train) - 1):
            ex = train[i]
            prompt_len = ex["prompt_len"]
            print(json.dumps({"sample_index": i, "decoded_prompt": tok.decode(ex["input_ids"][:prompt_len]),
                "decoded_response": tok.decode(ex["input_ids"][prompt_len:]),
                "input_ids_length": len(ex["input_ids"]), "prompt_length": prompt_len,
                "response_length": ex["response_len"], "masked_token_positions": [0, prompt_len - 1],
                "loss_token_count": sum(x != -100 for x in ex["labels"])}, ensure_ascii=False), flush=True)
        print(json.dumps(report, ensure_ascii=False), flush=True)


def ce_sum(outputs, labels):
    logits = outputs.logits[:, :-1, :].contiguous().float()
    shifted = labels[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), shifted.view(-1),
                           ignore_index=-100, reduction="sum")


def snapshot_rng():
    return (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone())


def check_rng(before):
    after = snapshot_rng()
    if not torch.equal(before[0], after[0]) or not torch.equal(before[1], after[1]):
        raise AssertionError("Validation changed RNG state")


@torch.no_grad()
def evaluate(model, valid, tok, device, world, output_path, step, epoch):
    before_rng = snapshot_rng()
    model.eval()
    indexes = range(dist.get_rank(), len(valid), world)
    loader = DataLoader(torch.utils.data.Subset(valid, indexes), batch_size=1,
                        collate_fn=lambda rows: collate(rows, tok.pad_token_id),
                        generator=torch.Generator().manual_seed(0))
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for cpu_batch in loader:
        batch = {k: v.to(device) for k, v in cpu_batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True)
            total_ce = ce_sum(out, batch["labels"])
        totals[0] += total_ce.detach().double()
        totals[1] += (batch["labels"][:, 1:] != -100).sum().double()
    dist.all_reduce(totals)
    nll = (totals[0] / totals[1]).item()
    ppl = math.exp(min(nll, 50.0))
    model.train()
    check_rng(before_rng)
    emit(output_path, {"kind": "validation", "step": step, "epoch": epoch,
                       "validation_loss": nll, "validation_ppl": ppl,
                       "validation_response_tokens": int(totals[1].item())})
    return nll, ppl


def save_model(model, destination, metadata, source_model):
    dist.barrier()
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()
    if rank0():
        destination.mkdir(parents=True, exist_ok=True)
        temp = destination / "pytorch_model.bin.tmp"
        torch.save(state, temp)
        os.replace(temp, destination / "pytorch_model.bin")
        for filename in ("config.json", "configuration_llama_moe.py", "modeling_llama_moe_hf.py",
                         "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                         "generation_config.json"):
            source = Path(source_model) / filename
            if source.exists():
                shutil.copy2(source, destination / filename)
        with (destination / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    del state
    dist.barrier()


@torch.no_grad()
def save_reference_logits(model, valid, tok, device, destination):
    model.eval()
    cpu_batch = collate([valid[0]], tok.pad_token_id)
    batch = {k: v.to(device) for k, v in cpu_batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True)
    if rank0():
        torch.save({"input_ids": cpu_batch["input_ids"],
                    "attention_mask": cpu_batch["attention_mask"],
                    "last_position_logits": outputs.logits[0, -1].detach().float().cpu()},
                   destination / "reference_logits.pt")
    model.train()
    dist.barrier()


def diagnostic(model, valid, tok, device, output_path, step, max_samples):
    """Native gate hooks on a fixed validation subset; no routing changes."""
    if max_samples <= 0:
        return
    gates = [m for m in model.modules() if type(m).__name__ == "TopKBalancedNoisyGate"]
    counts = [torch.zeros(g.num_experts, dtype=torch.float64, device=device) for g in gates]
    entropies = [torch.zeros((), dtype=torch.float64, device=device) for _ in gates]
    top_prob = [torch.zeros(2, dtype=torch.float64, device=device) for _ in gates]
    token_count = [torch.zeros((), dtype=torch.float64, device=device) for _ in gates]
    hooks = []
    for j, gate in enumerate(gates):
        def hook(_module, _inputs, out, idx=j):
            ids = out["topK_indices"].detach()
            counts[idx].scatter_add_(0, ids.flatten(), torch.ones_like(ids.flatten(), dtype=torch.float64))
            full = torch.softmax(_module.gate_network(_inputs[0]).float(), dim=-1).double()
            entropies[idx] += -(full * torch.log(full.clamp_min(1e-12))).sum()
            selected = full.gather(1, ids)
            top_prob[idx] += selected.sum(0)
            token_count[idx] += ids.shape[0]
        hooks.append(gate.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        for i in range(min(max_samples, len(valid))):
            cpu_batch = collate([valid[i]], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      use_cache=False, return_dict=True)
    for hook in hooks:
        hook.remove()
    model.train()
    if rank0():
        rows = []
        for i in range(len(gates)):
            c = counts[i].cpu().tolist()
            t = token_count[i].item()
            rows.append({"layer": i, "frequency": [v / max(t, 1) for v in c],
                         "router_entropy": (entropies[i] / max(t, 1)).item(),
                         "max_over_mean_load": max(c) / max(statistics.mean(c), 1e-12),
                         "experts_receiving_tokens": sum(v > 0 for v in c),
                         "top1_probability": (top_prob[i][0] / max(t, 1)).item(),
                         "top2_probability": (top_prob[i][1] / max(t, 1)).item()})
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"step": step, "subset_first_n": max_samples, "layers": rows}, f, indent=2)
    dist.barrier()


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    outdir = Path(args.output_dir)
    if rank0():
        outdir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    events = outdir / "events.jsonl"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train = DollyDataset(args.train_file, tok, args.max_length, args.max_prompt_length)
    valid = DollyDataset(args.valid_file, tok, args.max_length, args.max_prompt_length)
    sampler = DistributedSampler(train, num_replicas=world, rank=dist.get_rank(),
                                 shuffle=True, seed=args.seed, drop_last=False)
    loader = DataLoader(train, batch_size=args.micro_batch, sampler=sampler, num_workers=0,
                        collate_fn=lambda rows: collate(rows, tok.pad_token_id))
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    eval_steps = max(1, steps_per_epoch // 2)
    raw = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                torch_dtype=torch.float32, low_cpu_mem_usage=True)
    if raw.config.num_selects != 2:
        raise AssertionError(f"Expected native top-2, got {raw.config.num_selects}")
    if not raw.config.gate_use_balance:
        raise AssertionError("Original gate balance loss unexpectedly disabled")
    raw.config.use_cache = False
    raw.model.gradient_checkpointing = True
    if any("lora" in name.lower() or "adapter" in name.lower() for name, _ in raw.named_parameters()):
        raise AssertionError("PEFT/adapter parameters present")
    total_params = sum(p.numel() for p in raw.parameters())
    trainable_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    if trainable_params / total_params < 0.999:
        raise AssertionError("Full-parameter training is not enabled")
    decay_count = sum(p.numel() for n, p in raw.named_parameters() if p.requires_grad and
                      not (n.lower().endswith(".bias") or "norm" in n.lower()))
    no_decay_count = trainable_params - decay_count
    make_report(args, tok, train, valid, total_params, trainable_params,
                decay_count, no_decay_count, world, total_steps, eval_steps)
    layer_cls = type(raw.model.layers[0])
    from functools import partial
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)
    model = FSDP(raw, auto_wrap_policy=policy, mixed_precision=mp,
                 sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
                 device_id=device, sync_module_states=True)
    expected_grad_names = {name for name, param in model.named_parameters() if param.requires_grad}
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.lower().endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    best = float("inf")
    best_record = None
    step = 0
    samples_seen = 0
    target_tokens_seen = 0
    grad_seen = set()
    diagnostic(model, valid, tok, device, outdir / "router_pre_sft.json", 0, args.diagnostic_samples)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(steps_per_epoch):
            group = list(itertools.islice(iterator, args.grad_accum))
            if not group:
                break
            step += 1
            t0 = time.perf_counter()
            local_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in group)
            global_tokens_tensor = torch.tensor(local_tokens, dtype=torch.float64, device=device)
            dist.all_reduce(global_tokens_tensor)
            global_tokens = int(global_tokens_tensor.item())
            global_samples = sum(len(b["input_ids"]) for b in group) * world
            opt.zero_grad(set_to_none=True)
            local_ce = 0.0
            local_aux = 0.0
            lr_used = opt.param_groups[0]["lr"]
            for cpu_batch in group:
                batch = {k: v.to(device) for k, v in cpu_batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    use_cache=False, return_dict=True)
                    ce = ce_sum(outputs, batch["labels"])
                    aux = outputs.balance_loss
                    loss = ce * world / global_tokens
                    if aux is not None and bool((aux > 0).item()):
                        loss = loss + aux / len(group)
                        local_aux += float(aux.detach()) / len(group)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite train loss at step {step}")
                loss.backward()
                local_ce += float(ce.detach())
            if step == 100:
                for name, param in model.named_parameters():
                    if (param.requires_grad and param.grad is not None and param.grad.numel()
                            and bool(torch.any(param.grad != 0).item())):
                        grad_seen.add(name)
            grad_norm = model.clip_grad_norm_(args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}")
            opt.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            samples_seen += global_samples
            target_tokens_seen += global_tokens
            if step == 1 or step % args.log_steps == 0:
                sums = torch.tensor([local_ce, local_aux], dtype=torch.float64, device=device)
                dist.all_reduce(sums)
                emit(events, {"kind": "train", "step": step,
                    "epoch": epoch + (step - epoch * steps_per_epoch) / steps_per_epoch,
                    "train_response_CE": (sums[0] / global_tokens).item(),
                    "native_balance_loss": (sums[1] / world).item(),
                    "learning_rate": lr_used, "grad_norm": float(grad_norm),
                    "tokens_per_batch": global_tokens, "samples_per_batch": global_samples,
                    "step_time": elapsed,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(device),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(device)})
            if step % eval_steps == 0 or step == total_steps or step == args.max_steps:
                nll, ppl = evaluate(model, valid, tok, device, world, events, step,
                                    epoch + (step - epoch * steps_per_epoch) / steps_per_epoch)
                is_best = nll < best
                if is_best:
                    best = nll
                    best_record = {"best_epoch": epoch + (step - epoch * steps_per_epoch) / steps_per_epoch,
                                   "best_step": step, "best_validation_loss": nll,
                                   "best_validation_ppl": ppl}
                    if args.max_steps < 0:
                        save_model(model, outdir / "best_val_loss", best_record, args.model)
                        diagnostic(model, valid, tok, device, outdir / "router_best_sft.json",
                                   step, args.diagnostic_samples)
                emit(events, {"kind": "selection", "step": step, "is_best": is_best,
                              "best_validation_loss": best})
            if step in (steps_per_epoch, 5 * steps_per_epoch):
                diagnostic(model, valid, tok, device, outdir / f"router_step_{step}.json",
                           step, args.diagnostic_samples)
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break
    if args.max_steps > 0:
        save_model(model, outdir / "sanity_checkpoint", {"step": step, "validation": best_record}, args.model)
        save_reference_logits(model, valid, tok, device, outdir / "sanity_checkpoint")
    else:
        save_model(model, outdir / "last", {"step": step, "epoch": args.epochs}, args.model)
        diagnostic(model, valid, tok, device, outdir / "router_last.json", step, args.diagnostic_samples)
    gathered = [None] * world
    dist.all_gather_object(gathered, sorted(grad_seen))
    missing_grad_names = sorted(expected_grad_names - set(itertools.chain.from_iterable(gathered)))
    if rank0():
        with (outdir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump({"optimizer_steps": step, "samples_seen": samples_seen,
                       "target_tokens_seen": target_tokens_seen, "best": best_record,
                       "parameters_with_nonzero_gradient_first_100_steps":
                       len(set(itertools.chain.from_iterable(gathered))),
                       "expected_trainable_parameter_tensors": len(expected_grad_names),
                       "parameter_tensors_without_nonzero_gradient_step_100": missing_grad_names}, f, indent=2)
    dist.barrier()
    dist.destroy_process_group()
    if args.max_steps > 0 and missing_grad_names:
        raise AssertionError(f"Sanity gradient coverage failed for {len(missing_grad_names)} tensors")


if __name__ == "__main__":
    main()
