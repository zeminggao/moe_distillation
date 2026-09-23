"""Bounded six-GPU full-parameter on-policy GKD benchmark. Never starts a formal run."""
import argparse,json,os,time,random,contextlib
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent

def load(path,dtype):
    cfg=AutoConfig.from_pretrained(path,trust_remote_code=True)
    if cfg.model_type=='llama_moe':cfg.rope_scaling=None
    cfg._attn_implementation='eager' if cfg.model_type=='llama_moe' else 'sdpa'
    m=AutoModelForCausalLM.from_pretrained(path,config=cfg,trust_remote_code=True,torch_dtype=dtype,low_cpu_mem_usage=True)
    m.config.use_cache=False
    return m
