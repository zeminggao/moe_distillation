# MoE-to-Dense Distillation: Dolly and CITB

**Local release candidate, not yet uploaded.** Teacher full-parameter SFT and dense
TinyLlama distillation with GKD / TAID; Top-2, Top-8, Strawman, KID, KA and SAR.
Only Dolly and CITB are supported by this package. Historical experiment folders
are untouched. This is a portable source extraction, not a claim that new GPU runs
have reproduced all published numbers.

## Layout

| Directory | Contents |
|---|---|
| `sft/dolly` | Historical full-parameter, response-only CE teacher training |
| `sft/citb` | Fixed official joint split, tokenization, full-parameter teacher SFT |
| `data` | Dolly distillation preparation and split checks |
| `gkd` | Six-GPU on-policy reverse-KL training, routing methods, history and evaluation |
| `taid` | Two-GPU TAID, interpolation controller, routing methods and CPU tests |
| `configs` | Data path template; no real server paths or credentials |
| `docs` | Protocol, data provenance, release changes and reproducibility limits |

## Environment

Target: Linux, Python 3.10+, NVIDIA CUDA. Install a compatible PyTorch CUDA wheel
first, then `pip install -r requirements.txt`. The requirements are a candidate
compatibility range, not a frozen historical environment. A model directory must
contain the custom LLaMA-MoE implementation required by `trust_remote_code=True`.
Review that upstream code before use. TinyLlama and teacher tokenizers must agree.
See `THIRD_PARTY_NOTICES.md` for upstream dependencies and attribution.

## Data preparation

No datasets, model weights, samples, logs or checkpoints are distributed here.
Use the original fixed split assets or prepare from the recorded upstream sources;
do not randomly re-split. See [data instructions](docs/DATA.md).

1. Copy `configs/data.example.json` outside this repository and set absolute paths.
2. Prepare the common tokenized distillation representation:

```bash
python gkd/prepare_data.py --config /work/data.json --dataset dolly --output /work/dolly
python gkd/prepare_data.py --config /work/data.json --dataset citb --output /work/citb
python data/validate_fixed_splits.py --data /work/citb
```

3. For Strawman/KID, construct a **train-only** table before configuration. The two
   objectives have distinct history preprocessing; use separate data directories
   per objective. Never overwrite an existing history directory.

```bash
# GKD (six GPUs)
torchrun --standalone --nproc_per_node=6 gkd/history.py --data /work/gkd_dolly --teacher /models/dolly_teacher --microbatch 1
# TAID (two GPUs; CITB caps are 1024/128)
torchrun --standalone --nproc_per_node=2 taid/history.py --data /work/taid_dolly --teacher /models/dolly_teacher --prompt-cap 256 --response-cap 256 --microbatch 1
```

## Distillation

```bash
python configure.py --objective gkd --dataset dolly --method kid \
  --teacher /models/dolly_teacher --student /models/tinyllama \
  --data /work/gkd_dolly --output /work/runs/gkd_dolly_kid
python run.py --objective gkd --config /work/runs/gkd_dolly_kid/config.json --stage preflight
python run.py --objective gkd --config /work/runs/gkd_dolly_kid/config.json --stage train

python configure.py --objective taid --dataset citb --method kid \
  --teacher /models/citb_teacher --student /models/tinyllama \
  --data /work/taid_citb --output /work/runs/taid_citb_kid
python run.py --objective taid --config /work/runs/taid_citb_kid/config.json --stage preflight
python run.py --objective taid --config /work/runs/taid_citb_kid/config.json --stage train
```

Valid methods: `top2 top8 strawman kid ka sar`. Preflight performs normal three-step
and full-length two-step checks. Training requires successful checks for the same
configuration and code. KID keeps a 64-sample logical routing budget; microbatching
must not change selection into independent microbatch budgets.

All output remains local. There is no SSH service, automatic job queue, shutdown,
or background monitoring. GKD retains all complete snapshots and optimizer states,
so plan storage accordingly. TAID retains best/latest weights and latest recovery
state following the extracted trainer's policy. Both check save headroom.

`run.py --resume /path/to/checkpoint` uses the trainer's explicit recovery support.
GKD recovery is currently implemented for Top-2, Top-8 and SAR only; other methods
must not be resumed from weights while claiming exact recovery. TAID restores its
controller, optimizer, scheduler and RNG. SFT recovery limitations are in the docs.

## Teacher fine-tuning

```bash
# Historical Dolly teacher recipe; original fixed teacher split is required.
torchrun --standalone --nproc_per_node=2 sft/dolly/train.py \
  --model /models/llama-moe-original --train-file /data/dolly_teacher/train.json \
  --valid-file /data/dolly_teacher/valid.json --output-dir /work/dolly_teacher \
  --epochs 10 --micro-batch 2 --grad-accum 8 --lr 1e-5 \
  --weight-decay .01 --max-length 512 --max-prompt-length 256 --seed 42

# CITB joint full-parameter SFT (not sequential continual learning).
export CITB_WORK_DIR=/work/citb_teacher
export TEACHER_MODEL=/models/llama-moe-original
export CITB_SOURCE_DIR=/sources/CITB
python sft/citb/prepare_official.py
python sft/citb/tokenize_data.py
torchrun --standalone --nproc_per_node=2 sft/citb/train.py --smoke
torchrun --standalone --nproc_per_node=2 sft/citb/train.py
```

Use a fresh work directory. CITB's best-only SFT checkpoints do not contain full
optimizer recovery state. Historical resume flags are only for older full-state
checkpoints; they are not a guarantee of recovery from a best-only checkpoint.

## Evaluation and tests

```bash
python -m unittest discover -s taid -p 'test_*.py' -v
python tests/test_release.py
python gkd/evaluate_citb.py --checkpoint /models/selected_student \
  --data /work/citb --output /work/citb_test_results --batch-size 4
```

Choose checkpoints by validation ROUGE-L, never downstream test scores. The CITB
evaluation entry point reports the three held-out splits. Timing curves record
pure training separately from history construction, probes, saving and validation.
See [protocol](docs/PROTOCOL.md) for differences between objectives.

## Before publishing

The author's project license has not been chosen; no license is invented here.
Review third-party licenses, pin a tested dependency lock, and run the bundled
GPU preflight on the target hardware. Check `VALIDATION.md` for what was actually
tested during source organization. Upload only this directory, not its parent.
