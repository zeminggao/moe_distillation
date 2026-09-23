# Recorded protocol and interpretation

| Student setting | Dolly | CITB |
|---|---:|---:|
| GKD epochs / global steps | 10 / 290 | 10 / 350 |
| TAID configured epochs / global steps | 5 / 140 | 10 / 340 |
| Prompt / response caps | 256 / 256 | 1024 / 128 |
| Distillation train / dev examples | 10949 / 953 | 13411 / 3450 |

TAID CITB configuration does not mean every historical method finished training.
GKD retains incomplete batches; TAID drops them. KA has **two student optimizer
updates per global batch**, unlike other methods. Global batch is 384; normal
student microbatch is 4, accumulation 16 (six-GPU GKD) or 48 (two-GPU TAID).

Both update every dense student parameter; no student LoRA. BF16 computation,
FP32 student parameters/Adam states, gradient clipping 1, activation checkpointing.
AdamW defaults betas .9/.999 and epsilon 1e-8; cosine schedule, no LR warmup.
GKD LR 2e-5/decay 0; TAID LR 1e-4/decay .01 with name-based exclusions.
All teacher parameters are frozen **except SAR's router**, LR 2e-5.

GKD samples student responses at temperature 1, top-p 1, no top-k truncation;
optimizes reverse KL(student || teacher) on response positions, normalized per
response then per sample. No supervised hard-label loss is added.
TAID uses gold sequences, soft-target cross entropy over all valid prediction
positions (prompt included); the detached logits interpolate with t_start=.2,
t_end=1, alpha=.0005, beta=.99. The original soft CE scalar (target entropy not
subtracted) drives the adaptive controller. These are not identical data/loss
protocols and should not be described as such.

GKD fixed generated tensor padding and TAID per-batch padding differ. PAD=EOS;
padding is masked. History uses training gold response targets, prediction
position target-1, excludes special tokens; 80% coverage, expert-ID tie breaks.
Strawman takes history union native Top-2; KID removes extras subject to the
controller and each logical rank's 64-sample candidate budget.

Validation twice per epoch (midpoint rounded up and epoch end). Dolly generation
is sampling, six logical shards/seed 10+shard; CITB is greedy, 128 new tokens,
multi-reference ROUGE-L. Seed 42, rank offset for RNG, fixed epoch shuffle.
This is a single-run comparison, not a multi-seed estimate or bitwise equality
across hardware/objectives. The unified portable GKD entry point derives from
the later multi-dataset engine; it is not asserted to recreate every instruction
of the earlier separate Dolly executables.

Teacher SFT differs from student KD: historical Dolly teacher used 11435/500
split, full-parameter CE, two-GPU FSDP, batch32, LR1e-5, 10 epochs. Distillation
uses the later 10949/953 MiniLLM-derived split. **These two splits are not the same;
teacher versus distillation-dev overlap must be audited before claiming an
independent teacher-held-out evaluation.** The deployed Dolly teacher was the
historical last checkpoint, not necessarily the minimum-NLL checkpoint.
CITB SFT uses official joint132 tasks, global16, LR5e-5, constant schedule,
zero decay, max15 epochs, dev every500 steps, early stopping after3 non-improvements.
Teacher is a decoder-only MoE adaptation of an originally T5-based protocol.
