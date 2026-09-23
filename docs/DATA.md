# Fixed data assets

Model/data files are external inputs; no automatic downloads or private paths are
embedded. Retain the original SHA manifests alongside locally prepared files.

## Dolly teacher

The historical SFT entry point expects JSON arrays of `{prompt: string, output:
string}` and the original fixed 11435/500 split. Supply those existing assets;
this package intentionally does not invent a new split that would claim to
reproduce the historical teacher. It trains from the original MoE checkpoint.

## Dolly distillation

Source: https://huggingface.co/datasets/MiniLLM/dolly and the Encoder class in
https://github.com/microsoft/LMOps/tree/main/minillm . Use the recorded raw.jsonl
and upstream source revision when reproducing an old run.

```bash
export DOLLY_WORK_DIR=/work/dolly_prepared
export TOKENIZER_MODEL=/models/tinyllama
export MINILLM_SOURCE_DIR=/sources/LMOps/minillm
export DOLLY_RAW_JSONL=/datasets/minillm_dolly/raw.jsonl
python data/prepare_dolly_distillation.py
```

First1000 raw entries are validation, remainder train, before discarding prompts
over256 tokens. No added BOS, append EOS then cap total512. Expected retained
10949/953. The script executes the upstream Encoder class, so acquire/review its
source separately. Paths must point to local files; no raw data is uploaded here.

## CITB

Source: https://github.com/hyintell/CITB . The original repository tree identity was
`bf50533b5bced4c388691ecc75e26773da96b3fd` (tree object, not a claimed commit hash).
Clone/check out the recorded data snapshot into `CITB_SOURCE_DIR` with all Arrow
files and `data/tasks` and `data/splits/CIT_splits/cl_38_random_tasks.txt`.
`sft/citb/source_files.json` carries the original Arrow Git blob hashes; preparation
checks them before reading. CL split uses the recorded seed42 and official
100/25/25 procedure; duplicates are retained, not deduplicated.

Expected counts: train13411, dev3450, initial_test2500, cl_test950,
official_test2975. The last split represents unseen tasks, the first two held-out
tests are new instances of trained tasks. The task/instance IDs must not overlap
between training and held-out splits. Official instructions include definition
and up to two positive examples via the attributed NI collator.

The TAID loader adds occurrence suffixes to duplicate source IDs while retaining
every instance. GKD data uses original IDs. This must not silently become a new
deduplicated training dataset.
