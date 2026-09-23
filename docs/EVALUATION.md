# Evaluation

For CITB held-out generation tests, use `gkd/evaluate_citb.py` with a checkpoint
selected by the 3450-example dev ROUGE-L. It evaluates initial_test, cl_test and
official_test using greedy128 and all references. Training's final-model tests
and subsequent dev-best-model tests are different selections; label them clearly.

For downstream evaluation install lm-evaluation-harness at commit
`d6de81643928d653435c431bae19945d41d32520` in a separate environment, then run:

```bash
python -m lm_eval --model hf \
  --model_args pretrained=/models/dev_best_student,dtype=bfloat16,trust_remote_code=True,max_length=2048,use_fast_tokenizer=False \
  --tasks hellaswag --num_fewshot 10 --batch_size 4 --device cuda:0 \
  --seed 42,42,42,42 --output_path /work/hellaswag --log_samples

python -m lm_eval --model hf \
  --model_args pretrained=/models/dev_best_student,dtype=bfloat16,trust_remote_code=True,max_length=2048,use_fast_tokenizer=False \
  --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 4 --device cuda:0 \
  --seed 42,42,42,42 --output_path /work/truthfulqa --log_samples
```

Report HellaSwag `acc_norm` and TruthfulQA MC2 `acc`. Do not select checkpoints
using downstream scores. The harness is an external dependency, not vendored.
