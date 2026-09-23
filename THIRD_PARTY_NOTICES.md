# Third-party attribution

- `gkd/official_ni_collator.py` and `sft/citb/official_ni_collator.py` derive from
  Tk-Instruct `src/ni_collator.py`: https://github.com/yizhongw/Tk-Instruct .
- CITB data/splitting protocol: https://github.com/hyintell/CITB ; Zhang et al.,
  *CITB: A Benchmark for Continual Instruction Tuning*, Findings EMNLP2023,
  DOI 10.18653/v1/2023.findings-emnlp.633.
- Dolly processing follows Microsoft LMOps/MiniLLM:
  https://github.com/microsoft/LMOps/tree/main/minillm . Upstream Encoder is loaded
  from a user-supplied source checkout, not bundled.
- TAID loss/controller and optimizer grouping adapt SakanaAI's TAID reference:
  https://github.com/SakanaAI/TAID . Local modifications support MoE methods,
  explicit microbatch accumulation, DDP and local checkpoint persistence.
- LLaMA-MoE and TinyLlama pretrained weights/custom model code are external assets
  and remain subject to their own terms; they are not included here.

This source staging directory does not grant a new license for upstream code or
data. Upstream license texts and exact source identities must accompany a public
release; the author's own project license is intentionally left undecided.

Available upstream license texts are preserved under `licenses/` (Tk-Instruct MIT, LMOps MIT, TAID Apache-2.0). CITB did not expose a license at the checked conventional main-branch paths; its dataset redistribution permissions have not been asserted. No CITB dataset content is bundled. TAID reference fixtures under `taid/official` are used only by the CPU equivalence tests.
