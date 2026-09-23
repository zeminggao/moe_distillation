# Packaging changes from the experimental scripts

- Explicit allowlist extraction; no recursive copy of experiment output folders.
- Removed the third dataset's configuration, tokenizer adapter and evaluator branch.
- Exposed Dolly/CITB and six routing methods through `configure.py`.
- Added a finite preflight/train launcher; no automatic continuation between runs.
- GKD retains local checkpoints; removed the dependency on remote `.archived`
  markers to avoid a permanent storage backpressure wait without an SSH archiver.
  Free-space guards remain. This changes storage orchestration, not the KD loss.
- TAID final archival is local-only; no credential gate or detached SSH finalizer.
- Removed private deployment/admin scripts and server-specific paths. CITB SFT
  inputs are supplied through environment variables. The data builder reads a
  local upstream checkout and verifies recorded source file Git blob hashes.
- Preserved objective, masking, routing selection, KA double updates, SAR updates,
  KID logical batch64 budget, optimizer precision, seeds and save schedules in the
  extracted training engines. CPU tests exercise these central invariants.
- Included upstream test references for TAID, source attribution and available
  license texts. The test reference snippets are not standalone runtime modules.

This directory is a source release candidate. Its portable launch/storage changes
have not undergone a new end-to-end GPU training run. Do not claim byte-identical
reproduction of earlier separately implemented Dolly experiments.
