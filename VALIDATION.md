# Release validation — 2026-09-23

Passed:

- All four training entry points (GKD, TAID, Dolly SFT, CITB SFT) imported and
  returned `--help` successfully on Linux with GPUs hidden.

- Python syntax parsing of all bundled Python files.
- Three packaging/CPU checks, including generation of all 24
  objective × dataset × method configurations and GKD incomplete-batch coverage.
- Sixteen PyTorch CPU semantic tests on an existing Linux environment with
  `CUDA_VISIBLE_DEVICES` empty: TAID reference loss/gradients and controller,
  accumulation scaling, optimizer restoration, KA two updates, SAR microbatch
  gradients/student freezing, KID logical selection budget and controller resume.
- Public source scan for experiment server addresses, usernames and embedded
  deployment credentials; model weights/data samples/logs excluded from archive.

The first local CPU attempt lacked PyTorch. The first Linux test pass exposed two
missing upstream reference fixtures; those fixtures were added and all16 passed.
The local packaging checks require only Python's standard library.

Not performed:

- New GPU preflight or full training of this reorganized package.
- End-to-end rebuilding from raw upstream data or installing the candidate
  requirements into a fresh environment.
- New evaluation, multi-seed experiments, or claims of improved performance.

Run the documented preflight on the actual target machine before production.
