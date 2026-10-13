# Anonymous repository: deterministic evidence witnesses with abstention (CWF v2.1)

This repository accompanies an anonymous paper under double-blind review.

## Contents
- `benchmark/data/test3/` : 60 held-out cases (V01-V60) with the frozen gold, plus `MANIFEST.json`.
- `benchmark/annotations_A3.json`, `annotations_B3.json`, `adjudicated3.json` : annotator A (human), annotator B (LLM) and the adjudicated cases.
- `benchmark/*.py` : evaluation scripts (`stats3.py`, `selective3.py`, `oracle_tau.py`, `ablate_v2.py`, `llm_select_abstain.py`, `agree_C.py`, `debug_readonly.py`).
- `results/` : statistics, oracle-threshold, ablation and selective-prediction outputs reported in the paper.
- `REPRODUCE_test3.ps1`, `REPRODUCE_extra.ps1` : ordered commands that regenerate every number.

## Frozen files
CWF v2.1 and the test3 split were frozen before any system was run on test3 and are released unmodified.

## Notes
- All benchmark cases are synthetic. No patient data are included.
- NLI and Llama outputs are cached; re-running the Llama baselines needs a local Ollama installation.
- This is a research prototype, not a clinical tool.
