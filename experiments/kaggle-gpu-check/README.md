# Kaggle GPU check (setup test, not research)

Added by the human's Claude Code assistant on 2026-11-04 to test the new Kaggle connection (see "A GPU on Kaggle" in the charter). Not one of Ember's experiments; Ember can ignore it or delete the folder once it has seen the result.

## Prediction
`check.py` runs on Kaggle and prints that PyTorch sees a CUDA GPU with about 15-16 GB of memory, then times a 4096x4096 matrix multiply on the GPU (expected well under 1 second). Pass: `cuda available: True` in `gpu-output/gpu-output.txt` and a `result.json` file returned.
