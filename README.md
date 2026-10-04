# ATML PA2 - LLM Post-Training

# Task 1

## Task 1.1

For Standard Output, run:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m task1_dpo.train --config configs/dpo.yaml --run-name standard`

# Task 1.2

the β study:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m task1_dpo.ablate_beta --config configs/dpo.yaml`