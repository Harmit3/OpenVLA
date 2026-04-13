# Reproducibility Guide

## Environment
- Ubuntu 22.04.5 under WSL2
- NVIDIA RTX 2060
- Python 3.10 virtual environment
- OpenVLA-7B loaded in 4-bit mode

## Critical engineering note
The local inference path required a patch that prepends a 256-token image attention mask before `predict_action`.

## High-level reproduction order
1. Set up the Python environment.
2. Verify GPU and package versions.
3. Run `test_openvla_PATCHED_WORKING.py`.
4. Reproduce `manual40` with `run_metadata_experiment.py`.
5. Reproduce `auto50` with `run_bridge_batch_from_metadata.py`.
6. Regenerate summaries with `build_analysis_artifacts.py`.
