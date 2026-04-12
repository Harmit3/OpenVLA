
# OpenVLA Prompt-Robustness Project

This repository packages a completed research project on **prompt-induced action drift in OpenVLA-7B**.

## One-sentence summary
This project asks a simple safety question: **if the picture stays the same but the wording changes, does the robot action change too?**

## Completed experiments

### 1. Manual40
- 40 trajectories
- 40 initial images + 40 final images
- hand-curated prompts

### 2. Auto50
- 50 trajectories
- 50 initial images + 50 final images
- automatically sampled scripted BridgeData branch

## Main result
Across both experiments:
- paraphrased prompts stayed closest to the original prompt
- contradictory and neutral prompts caused larger drift
- final-state images were usually more fragile than initial-state images

## Repository layout
- `code/`: scripts used for extraction, running, and analysis
- `metadata/`: manual and automatic metadata files
- `results/`: raw result CSV files
- `analysis/`: summary tables, confidence intervals, family breakdowns, and failure cases
- `docs/`: long reports and paper drafts
- `logs/`: run logs


## Main files for reproducibility
- `code/run_metadata_experiment.py`
- `code/run_bridge_batch_from_metadata.py`
- `metadata/metadata_manual.csv`
- `metadata/metadata_sampled_filled.csv`
- `results/results_initial.csv`
- `results/results_final.csv`
- `results/results_all_auto50.csv`

## Future work
- connect action drift to actual task success in simulation
- add a drift-based safety detector
- compare OpenVLA against a second VLA model
- extend from image-only evaluation to closed-loop robot runs
