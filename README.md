# OpenVLA Prompt Sensitivity Project

This repository packages a completed research project on prompt sensitivity in OpenVLA-7B using BridgeData images.

## What this project asks
If the scene stays the same but the instruction wording changes, does the predicted robot action stay stable?

## Completed experiment branches
1. **manual40**: 40 manually curated BridgeData trajectories, with initial and final images (80 images total).
2. **auto50**: 50 automatically sampled trajectories from the scripted BridgeData archive (100 image rows total).

## Prompt variants
- normal
- paraphrased
- contradictory
- neutral

## Main findings
Across both experiments:
- Paraphrases stayed closest to the normal prompt.
- Contradictory and neutral prompts caused larger action drift.
- Final-state images were generally more fragile than initial-state images.

## Key summary numbers
| experiment | group | n | l2_para | l2_contra | l2_neutral |
|---|---:|---:|---:|---:|---:|
| manual40 | initial | 40 | 0.028126 | 0.042365 | 0.045330 |
| manual40 | final | 40 | 0.043141 | 0.059920 | 0.070772 |
| auto50 | initial | 50 | 0.056164 | 0.069716 | 0.073405 |
| auto50 | final | 50 | 0.080668 | 0.093857 | 0.082103 |

## Repository structure
- `code/`: core scripts used to run and analyze the experiments
- `metadata/`: manual and auto50 metadata files
- `results/`: raw result CSVs and logs
- `analysis/`: summary tables, bootstrap confidence intervals, family breakdown, failure cases
- `docs/`: project report, defense handbook, and paper draft material
- `pilot_experiments/`: Experiments ran before moving onto machine in colab and smoke tests
- `env`: env files from plateform
- `figures`: final result depicted in visual forms

## Quick start
1. Read `docs/OpenVLA_Professor_Final_Report.docx` for the full story.
2. Read `docs/OpenVLA_Defense_Master_Handbook.docx` for defense prep.
3. Read `docs/OpenVLA_CoRL_Final_Draft.docx` for the paper draft.

## Important note on the dataset
This repo does **not** include the full 30 GB scripted BridgeData archive. Instead, it includes the metadata and result artifacts needed to understand and reproduce the reported study structure. Users should download the raw dataset separately from the official source when needed.

## Core scripts
- `test_openvla_PATCHED_WORKING.py`
- `run_metadata_experiment.py`
- `extract_bridge_sample_clean.py`
- `fill_prompts_from_family.py`
- `run_bridge_batch_from_metadata.py`
- `build_analysis_artifacts.py`

## Citation-ready project references
- OpenVLA (CoRL 2025)
- BridgeData V2 (CoRL 2023)
- LeRobot (ICLR 2026)
- VLA survey (arXiv 2026)
