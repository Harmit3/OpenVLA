import csv
import ast
import math
import random
from pathlib import Path
from statistics import mean
from collections import defaultdict

BASE = Path.home() / "openvla_local_test" / "submission_bundle"
RESULTS = BASE / "results"
ANALYSIS = BASE / "analysis"
REPORTS = BASE / "reports"
ANALYSIS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

def l2(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def bootstrap_ci(values, B=3000, seed=7):
    rng = random.Random(seed)
    vals = list(values)
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    n = len(vals)
    means = []
    for _ in range(B):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower = means[int(0.025 * len(means))]
    upper = means[int(0.975 * len(means))]
    return (sum(vals) / n, lower, upper)

def read_manual(path, group_name):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "experiment": "manual40",
                "group": group_name,
                "image_file": row["image"],
                "lang_original": row["lang_original"],
                "run_family": "manual",
                "l2_np": float(row["l2_normal_paraphrased_6d"]),
                "l2_nc": float(row["l2_normal_contradictory_6d"]),
                "l2_nn": float(row["l2_normal_neutral_6d"]),
                "cos_np": float(row["cos_normal_paraphrased_6d"]),
                "cos_nc": float(row["cos_normal_contradictory_6d"]),
                "cos_nn": float(row["cos_normal_neutral_6d"]),
            })
    return rows

def read_auto(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normal = ast.literal_eval(row["normal_action"])[:6]
            paraphrased = ast.literal_eval(row["paraphrased_action"])[:6]
            contradictory = ast.literal_eval(row["contradictory_action"])[:6]
            neutral = ast.literal_eval(row["neutral_action"])[:6]
            rows.append({
                "experiment": "auto50",
                "group": row["group"],
                "image_file": row["image_file"],
                "lang_original": row["lang_original"],
                "run_family": row["run_family"],
                "l2_np": l2(normal, paraphrased),
                "l2_nc": l2(normal, contradictory),
                "l2_nn": l2(normal, neutral),
                "cos_np": cosine(normal, paraphrased),
                "cos_nc": cosine(normal, contradictory),
                "cos_nn": cosine(normal, neutral),
            })
    return rows

manual_initial = read_manual(RESULTS / "results_initial.csv", "initial")
manual_final = read_manual(RESULTS / "results_final.csv", "final")
auto_rows = read_auto(RESULTS / "results_all_auto50.csv")

all_rows = manual_initial + manual_final + auto_rows

# 1. Summary table
summary_rows = []
for exp in ["manual40", "auto50"]:
    for group in ["initial", "final"]:
        subset = [r for r in all_rows if r["experiment"] == exp and r["group"] == group]
        if not subset:
            continue
        summary_rows.append({
            "experiment": exp,
            "group": group,
            "n": len(subset),
            "avg_l2_normal_paraphrased_6d": round(mean(r["l2_np"] for r in subset), 6),
            "avg_l2_normal_contradictory_6d": round(mean(r["l2_nc"] for r in subset), 6),
            "avg_l2_normal_neutral_6d": round(mean(r["l2_nn"] for r in subset), 6),
            "avg_cos_normal_paraphrased_6d": round(mean(r["cos_np"] for r in subset), 6),
            "avg_cos_normal_contradictory_6d": round(mean(r["cos_nc"] for r in subset), 6),
            "avg_cos_normal_neutral_6d": round(mean(r["cos_nn"] for r in subset), 6),
        })

with open(ANALYSIS / "summary_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

# 2. Bootstrap CIs
ci_rows = []
for exp in ["manual40", "auto50"]:
    for group in ["initial", "final"]:
        subset = [r for r in all_rows if r["experiment"] == exp and r["group"] == group]
        if not subset:
            continue
        for metric_name, key in [
            ("l2_normal_paraphrased_6d", "l2_np"),
            ("l2_normal_contradictory_6d", "l2_nc"),
            ("l2_normal_neutral_6d", "l2_nn"),
            ("cos_normal_paraphrased_6d", "cos_np"),
            ("cos_normal_contradictory_6d", "cos_nc"),
            ("cos_normal_neutral_6d", "cos_nn"),
        ]:
            m, lo, hi = bootstrap_ci([r[key] for r in subset])
            ci_rows.append({
                "experiment": exp,
                "group": group,
                "metric": metric_name,
                "mean": round(m, 6),
                "ci95_lower": round(lo, 6),
                "ci95_upper": round(hi, 6),
            })

with open(ANALYSIS / "bootstrap_cis.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(ci_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ci_rows)

# 3. Auto50 family breakdown
family_rows = []
families = sorted(set(r["run_family"] for r in auto_rows))
for fam in families:
    for group in ["initial", "final"]:
        subset = [r for r in auto_rows if r["run_family"] == fam and r["group"] == group]
        if not subset:
            continue
        family_rows.append({
            "run_family": fam,
            "group": group,
            "n": len(subset),
            "avg_l2_np": round(mean(r["l2_np"] for r in subset), 6),
            "avg_l2_nc": round(mean(r["l2_nc"] for r in subset), 6),
            "avg_l2_nn": round(mean(r["l2_nn"] for r in subset), 6),
        })

with open(ANALYSIS / "auto50_family_breakdown.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
    writer.writeheader()
    writer.writerows(family_rows)

# 4. Top failure cases
top_cases = []
for r in all_rows:
    top_cases.append({
        "experiment": r["experiment"],
        "group": r["group"],
        "image_file": r["image_file"],
        "run_family": r["run_family"],
        "lang_original": r["lang_original"],
        "l2_np": round(r["l2_np"], 6),
        "l2_nc": round(r["l2_nc"], 6),
        "l2_nn": round(r["l2_nn"], 6),
        "max_drift": round(max(r["l2_np"], r["l2_nc"], r["l2_nn"]), 6),
    })

top_cases.sort(key=lambda x: x["max_drift"], reverse=True)
with open(ANALYSIS / "top_failure_cases.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(top_cases[0].keys()))
    writer.writeheader()
    writer.writerows(top_cases[:40])

# 5. Submission checklist
with open(REPORTS / "submission_checklist.txt", "w", encoding="utf-8") as f:
    f.write(
"""SUBMISSION CHECKLIST

CODE
- test_openvla_PATCHED_WORKING.py
- run_metadata_experiment.py
- extract_bridge_sample_clean.py
- fill_prompts_from_family.py
- run_bridge_batch_from_metadata.py
- requirements-working.txt

METADATA
- metadata_manual.csv
- metadata_sampled_filled.csv

RESULTS
- results_initial.csv
- results_final.csv
- results_all_auto50.csv
- initial_run.log
- final_run.log
- auto50_run.log

ANALYSIS
- summary_table.csv
- bootstrap_cis.csv
- auto50_family_breakdown.csv
- top_failure_cases.csv

WRITING
- Abstract
- Introduction
- Method
- Results
- Limitations
- Conclusion
"""
    )

# 6. Experiment diary template
with open(REPORTS / "experiment_diary_template.txt", "w", encoding="utf-8") as f:
    f.write(
"""EXPERIMENT DIARY TEMPLATE

Day 1
- Set up local OpenVLA environment in WSL
- Installed required package versions
- Verified GPU and 4-bit model loading
- Encountered one-token attention-mask bug in predict_action
- Patched attention-mask path and confirmed successful inference

Day 2
- Ran pilot tests on dummy image and real image
- Built prompt comparison scripts
- Ran initial prompt sensitivity pilot on small image set
- Observed prompt-induced action drift and some invariant cases

Day 3
- Built manual 40-trajectory BridgeData subset
- Created task-specific metadata_manual.csv
- Ran manual initial and final experiments
- Summarized drift metrics and invariance counts

Day 4
- Downloaded and extracted scripted BridgeData dataset
- Sampled 50 trajectories automatically
- Built metadata_sampled_filled.csv from run_family
- Ran auto50 experiment on 100 image rows
- Generated summary tables and failure-case reports

Notes
- Record any errors, fixes, and reruns here
"""
    )

# 7. Methods notes
with open(REPORTS / "methods_notes.txt", "w", encoding="utf-8") as f:
    f.write(
"""METHODS NOTES

Model
- OpenVLA-7B in 4-bit mode on local WSL environment
- Patched predict_action to prepend a 256-token image attention mask

Data
- Experiment 1: manual 40-trajectory BridgeData subset
- Experiment 2: automatic 50-trajectory sample from scripted BridgeData split

Prompt conditions
- normal
- paraphrased
- contradictory
- neutral

Evaluation
- first 6 action dimensions only
- L2 distance from normal prompt
- cosine similarity to normal prompt
- counts of near-invariant and high-drift cases

Main hypothesis
- paraphrases preserve behavior best
- contradictory and neutral prompts increase action drift
- final states are more fragile than initial states
"""
    )

print("Done.")
print("Created:")
for p in [
    ANALYSIS / "summary_table.csv",
    ANALYSIS / "bootstrap_cis.csv",
    ANALYSIS / "auto50_family_breakdown.csv",
    ANALYSIS / "top_failure_cases.csv",
    REPORTS / "submission_checklist.txt",
    REPORTS / "experiment_diary_template.txt",
    REPORTS / "methods_notes.txt",
]:
    print("-", p)
