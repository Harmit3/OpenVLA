import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import math
import types
import argparse
from statistics import mean

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument("--group", choices=["initial", "final", "all"], default="all")
parser.add_argument("--limit", type=int, default=0, help="0 means all")
args = parser.parse_args()

BASE = os.path.expanduser("~/openvla_local_test/bridge_v2")
METADATA = os.path.join(BASE, "metadata_manual.csv")
INITIAL_DIR = os.path.join(BASE, "images", "initial")
FINAL_DIR = os.path.join(BASE, "images", "final")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

checkpoint = "openvla/openvla-7b"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model in 4-bit...")
model = AutoModelForVision2Seq.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

_orig_predict_action = model.predict_action

def _patched_predict_action(self, input_ids=None, attention_mask=None,
                            pixel_values=None, unnorm_key=None, **kwargs):
    num_img_tokens = 256
    img_mask = torch.ones(
        attention_mask.shape[0],
        num_img_tokens,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    attention_mask = torch.cat([img_mask, attention_mask], dim=1)
    return _orig_predict_action.__func__(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        unnorm_key=unnorm_key,
        **kwargs
    )

model.predict_action = types.MethodType(_patched_predict_action, model)

def fix_prompt(s: str) -> str:
    return s.replace("\\n", "\n").strip()

def get_action(img_path: str, prompt: str):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    inputs = processor(prompt, img, return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].to("cuda:0")
    inputs["attention_mask"] = inputs["attention_mask"].to("cuda:0")
    inputs["pixel_values"] = inputs["pixel_values"].to("cuda:0", dtype=torch.float16)
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return np.array(action, dtype=np.float32)

def l2(a, b):
    return float(np.linalg.norm(a - b))

def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

rows = []
with open(METADATA, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if args.group != "all":
            fname = row["initial_image"] if args.group == "initial" else row["final_image"]
            rows.append({
                "trajectory_id": row["trajectory_id"],
                "group": args.group,
                "image": fname,
                "lang_original": row["lang_original"],
                "prompt_normal": fix_prompt(row["prompt_normal"]),
                "prompt_paraphrased": fix_prompt(row["prompt_paraphrased"]),
                "prompt_contradictory": fix_prompt(row["prompt_contradictory"]),
                "prompt_neutral": fix_prompt(row["prompt_neutral"]),
            })
        else:
            rows.append({
                "trajectory_id": row["trajectory_id"],
                "group": "initial",
                "image": row["initial_image"],
                "lang_original": row["lang_original"],
                "prompt_normal": fix_prompt(row["prompt_normal"]),
                "prompt_paraphrased": fix_prompt(row["prompt_paraphrased"]),
                "prompt_contradictory": fix_prompt(row["prompt_contradictory"]),
                "prompt_neutral": fix_prompt(row["prompt_neutral"]),
            })
            rows.append({
                "trajectory_id": row["trajectory_id"],
                "group": "final",
                "image": row["final_image"],
                "lang_original": row["lang_original"],
                "prompt_normal": fix_prompt(row["prompt_normal"]),
                "prompt_paraphrased": fix_prompt(row["prompt_paraphrased"]),
                "prompt_contradictory": fix_prompt(row["prompt_contradictory"]),
                "prompt_neutral": fix_prompt(row["prompt_neutral"]),
            })

if args.limit > 0:
    rows = rows[:args.limit]

print(f"Running {len(rows)} samples for group={args.group}")

results = []
for i, row in enumerate(rows, start=1):
    img_dir = INITIAL_DIR if row["group"] == "initial" else FINAL_DIR
    img_path = os.path.join(img_dir, row["image"])

    print(f"[{i}/{len(rows)}] {row['group']} | {row['image']}")

    actions = {}
    for key in ["normal", "paraphrased", "contradictory", "neutral"]:
        prompt = row[f"prompt_{key}"]
        actions[key] = get_action(img_path, prompt)
        print(f"  {key}: {actions[key]}")

    result = {
        "trajectory_id": row["trajectory_id"],
        "group": row["group"],
        "image": row["image"],
        "lang_original": row["lang_original"],
        "normal_action": actions["normal"].tolist(),
        "paraphrased_action": actions["paraphrased"].tolist(),
        "contradictory_action": actions["contradictory"].tolist(),
        "neutral_action": actions["neutral"].tolist(),

        "l2_normal_paraphrased_6d": l2(actions["normal"][:6], actions["paraphrased"][:6]),
        "l2_normal_contradictory_6d": l2(actions["normal"][:6], actions["contradictory"][:6]),
        "l2_normal_neutral_6d": l2(actions["normal"][:6], actions["neutral"][:6]),

        "cos_normal_paraphrased_6d": cosine(actions["normal"][:6], actions["paraphrased"][:6]),
        "cos_normal_contradictory_6d": cosine(actions["normal"][:6], actions["contradictory"][:6]),
        "cos_normal_neutral_6d": cosine(actions["normal"][:6], actions["neutral"][:6]),
    }
    results.append(result)

if args.group == "initial":
    out_csv = os.path.join(RESULTS_DIR, "results_initial.csv")
elif args.group == "final":
    out_csv = os.path.join(RESULTS_DIR, "results_final.csv")
else:
    out_csv = os.path.join(RESULTS_DIR, "results_all.csv")

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to: {out_csv}")

def summarize(group_rows):
    avg_np = mean(r["l2_normal_paraphrased_6d"] for r in group_rows)
    avg_nc = mean(r["l2_normal_contradictory_6d"] for r in group_rows)
    avg_nn = mean(r["l2_normal_neutral_6d"] for r in group_rows)
    print("avg_l2_normal_paraphrased_6d =", round(avg_np, 6))
    print("avg_l2_normal_contradictory_6d =", round(avg_nc, 6))
    print("avg_l2_normal_neutral_6d =", round(avg_nn, 6))

print("\nSummary:")
summarize(results)
