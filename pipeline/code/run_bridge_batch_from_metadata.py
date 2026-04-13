import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

CHECKPOINT = "openvla/openvla-7b"
BASE_DIR = Path.home() / "openvla_local_test" / "bridge_v2"
METADATA_CSV = BASE_DIR / "metadata_sampled_filled.csv"
OUT_CSV = BASE_DIR / "results" / "results_all_auto50.csv"


# smoke test first
LIMIT_TRAJECTORIES = None

processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForVision2Seq.from_pretrained(
    CHECKPOINT,
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

def run_action(img_path, prompt):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    inputs = processor(prompt, img, return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].to("cuda:0")
    inputs["attention_mask"] = inputs["attention_mask"].to("cuda:0")
    inputs["pixel_values"] = inputs["pixel_values"].to("cuda:0", dtype=torch.float16)
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return np.array(action, dtype=np.float32)

trajectory_rows = []
with open(METADATA_CSV, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        trajectory_rows.append(row)

if LIMIT_TRAJECTORIES is not None:
    trajectory_rows = trajectory_rows[:LIMIT_TRAJECTORIES]

rows = []
for row in trajectory_rows:
    rows.append({
        **row,
        "group": "initial",
        "image_file": row["initial_image"],
        "image_path": str(BASE_DIR / "images" / "initial" / row["initial_image"]),
    })
    rows.append({
        **row,
        "group": "final",
        "image_file": row["final_image"],
        "image_path": str(BASE_DIR / "images" / "final" / row["final_image"]),
    })

print(f"Running {len(rows)} image rows from {len(trajectory_rows)} trajectories")

fieldnames = [
    "trajectory_id", "group", "run_family", "run_stamp", "traj_group", "traj_name",
    "image_file", "lang_original",
    "normal_action", "paraphrased_action", "contradictory_action", "neutral_action"
]

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(rows, start=1):
        img_path = Path(row["image_path"])
        print(f"[{i}/{len(rows)}] traj={row['trajectory_id']} group={row['group']} file={img_path.name}")

        normal = run_action(img_path, row["prompt_normal"]).tolist()
        paraphrased = run_action(img_path, row["prompt_paraphrased"]).tolist()
        contradictory = run_action(img_path, row["prompt_contradictory"]).tolist()
        neutral = run_action(img_path, row["prompt_neutral"]).tolist()

        out_row = {
            "trajectory_id": row["trajectory_id"],
            "group": row["group"],
            "run_family": row["run_family"],
            "run_stamp": row["run_stamp"],
            "traj_group": row["traj_group"],
            "traj_name": row["traj_name"],
            "image_file": row["image_file"],
            "lang_original": row["lang_original"],
            "normal_action": normal,
            "paraphrased_action": paraphrased,
            "contradictory_action": contradictory,
            "neutral_action": neutral,
        }
        writer.writerow(out_row)
        f.flush()

print(f"Saved {OUT_CSV}")
