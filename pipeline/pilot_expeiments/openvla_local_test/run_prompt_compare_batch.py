import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import math
import types
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

checkpoint = "openvla/openvla-7b"
image_dir = "./images"
output_csv = "prompt_compare_results.csv"

processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

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

prompts = {
    "normal": "In: What action should the robot take to pick up the silver cup from the sink area?\nOut:",
    "paraphrased": "In: What should the robot do to grasp the silver cup near the sink?\nOut:",
    "contradictory": "In: What action should the robot take to avoid touching the silver cup and move away from it?\nOut:",
}

def get_action(img_path, prompt):
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

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for fname in image_files:
    path = os.path.join(image_dir, fname)
    print(f"Processing {fname}...")

    actions = {}
    for name, prompt in prompts.items():
        actions[name] = get_action(path, prompt)
        print(f"  {name}: {actions[name]}")

    row = {
        "image": fname,
        "normal_action": actions["normal"].tolist(),
        "paraphrased_action": actions["paraphrased"].tolist(),
        "contradictory_action": actions["contradictory"].tolist(),
        "l2_normal_paraphrased": l2(actions["normal"], actions["paraphrased"]),
        "l2_normal_contradictory": l2(actions["normal"], actions["contradictory"]),
        "cosine_normal_paraphrased": cosine(actions["normal"], actions["paraphrased"]),
        "cosine_normal_contradictory": cosine(actions["normal"], actions["contradictory"]),
    }
    rows.append(row)

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved results to {output_csv}")
