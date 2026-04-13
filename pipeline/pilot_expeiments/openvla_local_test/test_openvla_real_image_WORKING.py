import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import types
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

checkpoint = "openvla/openvla-7b"

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

image_path = "test.jpg"
prompt = "In: What action should the robot take to pick up the object?\nOut:"

img = Image.open(image_path).convert("RGB").resize((224, 224))

inputs = processor(prompt, img, return_tensors="pt")
inputs["input_ids"] = inputs["input_ids"].to("cuda:0")
inputs["attention_mask"] = inputs["attention_mask"].to("cuda:0")
inputs["pixel_values"] = inputs["pixel_values"].to("cuda:0", dtype=torch.float16)

action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print("Predicted action:", action)
