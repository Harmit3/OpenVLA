import csv
from pathlib import Path

IN_CSV = Path.home() / "openvla_local_test" / "bridge_v2" / "metadata_sampled_scaffold.csv"
OUT_CSV = Path.home() / "openvla_local_test" / "bridge_v2" / "metadata_sampled_filled.csv"

def infer_task(run_family: str) -> str:
    rf = run_family.lower()

    if "bin_pnp" in rf:
        return "pick up the object and place it in the bin"
    if "pnp_utensils" in rf:
        return "pick up the utensil"
    if "rigid_objects" in rf and "pnp" in rf:
        return "pick up the rigid object"
    if "soft_toys" in rf and "pnp" in rf:
        return "pick up the soft toy"
    if "many_objects_in_env" in rf:
        return "pick up the target object among many objects"
    if "pnp_objects" in rf:
        return "pick up the object"
    if "sweep" in rf:
        return "sweep the object across the surface"
    if "tabletop" in rf:
        return "manipulate the object on the tabletop"

    return "manipulate the target object"

def make_prompts(task: str):
    normal = f"In: What action should the robot take to {task}?\nOut:"
    paraphrased = f"In: What should the robot do to {task}?\nOut:"
    contradictory = f"In: What action should the robot take to avoid the target object and move away instead of trying to {task}?\nOut:"
    neutral = "In: What action should the robot take?\nOut:"
    return normal, paraphrased, contradictory, neutral

rows = []
with open(IN_CSV, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        task = infer_task(row["run_family"])
        normal, paraphrased, contradictory, neutral = make_prompts(task)
        row["lang_original"] = task
        row["prompt_normal"] = normal
        row["prompt_paraphrased"] = paraphrased
        row["prompt_contradictory"] = contradictory
        row["prompt_neutral"] = neutral
        rows.append(row)

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUT_CSV}")
print(f"Rows: {len(rows)}")
