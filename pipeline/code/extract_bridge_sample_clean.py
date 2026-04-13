import csv
import random
import shutil
from pathlib import Path

DATASET_ROOT = Path.home() / "bridge_raw" / "scripted_raw" / "scripted_raw"
OUT_ROOT = Path.home() / "openvla_local_test" / "bridge_v2"
SAMPLE_N = 50
SEED = 7

random.seed(SEED)

initial_dir = OUT_ROOT / "images" / "initial"
final_dir = OUT_ROOT / "images" / "final"
initial_dir.mkdir(parents=True, exist_ok=True)
final_dir.mkdir(parents=True, exist_ok=True)

def natural_sort_key(path_obj):
    name = path_obj.stem
    parts = name.split("_")
    try:
        return int(parts[-1])
    except Exception:
        return name

def collect_records():
    records = []
    for image_dir in DATASET_ROOT.rglob("images0"):
        traj_dir = image_dir.parent
        frames = sorted(
            [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=natural_sort_key
        )
        if len(frames) < 2:
            continue

        parts = traj_dir.parts
        # Example:
        # ... / 2023-05-25_bin_pnp / 2023-05-26_10-40-38 / raw / traj_group0 / traj1
        try:
            run_family = parts[-5]
            run_stamp = parts[-4]
            traj_group = parts[-2]
            traj_name = parts[-1]
        except Exception:
            continue

        records.append({
            "run_family": run_family,
            "run_stamp": run_stamp,
            "traj_group": traj_group,
            "traj_name": traj_name,
            "traj_dir": str(traj_dir),
            "first_frame": str(frames[0]),
            "last_frame": str(frames[-1]),
            "num_frames": len(frames),
        })
    return records

records = collect_records()
print(f"Found {len(records)} trajectories under {DATASET_ROOT}")

if not records:
    raise SystemExit("No trajectories found.")

sample_n = min(SAMPLE_N, len(records))
sampled = random.sample(records, sample_n)

rows = []
for i, rec in enumerate(sampled, start=1):
    tid = f"traj{i:03d}"
    initial_name = f"{tid}_initial.jpg"
    final_name = f"{tid}_final.jpg"

    shutil.copy2(rec["first_frame"], initial_dir / initial_name)
    shutil.copy2(rec["last_frame"], final_dir / final_name)

    # Prompt scaffold left intentionally editable by you later
    rows.append({
        "trajectory_id": f"{i:03d}",
        "run_family": rec["run_family"],
        "run_stamp": rec["run_stamp"],
        "traj_group": rec["traj_group"],
        "traj_name": rec["traj_name"],
        "initial_image": initial_name,
        "final_image": final_name,
        "num_frames": rec["num_frames"],
        "lang_original": "",
        "prompt_normal": "",
        "prompt_paraphrased": "",
        "prompt_contradictory": "",
        "prompt_neutral": "In: What action should the robot take?\nOut:",
    })

out_csv = OUT_ROOT / "metadata_sampled_scaffold.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_csv}")
print(f"Initial images copied: {len(list(initial_dir.glob('*.jpg')))}")
print(f"Final images copied: {len(list(final_dir.glob('*.jpg')))}")
