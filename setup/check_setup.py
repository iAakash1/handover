"""One command to see if this machine is ready: environments, models, datasets and GPU.

    python setup/check_setup.py

Everything marked MISSING comes with the command that fixes it.
"""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok_all = True


def report(ok, what, fix=""):
    global ok_all
    ok_all &= ok
    print(f"  {'ok     ' if ok else 'MISSING'}  {what}" + ("" if ok else f"\n           fix: {fix}"))


print("Environments (.venvs/)")
for env in ("keras", "soynet", "vlm", "florence", "tih"):
    ready = os.path.join(ROOT, ".venvs", env, ".ready")  # written only after a complete install
    report(os.path.exists(ready), env, f"python setup/setup_envs.py {env}")

print("Trained models")
files = json.load(open(os.path.join(ROOT, "setup", "models_manifest.json")))["files"]
missing = [f for f in files if not (os.path.exists(os.path.join(ROOT, f["path"]))
                                    and os.path.getsize(os.path.join(ROOT, f["path"])) == f["bytes"])]
report(not missing, f"{len(files) - len(missing)}/{len(files)} model files present", "python setup/download_models.py")

print("Datasets (not in git; see docs/datasets.md)")
DATA = {
    "PLANTVILLAGE_DIR": r"C:\Users\Admin\Desktop\VLM\PlantVillage",
    "TOMATO_DATASET_DIR": r"C:\Users\Admin\Desktop\iitb_fellowship\data\raw\PlantVillage",
    "TOMATO_SPLIT_DIR": r"C:\Users\Admin\Desktop\iitb_fellowship\data",
    "SOYNET_DIR": r"C:\Users\Admin\Desktop\latest\SoyNet\Preprocessing_SoyNet_Data\Camera Clicks_256_256",
    "TIH dataset (--dataset-root)": r"C:\Users\Admin\Desktop\TIH_IITB_August_2026\Tomato_12thAug26\Dataset",
}
for var, default in DATA.items():
    path = os.environ.get(var, default) if var.isupper() else default
    report(os.path.isdir(path), f"{var} = {path}", f"copy the dataset there or set {var.split()[0]} (docs/datasets.md)")
report(os.path.isdir(os.path.join(ROOT, "projects", "08_plantvlm_expert_captions", "1stJuly", "Dataset")),
       "project 08 images (1stJuly/Dataset)", "copy the 30 expert-captioned images into projects/08_plantvlm_expert_captions/1stJuly/Dataset")

print("GPU")
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    report(bool(out), out or "no NVIDIA GPU found", "VLM projects need an NVIDIA GPU with >= 16 GB (24 GB used here)")
except FileNotFoundError:
    report(False, "nvidia-smi not found", "install the NVIDIA driver")

print("\nREADY" if ok_all else "\nSome items are missing - fix them with the commands above.")
