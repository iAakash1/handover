# Start any of our Streamlit apps with the right environment, without remembering paths.
#
#   python run_app.py          (shows the menu)
#   python run_app.py 1        (starts app 1 straight away)
#
# Needs only plain Python; the apps themselves run in .venvs (run setup/setup_envs.py first).
# Stop an app with Ctrl+C in this window.

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = "Scripts" if os.name == "nt" else "bin"

# (title, env, folder, script)
APPS = [
    ("* TIH tomato, 11 classes - Qwen2.5-VL QLoRA, Sep 2026 (ICAIN paper) - diagnose, training, evaluation",
     "tih", "projects/10_tih_tomato_qwen_qlora", "streamlit_app.py"),
    ("* Crop disease app - Qwen2.5-VL LoRA, Apr 2026 - CLIP leaf check, GradCAM heat-map",
     "vlm", "projects/06_qwen25vl_plantvillage/crop_disease_app", "app.py"),
    ("* All models on one page - CNNs, SoyNet, every VLM",
     "vlm", ".", "demo_app.py"),
    ("PlantDx, 15 PlantVillage classes - May 2026",
     "vlm", "projects/07_plantdx_plantvillage", "app/05_app.py"),
    ("Qwen2.5-VL April demo (the 98.50% model)",
     "vlm", "projects/06_qwen25vl_plantvillage/qwen", "05_app.py"),
    ("BLIP-2 June re-run demo",
     "vlm", "projects/04_llava_blip2/blip2_rerun_june", "05_app.py"),
    ("BLIP-2 March demo",
     "vlm", "projects/04_llava_blip2/blip2_march", "05_app.py"),
]


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    if choice is None:
        print("\nWhich app?\n")
        for i, (title, env, _, _) in enumerate(APPS, 1):
            print(f"  {i}. {title}   [{env} env]")
        choice = input("\nNumber: ").strip()
    try:
        title, env, folder, script = APPS[int(choice) - 1]
    except (ValueError, IndexError):
        sys.exit(f"Pick a number from 1 to {len(APPS)}.")

    py = ROOT / ".venvs" / env / BIN / ("python.exe" if os.name == "nt" else "python")
    if not py.exists():
        sys.exit(f"The '{env}' environment is missing. Run first:  python setup/setup_envs.py")
    print(f"\nStarting: {title}\nIt opens in your browser; the address is printed below. Ctrl+C here to stop.\n")
    subprocess.run([str(py), "-m", "streamlit", "run", script], cwd=ROOT / folder)


if __name__ == "__main__":
    main()
