# One page to try every model we trained on your own photo.
#
#   .venvs\vlm\Scripts\python -m streamlit run demo_app.py
#
# Each model runs in its own environment (Keras, SoyNet, VLM, TIH...), so this page doesn't import any of them:
# it calls the same predict scripts you would run by hand, with the right .venvs python, and shows what they print.
# The CNNs answer in a few seconds; a VLM needs about a minute the first time (it loads a 7B model).
# For the full apps (CLIP check, GradCAM, history, training) use run_app.py.

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
BIN = "Scripts" if os.name == "nt" else "bin"


def env_python(env):
    return str(ROOT / ".venvs" / env / BIN / ("python.exe" if os.name == "nt" else "python"))


TIH_CFG = "configs/train/qwen25vl_tih_tomato_moreaccurate.yaml"
TIH_ADAPTER = "checkpoints/tih_tomato_moreaccurate"

# name -> (group, env, working folder, arguments before --image, one line about it)
MODELS = {
    "TIH tomato, 11 classes · Qwen2.5-VL QLoRA (Sep 2026, ICAIN paper)": (
        "Latest", "tih", "projects/10_tih_tomato_qwen_qlora",
        ["-m", "plantdx.cli", "company", "predict", "--config", TIH_CFG, "--adapter", TIH_ADAPTER],
        "92.76% on the 649-image TIH test set. Leaves and fruits: pests, blights, healthy."),
    "PlantVillage · Qwen2.5-VL LoRA (Apr 2026)": (
        "Latest", "vlm", "evaluation", ["vlm_predict.py", "--model", "qwen_april"],
        "98.50% on 200 PlantVillage images. Writes a full diagnosis: plant, condition, severity, symptoms."),
    "PlantVillage 15 classes · PlantDx Qwen2.5-VL (May 2026)": (
        "Latest", "vlm", "evaluation", ["vlm_predict.py", "--model", "plantdx"],
        "98.00%. Tomato, potato and pepper leaves."),
    "Tomato · ConvNeXt-Tiny, PyTorch (Jan 2026)": (
        "CNNs", "vlm", "projects/02_cnn_pytorch_convnext", ["predict.py"],
        "99.92% on the tomato test split. Our best CNN; label only."),
    "Tomato · ConvNeXt-Tiny, Keras (Jan 2026)": (
        "CNNs", "keras", "projects/01_cnn_keras_tomato", ["predict.py", "--model", "convnext"], "96.15%."),
    "Tomato · EfficientNetB3, Keras (Jan 2026)": (
        "CNNs", "keras", "projects/01_cnn_keras_tomato", ["predict.py", "--model", "efficientnet"], "93.59%."),
    "Tomato · MobileNetV2, Keras (Jan 2026)": (
        "CNNs", "keras", "projects/01_cnn_keras_tomato", ["predict.py", "--model", "mobilenet"], "90.90%."),
    "Tomato · custom CNN, Keras (Dec 2025)": (
        "CNNs", "keras", "projects/01_cnn_keras_tomato", ["predict.py", "--model", "cnn"],
        "Our first CNN, 32x32 input."),
    "Soybean healthy/diseased · EfficientNetB3 (Jan 2026)": (
        "CNNs", "soynet", "projects/03_soybean_soynet", ["predict_soynet.py", "--model", "efficientnet_b3"],
        "93.93% on the SoyNet test split."),
    "Soybean healthy/diseased · ResNet50 (Jan 2026)": (
        "CNNs", "soynet", "projects/03_soybean_soynet", ["predict_soynet.py", "--model", "resnet50"], "91.61%."),
    "PlantVillage · BLIP-2 LoRA (Mar 2026)": (
        "Earlier VLMs", "vlm", "evaluation", ["vlm_predict.py", "--model", "blip2_march"], "89.00%. Short captions."),
    "PlantVillage · BLIP-2 LoRA, June re-run": (
        "Earlier VLMs", "vlm", "evaluation", ["vlm_predict.py", "--model", "blip2_june"], "89.00%."),
    "Expert captions · PlantVLM Qwen2.5-VL (Aug 2026)": (
        "Earlier VLMs", "vlm", "evaluation", ["vlm_predict.py", "--model", "plantvlm"],
        "Trained on only 30 images: 2 of 6 right. See failed_experiments/."),
    "PlantVillage · LLaVA-1.6 LoRA (Mar 2026)": (
        "Didn't work out", "vlm", "evaluation", ["vlm_predict.py", "--model", "llava"],
        "2.00%: long generic answers. See failed_experiments/02."),
    "Qwen2.5-VL with no fine-tuning": (
        "Didn't work out", "vlm", "evaluation", ["vlm_predict.py", "--model", "qwen_base"],
        "8.50%: shows why fine-tuning was needed."),
    "Florence-2 with no fine-tuning": (
        "Didn't work out", "florence", "evaluation", ["vlm_predict.py", "--model", "florence"],
        "0%: describes the scene, never the disease."),
}

NOISE = ("warn", "Loading", "HF_TOKEN", "it/s]", "oneDNN", "tensorflow", "cuda_", "rebuild TensorFlow",
         "LoraConfig", "upgrade the PEFT", "I0000", "W0000", "Fetching")


def run(name, image_path):
    _, env, folder, args, _ = MODELS[name]
    py = env_python(env)
    if not os.path.exists(py):
        return False, f"The '{env}' environment is missing. Run: python setup/setup_envs.py"
    cmd = [py, *args, "--image", image_path]
    proc = subprocess.run(cmd, cwd=ROOT / folder, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = [l for l in (proc.stdout + "\n" + proc.stderr).splitlines()
             if l.strip() and not any(n in l for n in NOISE)]
    if proc.returncode != 0:
        return False, "\n".join(lines[-25:])
    return True, "\n".join(l for l in proc.stdout.splitlines() if l.strip() and not any(n in l for n in NOISE))


st.set_page_config(page_title="Crop disease models", page_icon="🌿", layout="wide")
st.title("🌿 All our crop-disease models")
st.caption("Chanakya Fellowship, TIH IIT Bombay · Dec 2025 – Sep 2026 · upload a photo, pick a model")

groups = ["Latest", "CNNs", "Earlier VLMs", "Didn't work out"]
with st.sidebar:
    group = st.radio("Models", groups, captions=["⭐ start here", "fast, label only", "", "kept for comparison"])
    names = [n for n, m in MODELS.items() if m[0] == group]
    name = st.selectbox("Model", names)
    st.info(MODELS[name][4])
    st.markdown("**Full apps** (CLIP check, GradCAM, history):\n\n`python run_app.py`")

photo = st.file_uploader("Leaf or fruit photo", type=["jpg", "jpeg", "png"])
if photo:
    left, right = st.columns([1, 2])
    left.image(photo, use_container_width=True)
    if right.button(f"Diagnose with: {name}", type="primary"):
        suffix = Path(photo.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(photo.getvalue())
        with right, st.spinner("Running... (VLMs take about a minute the first time)"):
            ok, out = run(name, tmp.name)
        os.unlink(tmp.name)
        (right.success if ok else right.error)("Result" if ok else "The model could not run")
        right.code(out or "(no output)", language=None)
else:
    st.write("Upload a photo to start. Tomato models expect a tomato leaf (the TIH model also takes fruits); "
             "SoyNet expects a soybean leaf.")
