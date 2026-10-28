# 05_app.py
# Streamlit demo — Qwen2.5-VL-7B PlantVillage Disease Detector
# Features: image upload, OOD detection toggle, structured diagnosis output
#
# Run: streamlit run 05_app.py
# Requires: streamlit>=1.40.0,<1.45.0

import streamlit as st
import torch
import json
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPProcessor,
    CLIPModel,
)
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────
ADAPTER_PATH    = "outputs/qwen25vl_plantvillage/best"
BASE_MODEL      = "Qwen/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
OOD_THRESHOLD   = 0.22
MAX_NEW_TOKENS  = 160
EVAL_REPORT     = "evaluation_report.json"

# Severity to color mapping
SEVERITY_COLORS = {
    "None":               "#22c55e",
    "Mild":               "#84cc16",
    "Mild to Moderate":   "#eab308",
    "Moderate":           "#f97316",
    "Moderate to Severe": "#ef4444",
    "Severe":             "#991b1b",
}

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantVillage Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1a0f; }
    .stApp { background-color: #0f1a0f; color: #e8f5e9; }
    h1 { color: #4ade80; font-family: 'Courier New', monospace; }
    h2, h3 { color: #86efac; }
    .diagnosis-box {
        background: #1a2e1a;
        border: 1px solid #4ade80;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background: #1a2e1a;
        border-left: 4px solid #4ade80;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.4rem 0;
    }
    .ood-warning {
        background: #3b1a1a;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        color: #fca5a5;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 999px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .stButton > button {
        background-color: #16a34a;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        cursor: pointer;
    }
    .stButton > button:hover { background-color: #15803d; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────────

@st.cache_resource
def load_vlm():
    """Load Qwen2.5-VL with 4-bit QLoRA adapter."""
    if not Path(ADAPTER_PATH).exists():
        return None, None, "Adapter not found. Run 02_finetune_qwen25vl.py first."

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        return model, processor, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_clip():
    try:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to("cuda")
        clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model.eval()
        return clip_model, clip_proc
    except Exception:
        return None, None


# ── Inference functions ───────────────────────────────────────────────────

def run_ood(image: Image.Image, clip_model, clip_proc, threshold: float) -> Tuple[bool, float]:
    inputs = clip_proc(
        text=["a photo of a plant leaf"],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    with torch.no_grad():
        out     = clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        sim     = (img_emb * txt_emb).sum().item()
    return sim < threshold, round(sim, 4)


def run_inference(image: Image.Image, model, processor, instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": instruction},
            ],
        }
    ]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=6,
            early_stopping=True,
        )
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def parse_diagnosis(text: str) -> dict:
    """Parse structured caption into fields."""
    import re
    fields = {}
    for field in ["Plant", "Condition", "Severity", "Pathogen"]:
        m = re.search(rf"{field}[:\s]+([^.]+)", text, re.IGNORECASE)
        if m:
            fields[field] = m.group(1).strip().rstrip(".")
    # Symptoms = everything after the 4th field (heuristic)
    parts = text.split(". ")
    if len(parts) > 4:
        fields["Symptoms"] = ". ".join(parts[4:]).strip()
    elif len(parts) > 1:
        fields["Symptoms"] = parts[-1].strip()
    return fields


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    enable_ood = st.toggle("OOD Detection (CLIP)", value=True)
    ood_thresh = st.slider(
        "OOD Threshold", min_value=0.10, max_value=0.40,
        value=OOD_THRESHOLD, step=0.01,
        help="Images with CLIP similarity below this are flagged as out-of-distribution"
    )

    st.markdown("---")
    st.markdown("**Custom instruction**")
    instruction = st.text_area(
        "",
        value=(
            "Analyze this plant leaf image and provide a detailed diagnosis. "
            "Include: plant name, disease or health condition, severity level, "
            "causal pathogen, and visible symptoms."
        ),
        height=120,
    )

    st.markdown("---")
    st.markdown("**Generation settings**")
    n_beams    = st.slider("Beam size",       2, 8, 4)
    rep_pen    = st.slider("Repetition pen.", 1.0, 4.0, 2.5, step=0.1)

    st.markdown("---")
    # Eval metrics panel
    if Path(EVAL_REPORT).exists():
        with open(EVAL_REPORT) as f:
            rpt = json.load(f)
        st.markdown("## 📊 Eval Metrics")
        cq = rpt.get("caption_quality", {})
        cl = rpt.get("classification", {})
        sv = rpt.get("severity", {})
        st.markdown(f"**BLEU-1** : `{cq.get('bleu1', '-')}`")
        st.markdown(f"**BLEU-4** : `{cq.get('bleu4', '-')}`")
        st.markdown(f"**ROUGE-L**: `{cq.get('rougeL', '-')}`")
        st.markdown(f"**Acc**    : `{cl.get('accuracy', '-')}`")
        st.markdown(f"**Macro F1**: `{cl.get('macro_f1', '-')}`")
        st.markdown(f"**Sev R²** : `{sv.get('r2', '-')}`")
    else:
        st.info("Run 04_evaluate.py to see metrics here.")


# ── Main UI ───────────────────────────────────────────────────────────────

st.markdown("# 🌿 PlantVillage Disease Detector")
st.markdown(
    "**Model:** Qwen2.5-VL-7B-Instruct · **Fine-tuned on:** PlantVillage (15 classes) · "
    "**Strategy:** 4-bit QLoRA r64 + Flash Attention 2"
)
st.markdown("---")

# Load models
with st.spinner("Loading Qwen2.5-VL-7B..."):
    vlm, proc, err = load_vlm()

if err:
    st.error(f"⚠️ Model load error: {err}")
    st.stop()

if enable_ood:
    with st.spinner("Loading CLIP for OOD detection..."):
        clip_m, clip_p = load_clip()
else:
    clip_m, clip_p = None, None

# Upload
uploaded = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG and PNG formats"
)

col1, col2 = st.columns([1, 1], gap="large")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    with col1:
        st.markdown("### 📷 Input Image")
        st.image(image, use_container_width=True)
        w, h = image.size
        st.caption(f"Size: {w} × {h}px")

    with col2:
        st.markdown("### 🔬 Diagnosis")

        # OOD check
        ood_flagged  = False
        ood_sim      = None

        if enable_ood and clip_m is not None:
            with st.spinner("Running OOD check..."):
                ood_flagged, ood_sim = run_ood(image, clip_m, clip_p, ood_thresh)

            if ood_sim is not None:
                ood_color = "#ef4444" if ood_flagged else "#22c55e"
                st.markdown(
                    f'<div style="color:{ood_color}; font-size:0.9rem;">'
                    f'CLIP score: <b>{ood_sim:.3f}</b> '
                    f'{"⚠️ OOD" if ood_flagged else "✅ In-distribution"}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        if ood_flagged:
            st.markdown(
                '<div class="ood-warning">'
                '⚠️ <b>Out-Of-Distribution image detected.</b><br>'
                'This image does not appear to be a plant leaf. '
                'The model may produce unreliable results. '
                'Proceeding anyway...'
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown("")

        # Run inference
        with st.spinner("Generating diagnosis..."):
            raw_caption = run_inference(image, vlm, proc, instruction)

        # Parse structured output
        fields = parse_diagnosis(raw_caption)

        # Display structured fields
        plant     = fields.get("Plant",     "—")
        condition = fields.get("Condition", "—")
        severity  = fields.get("Severity",  "—")
        pathogen  = fields.get("Pathogen",  "—")
        symptoms  = fields.get("Symptoms",  "")

        sev_color = SEVERITY_COLORS.get(severity, "#6b7280")

        st.markdown(f"""
        <div class="diagnosis-box">
            <div class="metric-card">🌱 <b>Plant</b>: {plant}</div>
            <div class="metric-card">🦠 <b>Condition</b>: {condition}</div>
            <div class="metric-card">
                📊 <b>Severity</b>:
                <span class="severity-badge" style="background:{sev_color}22; color:{sev_color}; border:1px solid {sev_color};">
                    {severity}
                </span>
            </div>
            <div class="metric-card">🔬 <b>Pathogen</b>: {pathogen}</div>
        </div>
        """, unsafe_allow_html=True)

        if symptoms:
            st.markdown("**Observed Symptoms**")
            st.markdown(
                f'<div class="diagnosis-box" style="font-size:0.9rem;">{symptoms}</div>',
                unsafe_allow_html=True
            )

        st.markdown("**Raw model output**")
        with st.expander("Show full caption"):
            st.code(raw_caption, language=None)

        # Copy button placeholder (Streamlit limitation)
        st.download_button(
            "💾 Download diagnosis (.txt)",
            data=raw_caption,
            file_name="plant_diagnosis.txt",
            mime="text/plain",
        )

else:
    with col1:
        st.markdown("### 📷 Input Image")
        st.info("Upload an image to get started.")
    with col2:
        st.markdown("### 🔬 Diagnosis")
        st.info("Diagnosis will appear here after uploading an image.")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Model: Qwen2.5-VL-7B-Instruct · Fine-tuned with QLoRA r64 · "
    "OOD detection via CLIP ViT-B/32 · PlantVillage dataset (15 classes, ~20k images)</small>",
    unsafe_allow_html=True
)
