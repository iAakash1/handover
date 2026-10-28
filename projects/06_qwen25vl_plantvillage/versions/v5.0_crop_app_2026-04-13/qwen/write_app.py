# write_app.py
# Run: python write_app.py
# Generates a beautiful 05_app.py with no null bytes

code = r'''
import streamlit as st
import torch
import json
import re
from pathlib import Path
from PIL import Image
from typing import Tuple

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)
from peft import PeftModel

ADAPTER_PATH    = "outputs/qwen25vl_plantvillage/best"
BASE_MODEL      = "Qwen/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EVAL_REPORT     = "evaluation_report.json"
OOD_THRESHOLD   = 0.22
MAX_NEW_TOKENS  = 160

DEFAULT_INSTRUCTION = (
    "Analyze this plant leaf image and provide a detailed diagnosis. "
    "Include: plant name, disease or health condition, severity level, "
    "causal pathogen, and visible symptoms."
)

SEVERITY_CONFIG = {
    "None":               {"color": "#4ade80", "bg": "#052e16", "icon": "✦", "label": "Healthy"},
    "Mild":               {"color": "#a3e635", "bg": "#1a2e05", "icon": "▲", "label": "Mild"},
    "Mild To Moderate":   {"color": "#facc15", "bg": "#2d1f05", "icon": "▲▲", "label": "Mild–Moderate"},
    "Moderate":           {"color": "#fb923c", "bg": "#2d1205", "icon": "▲▲▲", "label": "Moderate"},
    "Moderate To Severe": {"color": "#f87171", "bg": "#2d0a0a", "icon": "▲▲▲▲", "label": "Mod–Severe"},
    "Severe":             {"color": "#dc2626", "bg": "#1f0505", "icon": "▲▲▲▲▲", "label": "Severe"},
}

st.set_page_config(
    page_title="PlantVillage AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap');

:root {
    --bg:        #070d07;
    --bg2:       #0d160d;
    --bg3:       #111f11;
    --border:    #1a3a1a;
    --border2:   #254025;
    --green:     #4ade80;
    --green2:    #86efac;
    --green3:    #bbf7d0;
    --muted:     #4a7a4a;
    --text:      #e2f5e2;
    --mono:      'DM Mono', monospace;
    --serif:     'Fraunces', serif;
}

html, body { background: var(--bg) !important; }
.stApp { background: var(--bg) !important; font-family: var(--mono); color: var(--text); }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: var(--serif) !important;
    color: var(--green) !important;
    font-size: 1rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1rem !important;
}

/* Metric cards in sidebar */
[data-testid="stMetric"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.6rem 0.8rem !important;
}
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: var(--muted) !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-size: 1.1rem !important; color: var(--green) !important; font-family: var(--mono) !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--green) !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* Buttons */
.stButton > button, [data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--green2) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    background: var(--border) !important;
    border-color: var(--green) !important;
    color: var(--green) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Toggle */
[data-testid="stToggle"] span { background: var(--border2) !important; }

/* Slider */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--green) !important; }

/* Text area */
textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--green) !important; }

/* Custom classes */
.page-header {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.page-title {
    font-family: var(--serif);
    font-size: 2.8rem;
    font-weight: 300;
    color: var(--green);
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
}
.page-subtitle {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.page-tagline {
    font-family: var(--serif);
    font-style: italic;
    font-size: 0.95rem;
    color: var(--green2);
    margin-top: 0.3rem;
}
.model-pill {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    margin-right: 0.4rem;
    margin-top: 0.8rem;
}
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.diag-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.diag-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
}
.diag-row:last-child { border-bottom: none; padding-bottom: 0; }
.diag-key {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    padding-top: 0.15rem;
    min-width: 90px;
}
.diag-val {
    font-size: 0.95rem;
    color: var(--green3);
    text-align: right;
    flex: 1;
    font-family: var(--mono);
}
.sev-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
    font-family: var(--mono);
}
.symptoms-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    line-height: 1.7;
    color: var(--green2);
    margin-top: 0.5rem;
}
.ood-banner {
    background: #1a0505;
    border: 1px solid #7f1d1d;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #fca5a5;
    font-size: 0.85rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.ood-ok-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}
.stat-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.stat-val { font-size: 1.15rem; color: var(--green); font-family: var(--mono); font-weight: 500; }
.stat-lbl { font-size: 0.6rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.15rem; }
.f1-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.75rem;
}
.f1-row:last-child { border-bottom: none; }
.f1-name { color: var(--text); }
.f1-bar-wrap { flex: 1; margin: 0 0.8rem; height: 3px; background: var(--border); border-radius: 999px; }
.f1-bar { height: 100%; background: var(--green); border-radius: 999px; }
.f1-val { color: var(--green); min-width: 36px; text-align: right; }
.f1-perfect { color: #4ade80; }
.upload-hint {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    padding: 3rem 1rem;
    border: 1px dashed var(--border);
    border-radius: 12px;
    letter-spacing: 0.05em;
}
.diag-placeholder {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    padding: 3rem 1rem;
    border: 1px dashed var(--border);
    border-radius: 12px;
    font-family: var(--serif);
    font-style: italic;
}
.img-frame {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--bg2);
}
.img-meta {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    padding: 0.5rem 0.8rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vlm():
    if not Path(ADAPTER_PATH).exists():
        return None, None, f"Adapter not found at '{ADAPTER_PATH}'."
    try:
        processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        return model, processor, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_clip():
    try:
        m = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to("cuda")
        p = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        m.eval()
        return m, p
    except Exception:
        return None, None


def run_ood(image, clip_model, clip_proc, threshold):
    inputs = clip_proc(text=["a photo of a plant leaf"], images=image,
                       return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = clip_model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (img_emb * txt_emb).sum().item()
    return sim < threshold, round(sim, 4)


def run_inference(image, model, processor, instruction):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": instruction},
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             num_beams=4, repetition_penalty=2.5,
                             no_repeat_ngram_size=6, early_stopping=True)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def parse_caption(text):
    fields = {}
    for field in ["Plant", "Condition", "Severity", "Pathogen"]:
        m = re.search(rf"{field}[:\s]+([^.]+)", text, re.IGNORECASE)
        if m:
            fields[field] = m.group(1).strip().rstrip(".,")
    parts = text.split(". ")
    if len(parts) > 4:
        fields["Symptoms"] = ". ".join(parts[4:]).strip()
    elif len(parts) > 1:
        fields["Symptoms"] = parts[-1].strip()
    return fields


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Model")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a7a4a; line-height:1.8;'>
    <b style='color:#86efac;'>Qwen2.5-VL-7B-Instruct</b><br>
    bfloat16 · LoRA r32<br>
    PlantVillage · 15 classes<br>
    20,638 training images
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Detection")
    enable_ood = st.toggle("OOD Detection", value=True)
    ood_thresh = st.slider("CLIP Threshold", 0.10, 0.40, OOD_THRESHOLD, 0.01)

    st.markdown("---")
    st.markdown("## Instruction")
    instruction = st.text_area("", value=DEFAULT_INSTRUCTION, height=100, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("## Metrics")

    if Path(EVAL_REPORT).exists():
        with open(EVAL_REPORT) as f:
            rpt = json.load(f)
        cq = rpt.get("caption_quality", {})
        cl = rpt.get("classification",  {})
        sv = rpt.get("severity",        {})
        pf = rpt.get("per_class_f1",    {})

        st.markdown("""<div class='stat-grid'>""", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("BLEU-1",   f"{cq.get('bleu1',0):.4f}")
            st.metric("ROUGE-L",  f"{cq.get('rougeL',0):.4f}")
            st.metric("Accuracy", f"{cl.get('accuracy',0):.4f}")
            st.metric("Sev R²",   f"{sv.get('r2',0):.4f}")
        with col_b:
            st.metric("BLEU-4",   f"{cq.get('bleu4',0):.4f}")
            st.metric("CLIP Sim", f"{cq.get('clip_similarity',0):.4f}")
            st.metric("Macro F1", f"{cl.get('macro_f1',0):.4f}")
            st.metric("Sev MAE",  f"{sv.get('mae',0):.4f}")

        if pf:
            st.markdown("---")
            st.markdown("## Per-class F1")
            rows_html = ""
            for cls, f1 in sorted(pf.items(), key=lambda x: -x[1]):
                pct = int(f1 * 100)
                perfect = f1 >= 1.0
                val_cls = "f1-perfect" if perfect else "f1-val"
                rows_html += f"""
                <div class='f1-row'>
                    <span class='f1-name'>{cls}</span>
                    <div class='f1-bar-wrap'><div class='f1-bar' style='width:{pct}%'></div></div>
                    <span class='{val_cls}'>{f1:.3f}</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.info("Run 04_evaluate.py to see metrics.")


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <p class='page-title'>PlantVillage AI</p>
    <p class='page-tagline'>Automated plant disease diagnosis powered by vision-language AI</p>
    <div style='margin-top:0.8rem;'>
        <span class='model-pill'>Qwen2.5-VL-7B</span>
        <span class='model-pill'>LoRA r32</span>
        <span class='model-pill'>bfloat16</span>
        <span class='model-pill'>15 disease classes</span>
        <span class='model-pill'>98.5% accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Initialising model..."):
    vlm, proc, err = load_vlm()

if err:
    st.error(f"Model error: {err}")
    st.stop()

if enable_ood:
    clip_m, clip_p = load_clip()
else:
    clip_m, clip_p = None, None

# Upload
st.markdown("<p class='section-label'>Input</p>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

col1, col2 = st.columns([1, 1], gap="large")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    w, h  = image.size

    with col1:
        st.markdown("<p class='section-label'>Image</p>", unsafe_allow_html=True)
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div class='img-meta'>
            <span>{uploaded.name}</span>
            <span>{w} &times; {h} px</span>
        </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<p class='section-label'>Diagnosis</p>", unsafe_allow_html=True)

        ood_flagged = False
        if enable_ood and clip_m:
            with st.spinner("Running OOD check..."):
                ood_flagged, ood_score = run_ood(image, clip_m, clip_p, ood_thresh)

            if ood_flagged:
                st.markdown(f"""
                <div class='ood-banner'>
                    ⚠ Out-of-distribution image detected (CLIP score: {ood_score:.3f})
                    — results may be unreliable
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ood-ok-badge' style='background:#052e16;border:1px solid #166534;color:#4ade80;'>
                    ✓ &nbsp;In-distribution &nbsp;·&nbsp; CLIP {ood_score:.3f}
                </div>""", unsafe_allow_html=True)

        with st.spinner("Generating diagnosis..."):
            raw = run_inference(image, vlm, proc, instruction)

        fields   = parse_caption(raw)
        plant    = fields.get("Plant",     "—")
        cond     = fields.get("Condition", "—")
        sev      = fields.get("Severity",  "—")
        path     = fields.get("Pathogen",  "—")
        symptoms = fields.get("Symptoms",  "")

        sev_key = sev.title()
        cfg = SEVERITY_CONFIG.get(sev_key, {"color": "#6b7280", "bg": "#1c1c1c", "icon": "▲", "label": sev})

        sev_html = f"""
        <span class='sev-pill' style='background:{cfg["bg"]};color:{cfg["color"]};border:1px solid {cfg["color"]}40;'>
            {cfg["icon"]} &nbsp; {sev}
        </span>"""

        st.markdown(f"""
        <div class='diag-card'>
            <div class='diag-row'>
                <span class='diag-key'>Plant</span>
                <span class='diag-val'>{plant}</span>
            </div>
            <div class='diag-row'>
                <span class='diag-key'>Condition</span>
                <span class='diag-val'>{cond}</span>
            </div>
            <div class='diag-row'>
                <span class='diag-key'>Severity</span>
                <span class='diag-val'>{sev_html}</span>
            </div>
            <div class='diag-row'>
                <span class='diag-key'>Pathogen</span>
                <span class='diag-val'>{path}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if symptoms:
            st.markdown("<p class='section-label'>Observed Symptoms</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='symptoms-box'>{symptoms}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Raw model output"):
            st.code(raw, language=None)

        st.download_button(
            "↓ Download diagnosis",
            data=raw,
            file_name="plant_diagnosis.txt",
            mime="text/plain",
        )

else:
    with col1:
        st.markdown("<p class='section-label'>Image</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class='upload-hint'>
            Drop a plant leaf image above<br>
            <span style='font-size:0.7rem;opacity:0.5;'>JPG · JPEG · PNG</span>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<p class='section-label'>Diagnosis</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class='diag-placeholder'>
            Diagnosis will appear here after<br>uploading an image
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:3rem; padding-top:1rem; border-top:1px solid #1a3a1a;
     font-size:0.65rem; color:#2d5a2d; letter-spacing:0.1em; text-align:center;'>
    PLANTVILLAGE AI &nbsp;·&nbsp; Qwen2.5-VL-7B-Instruct &nbsp;·&nbsp;
    LoRA r32 &nbsp;·&nbsp; OOD via CLIP ViT-B/32 &nbsp;·&nbsp;
    Accuracy 98.5% &nbsp;·&nbsp; Macro F1 0.9827
</div>
""", unsafe_allow_html=True)
'''

with open("05_app.py", "w", encoding="utf-8") as f:
    f.write(code.lstrip())

with open("05_app.py", "rb") as f:
    raw = f.read()

null_count = raw.count(b"\x00")
print(f"Written: {len(raw)} bytes | Null bytes: {null_count}")
print("Ready — run: streamlit run 05_app.py" if null_count == 0 else "ERROR: null bytes found!")
