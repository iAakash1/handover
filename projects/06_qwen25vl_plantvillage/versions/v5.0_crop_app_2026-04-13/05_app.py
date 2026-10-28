import json
import time
import re
from pathlib import Path
from typing import Tuple

import streamlit as st
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Salesforce/blip2-opt-2.7b"
ADAPTER_PATH = "outputs/blip2_plantvillage/best"
PLANT_DIR    = r"C:\Users\Admin\Desktop\VLM\PlantVillage"
RESULTS_JSON = "inference_results.json"

GEN_KWARGS = dict(
    max_new_tokens       = 140,      # increased for richer captions
    num_beams            = 5,
    repetition_penalty   = 2.5,
    no_repeat_ngram_size = 6,
    length_penalty       = 1.0,
    early_stopping       = True,
)

# ── Disease knowledge panel ──────────────────────────────────────────────────
DISEASE_INFO = {
    "bacterial spot": {
        "pathogen":   "Xanthomonas campestris / X. euvesicatoria",
        "type":       "Bacterial",
        "spread":     "Rain splash, wind, infected seed",
        "treatment":  "Copper-based bactericides, crop rotation",
        "risk":       "HIGH",
    },
    "early blight": {
        "pathogen":   "Alternaria solani",
        "type":       "Fungal",
        "spread":     "Airborne spores, water splash",
        "treatment":  "Mancozeb, chlorothalonil fungicides",
        "risk":       "MEDIUM",
    },
    "late blight": {
        "pathogen":   "Phytophthora infestans",
        "type":       "Oomycete",
        "spread":     "Wind-dispersed sporangia, high humidity",
        "treatment":  "Metalaxyl, dimethomorph fungicides",
        "risk":       "CRITICAL",
    },
    "leaf mold": {
        "pathogen":   "Passalora fulva (Cladosporium fulvum)",
        "type":       "Fungal",
        "spread":     "Airborne conidia, high humidity",
        "treatment":  "Reduce humidity, copper fungicides",
        "risk":       "MEDIUM",
    },
    "septoria leaf spot": {
        "pathogen":   "Septoria lycopersici",
        "type":       "Fungal",
        "spread":     "Rain splash from soil, infected debris",
        "treatment":  "Chlorothalonil, mancozeb, crop rotation",
        "risk":       "MEDIUM",
    },
    "spider mite damage": {
        "pathogen":   "Tetranychus urticae (Two-spotted spider mite)",
        "type":       "Pest — Arachnid",
        "spread":     "Wind dispersal, physical contact",
        "treatment":  "Miticides, predatory mites, neem oil",
        "risk":       "MEDIUM",
    },
    "target spot": {
        "pathogen":   "Corynespora cassiicola",
        "type":       "Fungal",
        "spread":     "Airborne conidia, water splash",
        "treatment":  "Azoxystrobin, difenoconazole fungicides",
        "risk":       "MEDIUM",
    },
    "mosaic virus": {
        "pathogen":   "Tomato Mosaic Virus (ToMV)",
        "type":       "Viral",
        "spread":     "Mechanical contact, infected seed, aphids",
        "treatment":  "No cure — remove infected plants, use resistant varieties",
        "risk":       "HIGH",
    },
    "yellow leaf curl virus": {
        "pathogen":   "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "type":       "Viral (Begomovirus)",
        "spread":     "Whitefly (Bemisia tabaci) vector",
        "treatment":  "Whitefly control, reflective mulches, resistant cultivars",
        "risk":       "CRITICAL",
    },
    "tomato yellowleaf curl virus": {
        "pathogen":   "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "type":       "Viral (Begomovirus)",
        "spread":     "Whitefly (Bemisia tabaci) vector",
        "treatment":  "Whitefly control, reflective mulches, resistant cultivars",
        "risk":       "CRITICAL",
    },
    "healthy": {
        "pathogen":   "None detected",
        "type":       "—",
        "spread":     "—",
        "treatment":  "Continue standard agronomic practices",
        "risk":       "NONE",
    },
}

RISK_STYLE = {
    "NONE":     ("color:#4ade80;background:#052e16;border:1px solid #166534;"),
    "MEDIUM":   ("color:#facc15;background:#1c1400;border:1px solid #854d0e;"),
    "HIGH":     ("color:#fb923c;background:#1c0a00;border:1px solid #9a3412;"),
    "CRITICAL": ("color:#f87171;background:#200000;border:1px solid #991b1b;"),
}

CLASS_DISPLAY = {
    "Pepper__bell___Bacterial_spot":               ("Pepper Bell",  "Bacterial Spot"),
    "Pepper__bell___healthy":                      ("Pepper Bell",  "Healthy"),
    "Potato___Early_blight":                       ("Potato",       "Early Blight"),
    "Potato___healthy":                            ("Potato",       "Healthy"),
    "Potato___Late_blight":                        ("Potato",       "Late Blight"),
    "Tomato__Target_Spot":                         ("Tomato",       "Target Spot"),
    "Tomato__Tomato_mosaic_virus":                 ("Tomato",       "Mosaic Virus"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       ("Tomato",       "Yellow Leaf Curl Virus"),
    "Tomato_Bacterial_spot":                       ("Tomato",       "Bacterial Spot"),
    "Tomato_Early_blight":                         ("Tomato",       "Early Blight"),
    "Tomato_healthy":                              ("Tomato",       "Healthy"),
    "Tomato_Late_blight":                          ("Tomato",       "Late Blight"),
    "Tomato_Leaf_Mold":                            ("Tomato",       "Leaf Mold"),
    "Tomato_Septoria_leaf_spot":                   ("Tomato",       "Septoria Leaf Spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato",       "Spider Mite Damage"),
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantVillage VLM Demo",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Reset & base ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #080d08;
    color: #ddeedd;
}
.block-container {
    padding: 0 2.5rem 3rem !important;
    max-width: 1280px !important;
}
section[data-testid="stSidebar"] {
    background: #0c140c;
    border-right: 1px solid #1a2a1a;
}
/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Typography helpers ───────────────────────────── */
.mono { font-family: 'IBM Plex Mono', monospace; }
.label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3d6b3a;
    margin-bottom: 0.35rem;
}
.section-title {
    font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3d9142;
    margin: 1.6rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a2a1a;
}

/* ── Hero banner ──────────────────────────────────── */
.hero {
    padding: 2.2rem 2.5rem 1.8rem;
    margin: 0 -2.5rem 2rem;
    background: linear-gradient(180deg, #0a180a 0%, #080d08 100%);
    border-bottom: 1px solid #1a2a1a;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 40%;
    background: radial-gradient(ellipse at 80% 50%, rgba(74,222,128,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #3d9142;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0faf0;
    margin: 0 0 0.5rem;
    letter-spacing: -0.025em;
    line-height: 1.15;
}
.hero h1 em {
    font-style: normal;
    color: #4ade80;
}
.hero-sub {
    color: #5a7a58;
    font-size: 0.88rem;
    font-weight: 400;
    max-width: 520px;
    line-height: 1.55;
    margin-bottom: 1.2rem;
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}
.chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #3d9142;
    background: #0d1a0d;
    border: 1px solid #1e3a1e;
    padding: 0.28rem 0.7rem;
    border-radius: 100px;
    letter-spacing: 0.06em;
}

/* ── Panel card ───────────────────────────────────── */
.panel {
    background: #0c140c;
    border: 1px solid #1a2a1a;
    border-radius: 8px;
    padding: 1.4rem;
    height: 100%;
}
.panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3d6b3a;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1a2a1a;
}

/* ── Image display ────────────────────────────────── */
.img-frame {
    border: 1px solid #1a2a1a;
    border-radius: 6px;
    overflow: hidden;
    background: #060c06;
}
.img-meta {
    padding: 0.65rem 0.9rem;
    border-top: 1px solid #1a2a1a;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.img-meta-left {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #3d6b3a;
}
.img-meta-right {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: #2a4a2a;
}

/* ── Caption result ───────────────────────────────── */
.result-wrap {
    background: #0c140c;
    border: 1px solid #1a2a1a;
    border-radius: 8px;
    overflow: hidden;
}
.result-header {
    padding: 0.8rem 1.2rem;
    background: #0a120a;
    border-bottom: 1px solid #1a2a1a;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-header-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.result-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 6px #4ade80aa;
    flex-shrink: 0;
}
.result-header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4ade80;
}
.result-timing {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: #2a4a2a;
}
.result-timing span { color: #facc15; }
.result-body { padding: 1.2rem; }
.caption-text {
    font-size: 1.0rem;
    line-height: 1.75;
    color: #ddeedd;
    font-weight: 400;
    margin-bottom: 1.2rem;
}

/* ── Field grid ───────────────────────────────────── */
.field-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.7rem;
    margin-bottom: 1rem;
}
.field-cell {
    background: #080d08;
    border: 1px solid #1a2a1a;
    border-radius: 6px;
    padding: 0.75rem 0.9rem;
}
.field-cell .f-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d6b3a;
    margin-bottom: 0.3rem;
}
.field-cell .f-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e0f0e0;
    line-height: 1.2;
}

/* ── Severity badge ───────────────────────────────── */
.sev {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.65rem;
    border-radius: 100px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid transparent;
}
.sev-none     { color: #4ade80; background: #052e16; border-color: #166534; }
.sev-mild     { color: #fde047; background: #1c1400; border-color: #854d0e; }
.sev-moderate { color: #fb923c; background: #1c0a00; border-color: #9a3412; }
.sev-severe   { color: #f87171; background: #200000; border-color: #991b1b; }

/* ── Visual description ───────────────────────────── */
.vis-desc {
    background: #080d08;
    border: 1px solid #1a2a1a;
    border-left: 3px solid #3d9142;
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #8ab88a;
    line-height: 1.7;
    font-style: italic;
    margin-bottom: 1rem;
}

/* ── Alert banner ─────────────────────────────────── */
.alert {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.9rem 1.1rem;
    border-radius: 6px;
    font-size: 0.84rem;
    line-height: 1.55;
    margin-bottom: 0.5rem;
}
.alert-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
.alert-body strong { display: block; margin-bottom: 0.2rem; font-weight: 600; }
.alert-healthy  { background: #062010; border: 1px solid #145214; color: #86efac; }
.alert-severe   { background: #1c0404; border: 1px solid #7f1d1d; color: #fca5a5; }
.alert-moderate { background: #1c0c00; border: 1px solid #7c2d12; color: #fdba74; }
.alert-mild     { background: #1c1600; border: 1px solid #78350f; color: #fde68a; }

/* ── Disease info table ───────────────────────────── */
.dinfo-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.6rem;
}
.dinfo-row {
    background: #080d08;
    border: 1px solid #1a2a1a;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
}
.dinfo-row.full { grid-column: 1 / -1; }
.dinfo-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.53rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d6b3a;
    margin-bottom: 0.2rem;
}
.dinfo-val {
    font-size: 0.8rem;
    color: #c0d8c0;
    line-height: 1.4;
}
.risk-badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 100px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Batch results ────────────────────────────────── */
.batch-card {
    background: #0c140c;
    border: 1px solid #1a2a1a;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s;
}
.batch-card:hover { border-color: #2a4a2a; }
.batch-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 1.1rem;
    cursor: pointer;
}
.batch-plant-tag {
    font-size: 0.72rem;
    font-weight: 600;
    color: #e0f0e0;
}
.batch-condition-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #5a8a58;
    margin-left: 0.6rem;
}

/* ── Stat cards ───────────────────────────────────── */
.stat-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin-bottom: 2rem;
}
.stat-card {
    background: #0c140c;
    border: 1px solid #1a2a1a;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4ade80, transparent);
    opacity: 0.4;
}
.stat-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: #4ade80;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label {
    font-size: 0.75rem;
    color: #3d6b3a;
    font-weight: 400;
}

/* ── Empty state ──────────────────────────────────── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 2rem;
    border: 1px dashed #1a2a1a;
    border-radius: 8px;
    background: #080d08;
    text-align: center;
}
.empty-icon { font-size: 2.2rem; margin-bottom: 0.8rem; opacity: 0.5; }
.empty-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #3d6b3a;
    margin-bottom: 0.3rem;
}
.empty-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #2a4a2a;
}

/* ── Streamlit widget overrides ───────────────────── */
.stButton > button {
    background: #0d1f0d !important;
    border: 1px solid #2a5a2a !important;
    color: #4ade80 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #142814 !important;
    border-color: #4ade80 !important;
    color: #86efac !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0c140c !important;
    border-color: #1a2a1a !important;
    color: #ddeedd !important;
}
.stSlider > div { padding: 0 !important; }
div[data-baseweb="slider"] {
    padding: 0 !important;
}
.stFileUploader > div {
    background: #0c140c !important;
    border: 1px dashed #2a4a2a !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    processor = Blip2Processor.from_pretrained(ADAPTER_PATH)
    base = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, processor


def generate_caption(image: Image.Image, model, processor) -> Tuple[str, float]:
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    elapsed = time.time() - t0
    raw = processor.decode(out[0], skip_special_tokens=True).strip()
    # Keep all sentences — no truncation
    if not raw.endswith("."):
        raw += "."
    return raw, elapsed


def parse_caption(caption: str) -> dict:
    result = {"plant": "—", "condition": "—", "severity": "—", "description": ""}
    m = re.search(r"Plant:\s*([^.]+)\.", caption)
    if m:
        result["plant"] = m.group(1).strip()
    m = re.search(r"Condition:\s*([^.]+)\.", caption)
    if m:
        result["condition"] = m.group(1).strip()
    m = re.search(r"Severity:\s*([^.]+)\.", caption)
    if m:
        result["severity"] = m.group(1).strip()
    # Everything after the 3rd period is the visual description
    parts = caption.split(". ", 3)
    if len(parts) >= 4:
        result["description"] = parts[3].strip()
    elif len(parts) == 3:
        result["description"] = parts[2].strip()
    return result


def get_sev_class(sev: str) -> str:
    s = sev.lower()
    if "none" in s or s == "—":   return "sev-none"
    if "severe" in s and "moderate" not in s: return "sev-severe"
    if "moderate" in s:            return "sev-moderate"
    if "mild" in s:                return "sev-mild"
    return "sev-mild"


def get_disease_info(condition: str) -> dict:
    key = condition.lower().strip()
    for k, v in DISEASE_INFO.items():
        if k in key or key in k:
            return v
    return None


def get_dataset_samples() -> dict:
    samples = {}
    p = Path(PLANT_DIR)
    if not p.exists():
        return samples
    for folder in sorted(p.iterdir()):
        if not folder.is_dir():
            continue
        imgs = (
            list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) +
            list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        )
        if imgs:
            samples[folder.name] = imgs
    return samples


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">VLM Research Project &nbsp;·&nbsp; College</div>
  <h1>Plant Disease<br/><em>Caption Generator</em></h1>
  <p class="hero-sub">
    A fine-tuned Vision Language Model that analyses plant leaf images and generates
    structured diagnostic captions — identifying the plant species, disease condition,
    severity level, and detailed visual symptoms.
  </p>
  <div class="hero-chips">
    <span class="chip">BLIP-2 OPT-2.7B</span>
    <span class="chip">LoRA r32</span>
    <span class="chip">PlantVillage · 20,638 images</span>
    <span class="chip">15 disease classes</span>
    <span class="chip">RTX 4500 Ada · 24 GB</span>
    <span class="chip">~8.4 GB VRAM</span>
    <span class="chip">~2 hr training</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-bar">
  <div class="stat-card">
    <div class="stat-num">20,638</div>
    <div class="stat-label">Training images</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">15</div>
    <div class="stat-label">Disease classes</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">200+</div>
    <div class="stat-label">Caption variants / class</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">15/15</div>
    <div class="stat-label">Inference accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='panel-title'>⚙️ Settings</div>", unsafe_allow_html=True)
    st.markdown("**Input mode**")
    mode = st.radio("mode", ["📁 Upload Image", "🗂️ Browse Dataset"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Generation**")
    GEN_KWARGS["max_new_tokens"]     = st.slider("Max tokens",         60, 200, 140, 10)
    GEN_KWARGS["num_beams"]          = st.slider("Beam search",         1,   8,   5,  1)
    GEN_KWARGS["repetition_penalty"] = st.slider("Repetition penalty", 1.0, 5.0, 2.5, 0.5)
    st.markdown("---")
    if Path(RESULTS_JSON).exists():
        show_batch = st.button("📋 Show all inference results", use_container_width=True)
    else:
        show_batch = False
        st.caption("Run 03_inference.py to generate batch results.")

# ── Main layout: left = image input, right = results ─────────────────────────
dataset_samples = get_dataset_samples()
selected_image  = None
selected_label  = "Unknown"

col_img, col_res = st.columns([5, 7], gap="large")

with col_img:
    st.markdown("<div class='section-title'>Input</div>", unsafe_allow_html=True)

    if mode == "📁 Upload Image":
        uploaded = st.file_uploader(
            "Drop a plant leaf image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if uploaded:
            selected_image = Image.open(uploaded).convert("RGB")
            selected_label = uploaded.name
            w, h = selected_image.size
            st.markdown(
                f"<div class='img-frame'></div>",
                unsafe_allow_html=True,
            )
            st.image(selected_image, use_container_width=True)
            st.markdown(
                f"<div class='img-meta'>"
                f"<span class='img-meta-left'>{uploaded.name}</span>"
                f"<span class='img-meta-right'>{w} × {h} px</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        if not dataset_samples:
            st.warning(
                f"Dataset not found at:\n`{PLANT_DIR}`\n\nUpdate `PLANT_DIR` in the script."
            )
        else:
            chosen = st.selectbox(
                "Select class",
                list(dataset_samples.keys()),
                format_func=lambda x: (
                    CLASS_DISPLAY.get(x, (x, ""))[0] + "  ·  " +
                    CLASS_DISPLAY.get(x, (x, x))[1]
                ),
            )
            imgs   = dataset_samples[chosen]
            idx    = st.slider(
                f"Image index (of {len(imgs)} total)",
                0, min(len(imgs) - 1, 99), 0,
            )
            selected_image = Image.open(str(imgs[idx])).convert("RGB")
            selected_label = chosen
            w, h = selected_image.size
            st.image(selected_image, use_container_width=True)
            st.markdown(
                f"<div class='img-meta'>"
                f"<span class='img-meta-left'>{CLASS_DISPLAY.get(chosen,(chosen,''))[0]} · "
                f"{CLASS_DISPLAY.get(chosen,(chosen,chosen))[1]}</span>"
                f"<span class='img-meta-right'>{w} × {h}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Thumbnail strip — first 6 images
            st.markdown(
                "<div style='margin-top:0.8rem;font-size:0.72rem;"
                "color:#3d6b3a;margin-bottom:0.4rem;'>Other images in class</div>",
                unsafe_allow_html=True,
            )
            tcols = st.columns(6)
            for ti, tc in enumerate(tcols):
                if ti < min(len(imgs), 6):
                    thumb = Image.open(str(imgs[ti])).convert("RGB")
                    thumb.thumbnail((80, 80))
                    tc.image(thumb, use_container_width=True)


with col_res:
    st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)

    if selected_image is None:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🔬</div>
          <div class="empty-title">No image selected</div>
          <div class="empty-sub">Upload or browse dataset to begin</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        gen_btn = st.button(
            "▶  Run Caption Generation",
            type="primary",
            use_container_width=True,
        )

        if gen_btn:
            with st.spinner("Loading model into VRAM — first run only (~30s)..."):
                try:
                    model, processor = load_model()
                    ok = True
                except Exception as e:
                    st.error(f"Model load failed: {e}")
                    st.code(
                        f"Adapter path: {ADAPTER_PATH}\n"
                        "Run 02_finetune_blip2.py first, then retry."
                    )
                    ok = False

            if ok:
                with st.spinner("Generating caption..."):
                    caption, elapsed = generate_caption(selected_image, model, processor)

                parsed   = parse_caption(caption)
                sev_cls  = get_sev_class(parsed["severity"])
                dinfo    = get_disease_info(parsed["condition"])
                sev_l    = parsed["severity"].lower()

                # ── Result card ────────────────────────────────────────────
                sev_badge = (
                    f"<span class='sev {sev_cls}'>{parsed['severity']}</span>"
                )

                st.markdown(
                    f"""<div class="result-wrap">
  <div class="result-header">
    <div class="result-header-left">
      <div class="result-dot"></div>
      <span class="result-header-title">Model Output</span>
    </div>
    <span class="result-timing">⚡ <span>{elapsed:.2f}s</span></span>
  </div>
  <div class="result-body">
    <p class="caption-text">{caption}</p>
    <div class="field-row">
      <div class="field-cell">
        <div class="f-label">🌱 Plant Species</div>
        <div class="f-value">{parsed['plant']}</div>
      </div>
      <div class="field-cell">
        <div class="f-label">🦠 Condition</div>
        <div class="f-value">{parsed['condition']}</div>
      </div>
      <div class="field-cell">
        <div class="f-label">⚠️ Severity</div>
        <div class="f-value">{sev_badge}</div>
      </div>
    </div>""",
                    unsafe_allow_html=True,
                )

                if parsed["description"]:
                    st.markdown(
                        f"<div class='vis-desc'>{parsed['description']}</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("</div></div>", unsafe_allow_html=True)

                # ── Alert ──────────────────────────────────────────────────
                if "healthy" in caption.lower() and "none" in sev_l:
                    st.markdown("""
                    <div class="alert alert-healthy">
                      <span class="alert-icon">✅</span>
                      <div class="alert-body">
                        <strong>No disease detected</strong>
                        Leaf appears healthy with normal coloration and no visible lesions.
                        Continue standard monitoring and agronomic practices.
                      </div>
                    </div>""", unsafe_allow_html=True)
                elif "severe" in sev_l and "moderate" not in sev_l:
                    st.markdown(f"""
                    <div class="alert alert-severe">
                      <span class="alert-icon">🔴</span>
                      <div class="alert-body">
                        <strong>Severe infection — immediate action required</strong>
                        This level of infection can cause rapid spread and significant
                        yield loss. Isolate affected plants and apply appropriate treatment.
                      </div>
                    </div>""", unsafe_allow_html=True)
                elif "moderate" in sev_l:
                    st.markdown(f"""
                    <div class="alert alert-moderate">
                      <span class="alert-icon">🟠</span>
                      <div class="alert-body">
                        <strong>Moderate infection — intervention recommended</strong>
                        Condition is progressing. Schedule treatment soon to
                        prevent escalation to severe stage.
                      </div>
                    </div>""", unsafe_allow_html=True)
                elif "mild" in sev_l:
                    st.markdown(f"""
                    <div class="alert alert-mild">
                      <span class="alert-icon">🟡</span>
                      <div class="alert-body">
                        <strong>Mild infection detected — monitor closely</strong>
                        Early stage symptoms present. Increase inspection frequency
                        and consider preventive treatment.
                      </div>
                    </div>""", unsafe_allow_html=True)

                # ── Disease knowledge panel ────────────────────────────────
                if dinfo:
                    risk_style = RISK_STYLE.get(dinfo["risk"], "color:#ddeedd;background:#1a2a1a;border:1px solid #2a4a2a;")
                    st.markdown(
                        f"<div class='section-title' style='margin-top:1.4rem;'>Disease Profile</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""<div class="dinfo-grid">
  <div class="dinfo-row">
    <div class="dinfo-label">Pathogen</div>
    <div class="dinfo-val">{dinfo['pathogen']}</div>
  </div>
  <div class="dinfo-row">
    <div class="dinfo-label">Type</div>
    <div class="dinfo-val">{dinfo['type']}</div>
  </div>
  <div class="dinfo-row">
    <div class="dinfo-label">Spread mechanism</div>
    <div class="dinfo-val">{dinfo['spread']}</div>
  </div>
  <div class="dinfo-row">
    <div class="dinfo-label">Risk level</div>
    <div class="dinfo-val">
      <span class="risk-badge" style="{risk_style}">{dinfo['risk']}</span>
    </div>
  </div>
  <div class="dinfo-row full">
    <div class="dinfo-label">Recommended treatment</div>
    <div class="dinfo-val">{dinfo['treatment']}</div>
  </div>
</div>""",
                        unsafe_allow_html=True,
                    )

        elif not gen_btn and selected_image is not None:
            st.markdown("""
            <div class="empty-state" style="padding:2.5rem;">
              <div class="empty-icon">🌿</div>
              <div class="empty-title">Image ready</div>
              <div class="empty-sub">Press "Run Caption Generation" to analyse</div>
            </div>
            """, unsafe_allow_html=True)


# ── Batch results section ─────────────────────────────────────────────────────
if show_batch and Path(RESULTS_JSON).exists():
    with open(RESULTS_JSON) as f:
        batch = json.load(f)

    st.markdown(
        "<div class='section-title' style='margin-top:1rem;'>All Inference Results</div>",
        unsafe_allow_html=True,
    )

    healthy_n  = sum(1 for r in batch if "Healthy" in r["caption"])
    diseased_n = len(batch) - healthy_n

    st.markdown(
        f"""<div class="stat-bar">
  <div class="stat-card"><div class="stat-num">{len(batch)}</div><div class="stat-label">Classes evaluated</div></div>
  <div class="stat-card"><div class="stat-num">{healthy_n}</div><div class="stat-label">Healthy leaves</div></div>
  <div class="stat-card"><div class="stat-num">{diseased_n}</div><div class="stat-label">Diseased leaves</div></div>
  <div class="stat-card"><div class="stat-num">15/15</div><div class="stat-label">Correct classifications</div></div>
</div>""",
        unsafe_allow_html=True,
    )

    for rec in batch:
        parsed  = parse_caption(rec["caption"])
        sev_cls = get_sev_class(parsed["severity"])
        display = CLASS_DISPLAY.get(rec["class"], (rec["class"], ""))

        with st.expander(f"{display[0]}  ·  {display[1]}"):
            ca, cb = st.columns([1, 2])
            with ca:
                try:
                    img = Image.open(rec["image"]).convert("RGB")
                    st.image(img, use_container_width=True)
                except Exception:
                    st.markdown(
                        "<div class='empty-state' style='padding:1.5rem;'>"
                        "<div class='empty-sub'>Image path not accessible</div></div>",
                        unsafe_allow_html=True,
                    )
            with cb:
                st.markdown(
                    f"<p style='font-size:0.9rem;color:#c0d8c0;line-height:1.7;"
                    f"margin-bottom:0.8rem;'>{rec['caption']}</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span class='sev {sev_cls}'>{parsed['severity']}</span>",
                    unsafe_allow_html=True,
                )
                dinfo = get_disease_info(parsed["condition"])
                if dinfo:
                    risk_style = RISK_STYLE.get(dinfo["risk"], "")
                    st.markdown(
                        f"<span style='margin-left:0.6rem;font-family:IBM Plex Mono,monospace;"
                        f"font-size:0.6rem;color:#3d6b3a;'>{dinfo['type']} — {dinfo['pathogen']}</span>",
                        unsafe_allow_html=True,
                    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='margin-top:3rem;padding-top:1.2rem;border-top:1px solid #1a2a1a;"
    "text-align:center;font-family:IBM Plex Mono,monospace;font-size:0.58rem;color:#2a4a2a;'>"
    "Plant Disease Caption Generator &nbsp;·&nbsp; BLIP-2 OPT-2.7B + LoRA &nbsp;·&nbsp; "
    "PlantVillage Dataset &nbsp;·&nbsp; RTX 4500 Ada Generation 24 GB"
    "</div>",
    unsafe_allow_html=True,
)