# 05_app.py  (v2 — multi-model + OOD detection)
#
# RUN:
#   streamlit run 05_app.py
#
# NEW IN V2:
#   • Model selector: BLIP-2 vs Florence-2
#   • OOD detection: CLIP-based with user-adjustable threshold
#   • Full evaluation dashboard tab (reads evaluation_results.json)
#   • Side-by-side model comparison if both adapters are present

import streamlit as st
import torch
import json
import re
from pathlib import Path
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #0d1117; color: #e6edf3; }

    .block-container { padding: 1.5rem 2rem; max-width: 1300px; }

    .caption-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.7;
        color: #8b949e;
        margin-top: 0.5rem;
    }
    .caption-box span { color: #58a6ff; font-weight: 700; }

    .badge-healthy   { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950; }
    .badge-mild      { background:#2d2a17; color:#e3b341; border:1px solid #e3b341; }
    .badge-moderate  { background:#3a2413; color:#f0883e; border:1px solid #f0883e; }
    .badge-severe    { background:#3d1a1a; color:#f85149; border:1px solid #f85149; }
    .badge-ood       { background:#2a1a3d; color:#bc8cff; border:1px solid #bc8cff; }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        margin-bottom: 0.5rem;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #58a6ff; font-family: 'Space Mono', monospace; }
    .metric-card .lbl { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }

    .ood-warning {
        background: #2a1a3d;
        border: 1px solid #bc8cff;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 0.5rem;
        color: #bc8cff;
        font-weight: 600;
    }

    .knowledge-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .knowledge-card h4 { color: #e6edf3; font-size: 0.9rem; margin-bottom: 0.4rem; }

    h1, h2, h3, h4 { font-family: 'Space Mono', monospace; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }
    .stSelectbox label, .stSlider label, .stFileUploader label { color: #8b949e !important; font-size: 0.82rem !important; }
    .stButton > button { background: #21262d; border: 1px solid #30363d; color: #e6edf3; border-radius: 8px; }
    .stButton > button:hover { background: #30363d; border-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Knowledge base ────────────────────────────────────────────────────────────
DISEASE_KNOWLEDGE = {
    "late blight": {
        "pathogen": "Phytophthora infestans (water mold)",
        "spread": "Wind-borne sporangia, rain splash, infected tubers",
        "treatment": "Chlorothalonil, mancozeb, metalaxyl fungicides",
        "risk": "severe",
        "icon": "🍂",
    },
    "early blight": {
        "pathogen": "Alternaria solani (fungus)",
        "spread": "Rain splash, wind, infected debris",
        "treatment": "Copper-based fungicides, crop rotation",
        "risk": "moderate",
        "icon": "🎯",
    },
    "bacterial spot": {
        "pathogen": "Xanthomonas spp. (bacterium)",
        "spread": "Rain splash, wind, infected seeds",
        "treatment": "Copper hydroxide sprays, resistant cultivars",
        "risk": "moderate",
        "icon": "🔬",
    },
    "leaf mold": {
        "pathogen": "Passalora fulva / Cladosporium fulvum (fungus)",
        "spread": "Airborne spores, high humidity greenhouse conditions",
        "treatment": "Reduce humidity, ventilation, iprodione fungicide",
        "risk": "mild",
        "icon": "🍃",
    },
    "septoria leaf spot": {
        "pathogen": "Septoria lycopersici (fungus)",
        "spread": "Rain splash from soil, infected debris",
        "treatment": "Remove infected leaves, chlorothalonil, mancozeb",
        "risk": "moderate",
        "icon": "⚪",
    },
    "spider mite damage": {
        "pathogen": "Tetranychus urticae (two-spotted spider mite)",
        "spread": "Wind, direct plant contact, infested transplants",
        "treatment": "Miticides, insecticidal soap, predatory mites",
        "risk": "mild",
        "icon": "🕷️",
    },
    "target spot": {
        "pathogen": "Corynespora cassiicola (fungus)",
        "spread": "Airborne conidia, infected debris",
        "treatment": "Azoxystrobin, boscalid + pyraclostrobin fungicides",
        "risk": "moderate",
        "icon": "🎯",
    },
    "mosaic virus": {
        "pathogen": "Tomato Mosaic Virus / Tobacco Mosaic Virus",
        "spread": "Mechanical contact, infected tools, seed",
        "treatment": "No cure — remove infected plants, control aphid vectors",
        "risk": "severe",
        "icon": "🧬",
    },
    "yellow leaf curl virus": {
        "pathogen": "Tomato Yellow Leaf Curl Virus (TYLCV, begomovirus)",
        "spread": "Silverleaf whitefly (Bemisia tabaci)",
        "treatment": "Whitefly management, reflective mulches, resistant varieties",
        "risk": "severe",
        "icon": "🟡",
    },
    "tomato yellowleaf curl virus": {
        "pathogen": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "spread": "Silverleaf whitefly (Bemisia tabaci)",
        "treatment": "Whitefly management, imidacloprid, resistant varieties",
        "risk": "severe",
        "icon": "🟡",
    },
    "healthy": {
        "pathogen": "None detected",
        "spread": "N/A",
        "treatment": "Continue regular monitoring and maintenance",
        "risk": "none",
        "icon": "✅",
    },
}

SEVERITY_BADGE = {
    "none":             ("badge-healthy",  "✅ Healthy"),
    "mild":             ("badge-mild",     "🟡 Mild"),
    "mild to moderate": ("badge-moderate", "🟠 Mild–Moderate"),
    "moderate":         ("badge-moderate", "🟠 Moderate"),
    "moderate to severe":("badge-severe",  "🔴 Moderate–Severe"),
    "severe":           ("badge-severe",   "🔴 Severe"),
}

# ── Caption parser ─────────────────────────────────────────────────────────────
STRUCTURED_PATTERN = re.compile(
    r"Plant:\s*(?P<plant>[^.]+)\.\s*"
    r"Condition:\s*(?P<condition>[^.]+)\.\s*"
    r"Severity:\s*(?P<severity>[^.]+)\.",
    re.IGNORECASE,
)

def parse_caption(caption):
    m = STRUCTURED_PATTERN.search(caption)
    if not m:
        return None
    return m.group("plant").strip(), m.group("condition").strip(), m.group("severity").strip()


# ── Model cache ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_blip2():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from peft import PeftModel
    ADAPTER = "outputs/blip2_plantvillage/best"
    BASE    = "Salesforce/blip2-opt-2.7b"
    proc    = Blip2Processor.from_pretrained(ADAPTER)
    base    = Blip2ForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return model, proc


@st.cache_resource
def load_florence2():
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel
    ADAPTER = "outputs/florence2_plantvillage/best"
    BASE    = "microsoft/Florence-2-base-ft"
    proc    = AutoProcessor.from_pretrained(ADAPTER, trust_remote_code=True)
    base    = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return model, proc


@st.cache_resource
def load_clip():
    from transformers import CLIPProcessor, CLIPModel
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").half()
    model.eval()
    return proc, model


KNOWN_CLASS_DESCRIPTIONS = [
    "a tomato leaf with late blight disease showing dark lesions",
    "a tomato leaf with early blight disease showing target spots",
    "a tomato leaf with bacterial spot showing dark dots and yellow halo",
    "a tomato leaf with leaf mold showing olive fungal growth",
    "a tomato leaf with septoria leaf spot showing small round lesions",
    "a tomato leaf damaged by spider mites showing stippling",
    "a tomato leaf with target spot disease showing ringed brown spots",
    "a tomato leaf infected with mosaic virus showing mottled green pattern",
    "a tomato leaf with yellow leaf curl virus showing curling and yellowing",
    "a healthy tomato plant leaf with uniform green color",
    "a potato leaf with late blight disease",
    "a potato leaf with early blight disease",
    "a healthy potato leaf",
    "a pepper bell leaf with bacterial spot disease",
    "a healthy pepper bell leaf",
]


@torch.no_grad()
def detect_ood(image, clip_proc, clip_model, threshold):
    text_inputs = clip_proc(text=KNOWN_CLASS_DESCRIPTIONS, return_tensors="pt", padding=True).to("cuda")
    text_embeds = clip_model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    img_inputs  = clip_proc(images=image, return_tensors="pt").to("cuda")
    img_embed   = clip_model.get_image_features(**img_inputs)
    img_embed   = img_embed / img_embed.norm(dim=-1, keepdim=True)

    sims      = (img_embed @ text_embeds.T).squeeze(0).float().cpu().numpy()
    max_sim   = float(sims.max())
    best_idx  = int(sims.argmax())
    best_desc = KNOWN_CLASS_DESCRIPTIONS[best_idx]
    return max_sim < threshold, max_sim, best_desc


def run_blip2(image):
    model, proc = load_blip2()
    inputs = proc(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            repetition_penalty=3.0,
            no_repeat_ngram_size=8,
            length_penalty=0.6,
            early_stopping=True,
        )
    raw   = proc.decode(out[0], skip_special_tokens=True).strip()
    sents = raw.split(". ")
    clean = ". ".join(sents[:3]).strip()
    return clean if clean.endswith(".") else clean + "."


def run_florence2(image):
    model, proc = load_florence2()
    inputs = proc(text="<CAPTION>", images=image, return_tensors="pt").to("cuda", torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=6,
            length_penalty=0.6,
            early_stopping=True,
        )
    raw = proc.decode(out[0], skip_special_tokens=True).strip().replace("<CAPTION>", "").strip()
    sents = raw.split(". ")
    clean = ". ".join(sents[:3]).strip()
    return clean if clean.endswith(".") else clean + "."


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 PlantVision AI")
    st.markdown("**VLM Plant Disease Detection**")
    st.divider()

    model_choice = st.selectbox(
        "VLM Model",
        ["BLIP-2 OPT-2.7B", "Florence-2-base", "Both (Compare)"],
        help="Select which fine-tuned vision-language model to use for diagnosis"
    )

    st.divider()
    enable_ood = st.toggle("Enable OOD Detection", value=True,
        help="Uses CLIP to detect if uploaded image is not a plant leaf")
    ood_threshold = st.slider(
        "OOD Confidence Threshold",
        min_value=0.20, max_value=0.70, value=0.45, step=0.01,
        disabled=not enable_ood,
        help="CLIP cosine similarity below this → flagged as out-of-distribution"
    )

    st.divider()
    st.markdown("**Supported Plants**")
    st.markdown("🍅 Tomato · 🥔 Potato · 🫑 Pepper Bell")
    st.markdown("**Disease Classes**")
    st.markdown("15 classes across 3 plant species")


# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_diag, tab_eval, tab_compare = st.tabs(["🔬 Diagnosis", "📊 Evaluation", "⚖️ Model Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("### Upload a Plant Leaf Image")
    st.markdown("<p style='color:#8b949e;font-size:0.85rem;'>Supports JPG, PNG. Works best with clear leaf images.</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        col_img, col_result = st.columns([1, 1.4], gap="large")

        with col_img:
            st.image(image, use_column_width=True, caption=uploaded.name)

        with col_result:
            # OOD check first
            ood_flagged = False
            clip_sim    = None
            if enable_ood:
                clip_proc, clip_model = load_clip()
                with st.spinner("Running OOD check…"):
                    ood_flagged, clip_sim, best_desc = detect_ood(
                        image, clip_proc, clip_model, ood_threshold
                    )

            if ood_flagged:
                st.markdown(f"""
                <div class="ood-warning">
                    ⚠️ <strong>Out-of-Distribution Detected</strong><br/>
                    This image does not appear to be a plant leaf from our known classes.<br/>
                    <span style="font-size:0.8rem;font-weight:400;color:#8b949e;">
                    CLIP similarity: {clip_sim:.3f} (threshold: {ood_threshold})<br/>
                    Closest match: {best_desc}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Run selected model(s)
                models_to_run = []
                if model_choice == "BLIP-2 OPT-2.7B":
                    models_to_run = [("BLIP-2 OPT-2.7B", run_blip2)]
                elif model_choice == "Florence-2-base":
                    models_to_run = [("Florence-2-base", run_florence2)]
                else:
                    models_to_run = [("BLIP-2 OPT-2.7B", run_blip2), ("Florence-2-base", run_florence2)]

                captions = {}
                for mname, mfn in models_to_run:
                    with st.spinner(f"Running {mname}…"):
                        captions[mname] = mfn(image)

                # Display results
                for mname, caption in captions.items():
                    parsed = parse_caption(caption)

                    st.markdown(f"**{mname}**")

                    if parsed:
                        plant, condition, severity = parsed
                        sev_key = severity.lower()
                        badge_cls, badge_text = SEVERITY_BADGE.get(
                            sev_key, ("badge-moderate", "🟠 Unknown")
                        )
                        st.markdown(
                            f'<span class="badge {badge_cls}">{badge_text}</span>',
                            unsafe_allow_html=True
                        )

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="val">{plant}</div>
                                <div class="lbl">Plant</div>
                            </div>""", unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="val" style="font-size:1rem;">{condition}</div>
                                <div class="lbl">Condition</div>
                            </div>""", unsafe_allow_html=True)

                        # Knowledge card
                        cond_key = condition.lower()
                        if cond_key in DISEASE_KNOWLEDGE:
                            info = DISEASE_KNOWLEDGE[cond_key]
                            st.markdown(f"""
                            <div class="knowledge-card">
                                <h4>{info['icon']} Disease Profile</h4>
                                <b>Pathogen:</b> {info['pathogen']}<br/>
                                <b>Spread:</b> {info['spread']}<br/>
                                <b>Treatment:</b> {info['treatment']}
                            </div>""", unsafe_allow_html=True)

                    # Raw caption
                    cap_html = caption.replace("Plant:", "<span>Plant:</span>")
                    cap_html = cap_html.replace("Condition:", "<span>Condition:</span>")
                    cap_html = cap_html.replace("Severity:", "<span>Severity:</span>")
                    st.markdown(
                        f'<div class="caption-box">{cap_html}</div>',
                        unsafe_allow_html=True
                    )

                    if clip_sim is not None:
                        st.markdown(
                            f'<p style="color:#8b949e;font-size:0.78rem;margin-top:0.4rem;">CLIP similarity: {clip_sim:.3f}</p>',
                            unsafe_allow_html=True
                        )

                    if len(models_to_run) > 1:
                        st.divider()
    else:
        st.markdown("""
        <div style="border:2px dashed #30363d;border-radius:12px;padding:3rem;text-align:center;color:#8b949e;margin-top:1rem;">
            <div style="font-size:3rem;">🌿</div>
            <div style="font-size:1rem;margin-top:0.5rem;">Upload a plant leaf image to begin diagnosis</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Supported: JPG, PNG</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    eval_file = Path("evaluation_results.json")
    if eval_file.exists():
        with open(eval_file) as f:
            raw = json.load(f)

        # Normalise: could be a list or a single dict
        if isinstance(raw, dict):
            all_evals = [raw]
        else:
            all_evals = raw

        for res in all_evals:
            st.markdown(f"### 📊 {res['model']}")
            c1, c2, c3, c4 = st.columns(4)
            def mc(col, val, lbl):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="val">{val}</div>
                    <div class="lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            mc(c1, f"{res['bleu4']:.3f}",    "BLEU-4")
            mc(c2, f"{res['rouge_l']:.3f}",  "ROUGE-L")
            mc(c3, f"{res['macro_f1']:.3f}", "Macro F1")
            mc(c4, f"{res['severity_r2']:.3f}", "Severity R²")

            c5, c6, c7, c8 = st.columns(4)
            mc(c5, f"{res['bleu1']:.3f}",          "BLEU-1")
            mc(c6, f"{res['accuracy']:.3f}",       "Accuracy")
            mc(c7, f"{res['macro_precision']:.3f}","Precision")
            mc(c8, f"{res['macro_recall']:.3f}",   "Recall")

            # Per-class F1
            if res.get("per_class_f1"):
                st.markdown("**Per-class F1 Score**")
                sorted_classes = sorted(res["per_class_f1"].items(), key=lambda x: x[1], reverse=True)
                for cls, score in sorted_classes:
                    col_lbl, col_bar, col_val = st.columns([3, 5, 1])
                    col_lbl.markdown(f"<span style='font-size:0.82rem;color:#8b949e;'>{cls}</span>", unsafe_allow_html=True)
                    col_bar.progress(float(score))
                    col_val.markdown(f"<span style='color:#58a6ff;font-size:0.82rem;'>{score:.3f}</span>", unsafe_allow_html=True)

            st.divider()
    else:
        st.info("No evaluation results found. Run `python 04_evaluate.py --model blip2` to generate them.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    eval_file = Path("evaluation_results.json")
    if eval_file.exists():
        with open(eval_file) as f:
            raw = json.load(f)
        all_evals = raw if isinstance(raw, list) else [raw]

        if len(all_evals) >= 2:
            res_a, res_b = all_evals[0], all_evals[1]
            metrics = [
                ("BLEU-1",         "bleu1"),
                ("BLEU-4",         "bleu4"),
                ("ROUGE-L",        "rouge_l"),
                ("Accuracy",       "accuracy"),
                ("Macro F1",       "macro_f1"),
                ("Precision",      "macro_precision"),
                ("Recall",         "macro_recall"),
                ("Severity R²",    "severity_r2"),
                ("CLIP Sim Mean",  "clip_sim_mean"),
            ]

            st.markdown("### ⚖️ Side-by-Side Comparison")
            hdr = st.columns([3, 2, 2, 1])
            hdr[0].markdown("**Metric**")
            hdr[1].markdown(f"**{res_a['model'][:18]}**")
            hdr[2].markdown(f"**{res_b['model'][:18]}**")
            hdr[3].markdown("**Winner**")

            for label, key in metrics:
                va, vb = res_a.get(key, 0), res_b.get(key, 0)
                winner = res_a["model"][:12] if va >= vb else res_b["model"][:12]
                cols   = st.columns([3, 2, 2, 1])
                cols[0].markdown(f"<span style='color:#8b949e;'>{label}</span>", unsafe_allow_html=True)
                color_a = "#3fb950" if va >= vb else "#8b949e"
                color_b = "#3fb950" if vb > va  else "#8b949e"
                cols[1].markdown(f"<span style='color:{color_a};font-weight:600;'>{va:.4f}</span>", unsafe_allow_html=True)
                cols[2].markdown(f"<span style='color:{color_b};font-weight:600;'>{vb:.4f}</span>", unsafe_allow_html=True)
                cols[3].markdown(f"<span style='color:#58a6ff;font-size:0.8rem;'>{'↑ A' if va >= vb else '↑ B'}</span>", unsafe_allow_html=True)
        else:
            st.info("Run evaluation for both models to see a comparison:\n```\npython 04_evaluate.py --model blip2 --compare florence2\n```")
    else:
        st.info("Run `python 04_evaluate.py --model blip2 --compare florence2` first to populate this tab.")