"""Reusable UI components — Streamlit-native, with a small amount of CSS.

Presentation only: these render dicts produced by ``app.inference`` /
``app.history`` / ``app.evaluation_view`` and never do I/O or inference.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.evaluation_view import EvalSummary
from app.utils import (
    FINE_TUNING,
    FRAMEWORK,
    MODEL_DISPLAY,
    MODEL_ID,
    format_seconds,
    pretty_disease,
)

# status -> (emoji, hex color, headline word)
_STATUS_STYLE = {
    "confident": ("🩺", "#16a34a", "Diagnosis"),
    "low_confidence": ("⚠️", "#f59e0b", "Low-confidence diagnosis"),
    "unknown": ("❓", "#6b7280", "Unknown"),
}

_CSS = """
<style>
:root {
  --pdx-muted: #6b7280;
  --pdx-border: rgba(148, 163, 184, 0.25);
  --pdx-surface: rgba(148, 163, 184, 0.06);
}
.pdx-hero { padding: 0.25rem 0 0.75rem 0; }
.pdx-hero h1 { margin-bottom: 0.1rem; font-weight: 700; letter-spacing: -0.02em; }
.pdx-sub { color: var(--pdx-muted); font-size: 0.95rem; margin-top: 0; }
.pdx-chip {
  display: inline-block; padding: 0.15rem 0.6rem; margin: 0.15rem 0.35rem 0.15rem 0;
  border: 1px solid var(--pdx-border); border-radius: 999px;
  font-size: 0.8rem; color: var(--pdx-muted); background: var(--pdx-surface);
}
.pdx-conf-track {
  width: 100%; height: 12px; border-radius: 999px;
  background: var(--pdx-surface); overflow: hidden; margin: 0.35rem 0 0.15rem 0;
}
.pdx-conf-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
.pdx-conf-row { display: flex; justify-content: space-between; align-items: baseline; }
.pdx-conf-pct { font-weight: 700; font-size: 1.05rem; }
.pdx-conf-label {
  color: var(--pdx-muted); font-size: 0.85rem;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.pdx-diag { font-size: 1.05rem; line-height: 1.5; }
.pdx-muted { color: var(--pdx-muted); }
</style>
"""


def inject_css() -> None:
    """Inject the app's small style block once per session."""
    if not st.session_state.get("_pdx_css"):
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state["_pdx_css"] = True


def header() -> None:
    """The top hero banner."""
    st.markdown(
        '<div class="pdx-hero">'
        "<h1>🌿 PlantDx</h1>"
        '<p class="pdx-sub">Knowledge-grounded leaf-disease captioning for tomato & mango — '
        "upload a leaf, get a grounded diagnosis.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def confidence_meter(record: dict[str, Any]) -> None:
    """A status-colored confidence bar: green / orange / grey."""
    status = str(record.get("status", "unknown"))
    _emoji, color, _word = _STATUS_STYLE.get(status, _STATUS_STYLE["unknown"])
    conf = record.get("confidence")
    threshold = record.get("confidence_threshold")

    if conf is None:
        st.markdown(
            '<div class="pdx-conf-row"><span class="pdx-conf-label">Confidence</span></div>'
            '<div class="pdx-muted" style="font-size:0.88rem;">Not available.</div>',
            unsafe_allow_html=True,
        )
        return

    pct = max(0.0, min(1.0, conf)) * 100
    caption = {
        "confident": "generation certainty",
        "low_confidence": "below threshold",
        "unknown": "no disease named",
    }.get(status, "")
    thr = f" · threshold {threshold * 100:.0f}%" if isinstance(threshold, int | float) else ""
    st.markdown(
        f'<div class="pdx-conf-row">'
        f'<span class="pdx-conf-label">Confidence · {html.escape(caption)}{thr}</span>'
        f'<span class="pdx-conf-pct" style="color:{color};">{pct:.1f}%</span>'
        f"</div>"
        f'<div class="pdx-conf-track">'
        f'<div class="pdx-conf-fill" style="width:{pct:.1f}%; background:{color};"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def result_card(record: dict[str, Any]) -> None:
    """Render one prediction as a polished, bordered, status-aware card."""
    status = str(record.get("status", "unknown"))
    emoji, _color, word = _STATUS_STYLE.get(status, _STATUS_STYLE["unknown"])
    disease = record.get("disease_name") or pretty_disease(record.get("disease_id") or "")
    heading = "Unknown condition" if status == "unknown" else disease
    crop = str(record.get("crop", "")).capitalize()

    with st.container(border=True):
        top = st.columns([3, 2])
        with top[0]:
            st.markdown(f"### {emoji} {heading}")
            st.caption(word)
            if status != "unknown" and record.get("common_name"):
                st.markdown(
                    f'<span class="pdx-muted">{html.escape(record["common_name"])}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="pdx-diag">{record.get("diagnosis", "")}</div>',
                unsafe_allow_html=True,
            )
        with top[1]:
            confidence_meter(record)

        st.divider()

        meta = st.columns(4)
        secs = record.get("inference_seconds")
        meta[0].metric(
            "Inference time",
            format_seconds(secs) if isinstance(secs, int | float) else "—",
        )
        meta[1].metric("Crop", crop or "—")
        meta[2].metric("Tokens", record.get("generation_tokens") or "—")
        mem = record.get("peak_memory_gb")
        meta[3].metric("Peak mem", f"{mem:.1f} GB" if isinstance(mem, int | float) else "—")

        st.markdown("**Generated caption**")
        st.info(record.get("caption") or "_(empty)_")

        symptoms = record.get("symptoms") or []
        if symptoms and status == "confident":
            st.markdown(
                "**Documented symptoms** "
                '<span class="pdx-muted">(from the Disease Knowledge Base)</span>',
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {html.escape(str(s))}" for s in symptoms))

        with st.expander("Model & adapter details"):
            st.markdown(
                f'<span class="pdx-chip">Model · {html.escape(MODEL_DISPLAY)}</span>'
                f'<span class="pdx-chip">Fine-tuning · {FINE_TUNING}</span>'
                f'<span class="pdx-chip">Framework · {FRAMEWORK}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Base model: `{record.get('model', MODEL_ID)}`")
            st.caption(f"Adapter run: `{record.get('run_name', '—')}`")
            st.caption(f"Adapter path: `{record.get('adapter', '—')}`")
            st.caption(f"Instruction: {record.get('instruction', '—')}")


def adapter_panel(info: dict[str, Any], *, run_name: str, crop: str) -> None:
    """Prove the trained LoRA adapter — not the base model — is attached."""
    attached = bool(info.get("attached"))
    if attached:
        st.success("LoRA adapter attached", icon="✅")
    else:
        st.error("LoRA adapter NOT detected — base model only", icon="⛔")

    params = info.get("trainable_params")
    rows = {
        "Crop": crop.capitalize(),
        "Base model": MODEL_ID,
        "Adapter run": run_name,
        "Checkpoint": info.get("adapter_dir", "—"),
        "Fine-tune type": info.get("fine_tune_type", "—"),
        "LoRA rank": info.get("rank", "—"),
        "LoRA scale": info.get("scale", "—"),
        "Adapted modules": info.get("num_target_modules", "—"),
        "Trainable params": f"{params:,}" if isinstance(params, int) else "—",
        "Adapter tensors": info.get("lora_tensor_count", "—"),
        "Weights checksum": info.get("weights_checksum", "—"),
    }
    st.table({"Field": list(rows.keys()), "Value": [str(v) for v in rows.values()]})


def evaluation_panel(summary: EvalSummary | None, per_disease: list[dict[str, Any]]) -> None:
    """Show the crop's held-out evaluation results (base vs fine-tuned)."""
    if summary is None:
        st.caption(
            "No evaluation report found for this crop yet. Run "
            "`plantdx evaluate --stage all --adapter checkpoints/<run>` to generate one."
        )
        return

    st.caption(
        f"Held-out **test split** ({summary.sample_count} images) from "
        f"`{summary.run_name}` — PlantVillage-style images."
    )
    cols = st.columns(2)
    cols[0].metric(
        "Accuracy (fine-tuned)",
        f"{summary.finetuned_accuracy * 100:.1f}%",
        delta=f"{(summary.finetuned_accuracy - summary.base_accuracy) * 100:+.1f} pts vs base",
    )
    cols[1].metric(
        "Macro-F1 (fine-tuned)",
        f"{summary.finetuned_f1_macro:.3f}",
        delta=f"{(summary.finetuned_f1_macro - summary.base_f1_macro):+.3f} vs base",
    )
    if per_disease:
        st.markdown("**Per-disease accuracy (fine-tuned)**")
        st.dataframe(
            {
                "Disease": [pretty_disease(r["disease_id"]) for r in per_disease],
                "Samples": [r["sample_count"] for r in per_disease],
                "Accuracy": [f"{r['accuracy'] * 100:.0f}%" for r in per_disease],
                "F1": [f"{r['f1']:.2f}" for r in per_disease],
            },
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "These are in-distribution scores. Casual field/phone photos differ from "
        "the training data and will score lower."
    )


def about_panel() -> None:
    """The sidebar/expander About block."""
    st.markdown(
        f"""
**PlantDx** — a knowledge-grounded framework for agricultural vision-language models.

| | |
|---|---|
| **Model** | {MODEL_DISPLAY} |
| **Fine-tuning** | {FINE_TUNING} |
| **Framework** | {FRAMEWORK} |
| **Crops** | Tomato · Mango |

Every diagnosis restates a class the model was trained on; the listed symptoms
are read verbatim from the curated Disease Knowledge Base, never generated.
This demo runs the already-trained adapters and changes nothing in the pipeline.
        """
    )
