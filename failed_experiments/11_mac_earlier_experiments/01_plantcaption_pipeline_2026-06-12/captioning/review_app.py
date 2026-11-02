"""
review_app.py
=============
A Streamlit review console for the generated captions. This is the artefact to
demo to your supervisor: it shows, per image, the two halves of the model side
by side --

    Part A  (agriculture)  : crop, disease, pathogen, severity, symptoms
    Part B  (image pattern): shot, lighting, hand/object, background, focus

-- followed by the five captions, each editable. You can approve images, edit
any caption in place, and export a cleaned train.jsonl containing only approved
records.

Run:
    streamlit run review_app.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import streamlit as st

import config

st.set_page_config(page_title="PlantCaption Review", layout="wide")

# --- light styling to match your deep-forest theme -------------------------- #
st.markdown(
    """
    <style>
      .stApp { background-color: #0d1410; }
      h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; color: #e8efe4; }
      .mono { font-family: 'DM Mono', monospace; color: #9fc3a0; font-size: 0.85rem; }
      .pill { display:inline-block; padding:2px 10px; border-radius:12px;
              font-family:'DM Mono',monospace; font-size:0.78rem; margin-right:6px; }
      .sev-healthy  { background:#1d3b27; color:#9be8b0; }
      .sev-mild     { background:#27401d; color:#cbe89b; }
      .sev-moderate { background:#41391d; color:#e8d39b; }
      .sev-severe   { background:#41231d; color:#e89b9b; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_master(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sev_pill(sev: str) -> str:
    return f'<span class="pill sev-{sev}">{sev.upper()}</span>'


def main() -> None:
    st.title("PlantCaption — caption review console")

    if not os.path.exists(config.MASTER_JSON):
        st.error(
            f"No captions found. Run the pipeline first:\n\n"
            f"`python run_pipeline.py --backend qwen`"
        )
        return

    master = load_master(config.MASTER_JSON)

    # session state holds edits + approvals
    if "edits" not in st.session_state:
        st.session_state.edits = {}      # image_id -> list[str] edited captions
    if "approved" not in st.session_state:
        st.session_state.approved = {}   # image_id -> bool

    # ---- sidebar: filters + per-class stats ------------------------------- #
    with st.sidebar:
        st.header("Filters")
        crops = sorted({r["agronomy"]["crop"] for r in master})
        sel_crops = st.multiselect("Crop", crops, default=crops)
        sevs = ["healthy", "mild", "moderate", "severe"]
        sel_sev = st.multiselect("Severity", sevs, default=sevs)

        st.divider()
        st.header("Dataset")
        st.markdown(f"<span class='mono'>{len(master)} images · "
                    f"{len(master) * config.CAPTIONS_PER_IMAGE} captions</span>",
                    unsafe_allow_html=True)
        n_appr = sum(1 for v in st.session_state.approved.values() if v)
        st.markdown(f"<span class='mono'>{n_appr} approved</span>",
                    unsafe_allow_html=True)

        st.divider()
        if st.button("Export approved -> train_reviewed.jsonl", use_container_width=True):
            _export(master)

    view = [r for r in master
            if r["agronomy"]["crop"] in sel_crops and r["severity"] in sel_sev]
    st.caption(f"Showing {len(view)} of {len(master)} images")

    # ---- per-image cards -------------------------------------------------- #
    for rec in view:
        iid = rec["image_id"]
        with st.container(border=True):
            left, right = st.columns([1, 2], gap="large")

            with left:
                img_abs = os.path.join(config.PROJECT_ROOT, rec["image_path"])
                if os.path.exists(img_abs):
                    st.image(img_abs, use_container_width=True)
                else:
                    st.warning(f"image not found: {rec['image_path']}")
                st.markdown(sev_pill(rec["severity"]), unsafe_allow_html=True)
                st.session_state.approved[iid] = st.checkbox(
                    "Approve", value=st.session_state.approved.get(iid, False),
                    key=f"appr_{iid}")

            with right:
                a, b = rec["agronomy"], rec["visual"]
                st.markdown(f"### {a['crop'].title()} — {a['disease']}")
                st.markdown(
                    f"<span class='mono'>PART A · pathogen: {a['pathogen']} "
                    f"({a['pathogen_type']})</span>", unsafe_allow_html=True)
                st.markdown(
                    f"<span class='mono'>PART B · {b['shot_type']} · {b['lighting']} · "
                    f"hand/object: {b['hand_or_object']} · bg: {b['background']} · "
                    f"{b['estimated_pct_affected']}% affected</span>",
                    unsafe_allow_html=True)

                st.write("")
                edited = st.session_state.edits.get(
                    iid, [c["text"] for c in rec["captions"]])
                new_edited = []
                for i, cap in enumerate(rec["captions"]):
                    val = st.text_area(
                        f"{i+1}. {cap['archetype']}",
                        value=edited[i],
                        key=f"cap_{iid}_{i}",
                        height=80)
                    new_edited.append(val)
                st.session_state.edits[iid] = new_edited


def _export(master: List[Dict]) -> None:
    out_path = os.path.join(config.OUTPUTS_DIR, "train_reviewed.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in master:
            iid = rec["image_id"]
            if not st.session_state.approved.get(iid, False):
                continue
            caps = st.session_state.edits.get(
                iid, [c["text"] for c in rec["captions"]])
            for i, cap in enumerate(rec["captions"]):
                row = {
                    "image": rec["image_path"],
                    "messages": [
                        {"role": "user", "content": f"<image>\n{config.INSTRUCTION}"},
                        {"role": "assistant", "content": caps[i]},
                    ],
                    "meta": {
                        "image_id": iid,
                        "crop": rec["agronomy"]["crop"],
                        "disease": rec["agronomy"]["disease"],
                        "severity": rec["severity"],
                        "archetype": cap["archetype"],
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    st.success(f"Exported {n} approved captions to {out_path}")


if __name__ == "__main__":
    main()
