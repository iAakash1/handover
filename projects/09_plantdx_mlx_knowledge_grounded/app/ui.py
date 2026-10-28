"""Page layout + orchestration: sidebar, upload/inference flow, history, panels.

The only module that reads Streamlit widget state. It drives one prediction per
uploaded file (cached in ``st.session_state`` so a rerun never re-runs the model),
files each original immutably under its predicted class, writes a metadata
sidecar + prediction record + append-only log, and surfaces adapter verification
and held-out evaluation results in dedicated tabs.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from app import history, storage
from app.components import (
    about_panel,
    adapter_panel,
    evaluation_panel,
    result_card,
)
from app.evaluation_view import load_eval_summary, load_per_disease
from app.inference import (
    adapter_info,
    load_for_crop,
    mlx_import_status,
    resolve_crop_adapter,
    run_inference,
)
from app.logging_setup import get_logger
from app.utils import (
    ALLOWED_UPLOAD_TYPES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    FINE_TUNING,
    FRAMEWORK,
    MODEL_DISPLAY,
    crop_profile,
    is_supported_upload,
    pretty_disease,
)
from plantdx.core.exceptions import PlantDxError

_SESSION_RESULTS = "session_results"  # dict[key -> record]
_REOPENED = "reopened_record"
_WARM = "warm_crops"  # dict[crop -> load_seconds]
_log = get_logger()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar and return the chosen settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        crop = st.radio(
            "Select crop",
            options=("tomato", "mango"),
            format_func=lambda c: f"{crop_profile(c).emoji}  {c.capitalize()}",
            horizontal=True,
            help="Loads that crop's trained adapter.",
        )
        profile = crop_profile(crop)
        adapter_ready = profile.adapter_dir.exists()

        st.markdown("### Model")
        st.caption(f"**{MODEL_DISPLAY}**")
        st.caption(f"Fine-tuning · {FINE_TUNING}  ·  Framework · {FRAMEWORK}")

        st.markdown("### Adapter")
        warm = st.session_state.get(_WARM, {})
        if not adapter_ready:
            st.warning(f"`{profile.run_name}` not found", icon="⚠️")
        elif crop in warm:
            st.success(f"`{profile.run_name}` · warm ({warm[crop]:.0f}s load)", icon="✅")
        else:
            st.info(f"`{profile.run_name}` · loads on first prediction", icon="🧊")

        st.markdown("### Device")
        st.caption("Apple MLX (GPU, auto) · greedy / deterministic decoding")

        st.markdown("### Inference settings")
        max_tokens = st.slider(
            "Max new tokens", 32, 256, 128, 16, help="Upper bound on the caption length."
        )
        threshold = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            DEFAULT_CONFIDENCE_THRESHOLD,
            0.05,
            help=(
                "A named prediction below this generation-confidence is shown as "
                "low-confidence (likely out-of-distribution) rather than asserted."
            ),
        )

        st.divider()
        _render_history_sidebar()

        st.divider()
        with st.expander("About PlantDx"):
            about_panel()

    return {
        "crop": crop,
        "max_tokens": max_tokens,
        "confidence_threshold": threshold,
        "adapter_ready": adapter_ready,
    }


def _render_history_sidebar() -> None:
    st.markdown("### 🕑 History")
    entries = history.list_history()
    if not entries:
        st.caption("No predictions yet.")
        return

    for i, entry in enumerate(entries[:15]):
        disease = entry.get("disease_name") or pretty_disease(entry.get("disease_id") or "")
        crop = entry.get("crop")
        emoji = crop_profile(crop).emoji if crop in ("tomato", "mango") else "•"
        ts = str(entry.get("timestamp", ""))[:19].replace("T", " ")
        if st.button(f"{emoji} {disease} · {ts}", key=f"hist_{i}", width="stretch"):
            record = history.load_record(entry.get("prediction_json", ""))
            st.session_state[_REOPENED] = record or entry


# --------------------------------------------------------------------------- #
# Main panel (tabs)
# --------------------------------------------------------------------------- #


def render_main(settings: dict[str, Any]) -> None:
    """Render the Diagnose / Adapter / Evaluation tabs."""
    tab_labels = ["🔬 Diagnose", "🧬 Adapter & model", "📊 Evaluation"]
    diagnose, adapter_tab, eval_tab = st.tabs(tab_labels)
    with diagnose:
        _render_diagnose(settings)
    with adapter_tab:
        _render_adapter_tab(settings)
    with eval_tab:
        _render_evaluation_tab(settings)


def _render_diagnose(settings: dict[str, Any]) -> None:
    crop = settings["crop"]
    profile = crop_profile(crop)

    st.markdown(
        f"#### Upload {profile.emoji} **{profile.label}** leaf images "
        "<span style='opacity:0.6;font-weight:400;'>— JPG · JPEG · PNG · multiple allowed</span>",
        unsafe_allow_html=True,
    )

    ok, reason = _env_status()
    if not ok:
        st.error(_env_help(reason), icon="🧩")

    uploads = st.file_uploader(
        "Drag & drop or browse",
        type=ALLOWED_UPLOAD_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not settings["adapter_ready"]:
        st.info(
            f"The **{profile.label}** adapter (`{profile.adapter_dir}`) isn't present in this "
            "checkout, so inference is disabled for this crop. Uploads are still saved.",
            icon="⚠️",
        )

    _render_reopened()

    if not uploads:
        if not history.list_history():
            _empty_state()
        return

    for file in uploads:
        _process_one(file, settings)


def _render_adapter_tab(settings: dict[str, Any]) -> None:
    crop = settings["crop"]
    profile = crop_profile(crop)
    st.markdown("#### Adapter verification")
    if not settings["adapter_ready"]:
        st.warning(f"No `{profile.run_name}` checkpoint present in this checkout.", icon="⚠️")
        return
    try:
        info = adapter_info(str(resolve_crop_adapter(crop)))
    except PlantDxError as exc:
        st.error(str(exc), icon="⚠️")
        return
    adapter_panel(info, run_name=profile.run_name, crop=crop)

    warm = st.session_state.get(_WARM, {})
    st.markdown("#### Runtime")
    cols = st.columns(2)
    cols[0].metric("Model status", "Warm ✅" if crop in warm else "Cold 🧊")
    cols[1].metric("Load time", f"{warm[crop]:.1f} s" if crop in warm else "—")
    st.caption(
        "The base model + adapter are loaded once and cached for the whole session "
        "(`st.cache_resource`, one model resident at a time). Predictions reuse the "
        "warm model — nothing reloads per click."
    )


def _render_evaluation_tab(settings: dict[str, Any]) -> None:
    crop = settings["crop"]
    st.markdown(f"#### Held-out evaluation — {crop.capitalize()}")
    evaluation_panel(load_eval_summary(crop), load_per_disease(crop))


# --------------------------------------------------------------------------- #
# One uploaded file
# --------------------------------------------------------------------------- #


def _process_one(file: Any, settings: dict[str, Any]) -> None:
    crop = settings["crop"]
    data = file.getvalue()
    key = ":".join(
        str(x)
        for x in (
            crop,
            file.name,
            len(data),
            settings["max_tokens"],
            settings["confidence_threshold"],
        )
    )

    with st.container(border=True):
        cols = st.columns([1, 1])
        with cols[0]:
            if not is_supported_upload(file.name):
                st.error(f"Unsupported file type: **{file.name}**. Use JPG, JPEG, PNG.", icon="🚫")
                return
            valid, why = _validate_image(data)
            if not valid:
                st.error(f"Couldn't read **{file.name}**: {why}", icon="🖼️")
                return
            st.image(data, caption=file.name, width="stretch")

        with cols[1]:
            store = st.session_state.setdefault(_SESSION_RESULTS, {})
            if key in store:
                record = store[key]
            else:
                record = _run_and_persist(file, data, settings)
                if record is None:
                    return
                store[key] = record

    result_card(record)
    _download_buttons(record, key)


def _run_and_persist(file: Any, data: bytes, settings: dict[str, Any]) -> dict[str, Any] | None:
    crop = settings["crop"]
    profile = crop_profile(crop)

    if not settings["adapter_ready"]:
        # No adapter: still keep the original (unknown bucket), skip inference.
        try:
            storage.save_upload(data, file.name, crop, "unknown")
        except OSError as exc:
            st.error(f"Could not save the upload: {exc}", icon="💾")
        st.warning("Saved. Inference skipped — adapter not available.", icon="⏸️")
        return None

    ok, reason = _env_status()
    if not ok:
        st.error(_env_help(reason), icon="🧩")
        return None

    load_msg = (
        "Running inference…"
        if crop in st.session_state.get(_WARM, {})
        else f"Loading {profile.label} model (first prediction)…"
    )
    try:
        with st.spinner(load_msg):
            handle = load_for_crop(crop)
            st.session_state.setdefault(_WARM, {})[crop] = handle.load_seconds
    except PlantDxError as exc:
        st.error(_friendly(str(exc)), icon="🧩")
        return None

    result = _infer_on_bytes(handle, data, file.name, crop, settings)
    if result is None:
        return None

    # File the original under its predicted class (confident) or "unknown".
    subfolder = storage.upload_subfolder(result["disease_id"], crop, result["status"])
    try:
        image_path = storage.save_upload(data, file.name, crop, subfolder)
    except OSError as exc:
        st.error(f"Prediction done but could not save the image: {exc}", icon="💾")
        return result

    info = _safe_adapter_info(crop)
    record = storage.build_record(
        crop=crop, image_path=image_path, original_name=file.name, result=result, adapter=info
    )
    try:
        storage.write_sidecar(image_path, record)
        prediction_path = storage.write_prediction(record)
        record["prediction_json"] = str(prediction_path)
        history.append_history(record)
        storage.append_prediction_log(record)
    except OSError as exc:
        st.warning(f"Result computed but not fully saved: {exc}", icon="💾")
    return record


def _infer_on_bytes(
    handle: Any, data: bytes, name: str, crop: str, settings: dict[str, Any]
) -> dict[str, Any] | None:
    """Run inference on the raw bytes via a temp file (image saved after, by class)."""
    suffix = Path(name).suffix or ".jpg"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with st.spinner("Running inference…"):
            return run_inference(
                handle,
                tmp_path,
                crop,
                max_tokens=settings["max_tokens"],
                confidence_threshold=settings["confidence_threshold"],
            )
    except PlantDxError as exc:
        st.error(_friendly(str(exc)), icon="⚠️")
        return None
    except Exception as exc:  # never crash the app on an unexpected model error
        _log.exception("inference failed for %s", name)
        st.error(f"Inference failed unexpectedly: {exc}", icon="⚠️")
        return None
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _safe_adapter_info(crop: str) -> dict[str, Any] | None:
    try:
        return adapter_info(str(resolve_crop_adapter(crop)))
    except PlantDxError:
        return None


def _download_buttons(record: dict[str, Any], key: str) -> None:
    disease = record.get("disease_name") or pretty_disease(record.get("disease_id") or "")
    slug = record.get("timestamp_slug") or "prediction"
    cols = st.columns(2)
    cols[0].download_button(
        "⬇️  Download result (JSON)",
        data=storage.result_to_json(record),
        file_name=f"{slug}.json",
        mime="application/json",
        width="stretch",
        key=f"dl_json_{key}",
    )
    cols[1].download_button(
        "⬇️  Download report (Markdown)",
        data=storage.result_to_markdown(record),
        file_name=f"{slug}_{disease.replace(' ', '_').lower()}.md",
        mime="text/markdown",
        width="stretch",
        key=f"dl_md_{key}",
    )


def _render_reopened() -> None:
    record = st.session_state.get(_REOPENED)
    if not record:
        return
    st.markdown("#### 🔁 Reopened prediction")
    top = st.columns([1, 2])
    image_path = record.get("image_path")
    with top[0]:
        try:
            if image_path and Path(image_path).is_file():
                st.image(image_path, width="stretch")
            else:
                st.caption("Original image unavailable.")
        except Exception:
            st.caption("Original image unavailable.")
    with top[1]:
        result_card(record)
    _download_buttons(record, key=f"reopen_{record.get('timestamp_slug', 'x')}")
    if st.button("Close", key="close_reopened"):
        st.session_state.pop(_REOPENED, None)
        st.rerun()
    st.divider()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _env_status() -> tuple[bool, str]:
    """Whether the model backend can load here — probed once per session."""
    if "_mlx_status" not in st.session_state:
        st.session_state["_mlx_status"] = mlx_import_status()
    status: tuple[bool, str] = st.session_state["_mlx_status"]
    return status


def _validate_image(data: bytes) -> tuple[bool, str]:
    """Verify the bytes decode as an image (catches corrupt/empty uploads)."""
    try:
        from PIL import Image

        Image.open(io.BytesIO(data)).verify()  # type: ignore[no-untyped-call]
        return True, ""
    except Exception as exc:  # PIL raises many types on bad data
        return False, f"{type(exc).__name__}"


def _friendly(message: str) -> str:
    if "mlx-vlm is not importable" in message:
        return _env_help("numpy_abi")
    return message


def _env_help(reason: str) -> str:
    """A precise, actionable message for why the model can't load here."""
    if reason == "numpy_abi":
        return (
            "**mlx-vlm** is installed here but can't import — its `transformers` / "
            "`numba` dependency chain hits a NumPy 1.x-vs-2.x ABI conflict in **this "
            "Python interpreter** (a known environment issue, not an app or model "
            "problem; see `docs/EVALUATION.md#troubleshooting`).\n\n"
            "**Launch the app with the interpreter that already has a working ML "
            "stack + mlx-vlm** — the same one training/evaluation use. Because "
            "`conda activate` may not change which `streamlit` runs, use its **full "
            "path** to be certain:\n\n"
            "```bash\n"
            "~/miniforge3/envs/vlm/bin/python -m streamlit run streamlit_app.py\n"
            "```\n\n"
            "(Plain `streamlit run …` can silently pick a different Python.) "
            'Alternatively, repair this interpreter with `pip install -U "numba>=0.59" '
            '"llvmlite>=0.42"`.'
        )
    if reason == "missing":
        return (
            "This interpreter doesn't have **mlx-vlm** installed, which is required "
            "to run the model. Launch the app with the interpreter that has it, e.g.:\n\n"
            "```bash\n"
            "~/miniforge3/envs/vlm/bin/python -m streamlit run streamlit_app.py\n"
            "```"
        )
    return (
        "The model backend couldn't be loaded in this environment. Launch the app "
        "from the environment used for training/evaluation (which has a working "
        "mlx-vlm), then reload."
    )


def _empty_state() -> None:
    st.info(
        "Upload one or more leaf images above to get a grounded diagnosis. Originals "
        "are filed under `uploads/<crop>/<predicted_class>/` and every result is "
        "written to `predictions/` and `logs/predictions.jsonl`.",
        icon="🌱",
    )
