"""The Training page.

Orchestration only: every decision, check, and subprocess lives in
``app.training_service``.

Nothing on this page starts a full training run on its own. Rendering the page,
running preflight, preparing the dataset, and running the smoke test all stop by
themselves. The single call to ``training_service.start_training`` sits inside
the RUN TRAINING button's click branch and nowhere else.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

from app import training_service as svc
from app.logging_setup import get_logger

_PROCESS = "tih_train_process"
_STATE = "tih_train_state"
_PREFLIGHT = "tih_preflight"
_LAUNCHING = "tih_launching"

_log = get_logger()


def render_training(settings: dict[str, Any]) -> None:
    """Render the whole Training page."""
    del settings
    st.markdown("### Company tomato dataset — Qwen2.5-VL QLoRA fine-tune")

    try:
        cfg = svc.load_config(_overrides())
        dataset = svc.load_dataset(cfg)
    except Exception as exc:
        st.error(f"Could not load the training configuration: {exc}", icon="⚠️")
        return

    state = st.session_state.get(_STATE) or svc.read_state()
    process = st.session_state.get(_PROCESS)
    current = svc.status(state, process)

    _configuration_panel(cfg, dataset)

    if current == svc.RUNNING:
        _running_panel(cfg, state, process)
        return

    svc.release_lock_if_finished(state)
    _outcome_banner(current, state)

    report = _preflight_panel(cfg)
    _advanced_settings(cfg)
    _smoke_panel(cfg)
    _run_button(cfg, report)
    _results_panel(cfg, state, current)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _configuration_panel(cfg: Any, dataset: Any) -> None:
    from plantdx.company.environment import gpu_telemetry

    gpu = gpu_telemetry()
    from plantdx.company.environment import environment_facts

    cuda_gpu = environment_facts().gpu
    counts = _split_counts(cfg)

    with st.container(border=True):
        st.markdown("**Training configuration**")
        cols = st.columns(4)
        cols[0].metric("Model", "Qwen2.5-VL-7B-Instruct")
        cols[1].metric("Fine-tuning", "QLoRA 4-bit")
        cols[2].metric("Dataset", f"TIH / {dataset.name}")
        cols[3].metric("Classes", len(dataset.classes))

        cols = st.columns(4)
        cols[0].metric("Train images", counts.get("train", "—"))
        cols[1].metric("Validation images", counts.get("validation", "—"))
        cols[2].metric("Test images", counts.get("test", "—"))
        cols[3].metric(
            "GPU",
            cuda_gpu.name or "not detected",
            f"{cuda_gpu.total_vram_gb} GiB VRAM" if cuda_gpu.available else "",
            delta_color="off",
        )

        cols = st.columns(5)
        cols[0].metric("Epochs", f"{cfg.optim.epochs:g}")
        cols[1].metric("Learning rate", f"{cfg.optim.learning_rate:.0e}")
        cols[2].metric("LoRA", f"r{cfg.lora.rank} / alpha{cfg.lora.alpha}")
        cols[3].metric("Gradient accumulation", cfg.optim.gradient_accumulation_steps)
        cols[4].metric("Effective batch", cfg.optim.effective_batch_size)

        smoke = svc.smoke_status(cfg)
        cols = st.columns(2)
        cols[0].metric(
            "Estimated training + evaluation",
            f"{smoke.estimated_total_seconds / 3600:.2f} h"
            if smoke and smoke.estimated_total_seconds
            else "not measured",
        )
        cols[1].metric(
            "Smoke peak VRAM",
            f"{smoke.peak_memory_gb:.2f} GiB" if smoke and smoke.peak_memory_gb else "not measured",
        )
        st.caption(
            "Epoch validation · early stopping patience 1 · best validation macro F1 checkpoint · one final test"
        )
        st.caption(
            f"Run `{cfg.run_name}` · vision tower frozen · image max side "
            f"{cfg.model.image_max_side}px · seed {cfg.seed} · "
            f"raw dataset `{dataset.root}` (read-only)"
        )


def _split_counts(cfg: Any) -> dict[str, int]:
    from plantdx.company.dataset import load_manifest

    try:
        manifest = load_manifest(cfg.dataset_dir)
    except Exception:
        return {}
    counts = manifest.get("split", {}).get("counts", {})
    return {k: int(v) for k, v in counts.items()} if isinstance(counts, dict) else {}


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def _preflight_panel(cfg: Any) -> Any:
    from plantdx.company.preflight import FAIL, OK

    head = st.columns([3, 1, 1])
    head[0].markdown("##### Pre-flight status")
    deep = head[1].toggle(
        "Deep check",
        value=False,
        help="Also load the processor and encode one real sample (slower)",
    )
    if head[2].button("Re-run checks", width="stretch"):
        st.session_state.pop(_PREFLIGHT, None)

    cache_key = (deep, tuple(sorted(_overrides().items())))
    cached = st.session_state.get(_PREFLIGHT)
    if cached is None or cached[0] != cache_key:
        with st.spinner("Checking the environment, dataset, model, and smoke test..."):
            report = svc.run_preflight(_overrides(), deep=deep)
        st.session_state[_PREFLIGHT] = (cache_key, report)
    else:
        report = cached[1]

    st.dataframe(
        [
            {
                "": {OK: "✅", FAIL: "❌"}.get(check.status, "⚠️"),
                "Check": check.name,
                "Detail": check.detail,
            }
            for check in report.checks
        ],
        hide_index=True,
        width="stretch",
    )

    if report.ready:
        st.success("**STATUS: 🟢 READY TO TRAIN** — every critical check passed.")
    else:
        blocking = ", ".join(check.name for check in report.blocking)
        st.error(f"**STATUS: 🔴 NOT READY** — blocked by: {blocking}")
    del cfg
    return report


def _advanced_settings(cfg: Any) -> None:
    with st.expander("Advanced settings", expanded=False):
        st.caption(
            "Defaults come from `configs/train/qwen25vl_tih_tomato_strong.yaml` and are the "
            "validated configuration. Change them only if you know why."
        )
        cols = st.columns(3)
        cols[0].number_input("Epochs", 1.0, 3.0, float(cfg.optim.epochs), 1.0, key="tih_epochs")
        cols[1].number_input(
            "Learning rate",
            1e-6,
            1e-3,
            float(cfg.optim.learning_rate),
            1e-6,
            format="%.6f",
            key="tih_lr",
        )
        cols[2].number_input(
            "Gradient accumulation",
            1,
            64,
            int(cfg.optim.gradient_accumulation_steps),
            1,
            key="tih_accum",
        )
        cols = st.columns(3)
        cols[0].number_input("LoRA rank", 1, 256, int(cfg.lora.rank), 1, key="tih_rank")
        cols[1].number_input("LoRA alpha", 1, 512, int(cfg.lora.alpha), 1, key="tih_alpha")
        cols[2].number_input(
            "Image max side",
            112,
            1120,
            int(cfg.model.image_max_side),
            28,
            key="tih_side",
            help="Longest image side after resize; must be a multiple of 28.",
        )
        cols = st.columns(2)
        cols[0].number_input("Seed", 0, 2**31 - 1, int(cfg.seed), 1, key="tih_seed")
        cols[1].text_input("Run name", cfg.run_name, key="tih_run_name")


def _overrides() -> dict[str, Any]:
    """Read the advanced-settings widgets into config overrides."""
    keys = {
        "tih_epochs": "epochs",
        "tih_lr": "learning_rate",
        "tih_accum": "gradient_accumulation_steps",
        "tih_rank": "lora_rank",
        "tih_alpha": "lora_alpha",
        "tih_side": "image_max_side",
        "tih_seed": "seed",
        "tih_run_name": "run_name",
    }
    out: dict[str, Any] = {}
    for widget, key in keys.items():
        value = st.session_state.get(widget)
        if value not in (None, ""):
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Smoke test (tiny, explicit, never rolls into a full run)
# --------------------------------------------------------------------------- #


def _smoke_panel(cfg: Any) -> None:
    st.markdown("##### QLoRA smoke test")
    previous = svc.smoke_status(cfg)
    cols = st.columns([3, 1])
    if previous is None:
        cols[0].warning(
            "Not run yet. It trains 2 optimizer steps on a handful of images to "
            "prove image load → processor → forward → loss → backward → optimizer → "
            "adapter save on this GPU. Preflight requires it.",
            icon="🧪",
        )
    elif previous.passed:
        detail = previous.detail
        if previous.final_loss is not None:
            detail += f" · final loss {previous.final_loss:.3f}"
        if previous.peak_memory_gb:
            detail += f" · peak {previous.peak_memory_gb} GB"
        cols[0].success(f"Passed — {detail}", icon="🧪")
    else:
        cols[0].error(f"Failed — {previous.detail}", icon="🧪")

    if cols[1].button("Run smoke test", width="stretch", key="tih_smoke_btn"):
        with st.spinner("Training two steps on a handful of images..."):
            try:
                from plantdx.company.smoke import run_smoke_test

                run_smoke_test(cfg, svc.load_dataset(cfg))
            except Exception as exc:
                _log.exception("smoke test crashed")
                st.error(f"Smoke test crashed: {type(exc).__name__}: {exc}", icon="⚠️")
                return
        st.session_state.pop(_PREFLIGHT, None)
        st.rerun()


# --------------------------------------------------------------------------- #
# The one place that starts training
# --------------------------------------------------------------------------- #


def _run_button(cfg: Any, report: Any) -> None:
    st.markdown("##### Run training")
    resume_target = _resume_target(cfg)
    disabled = not report.ready

    cols = st.columns([2, 1, 2])
    if cols[0].button(
        "🚀  RUN TRAINING",
        type="primary",
        width="stretch",
        disabled=disabled,
        key="tih_run_training",
    ):
        _launch(resume=False)

    if resume_target is not None and cols[1].button(
        "⏩ Resume", width="stretch", disabled=disabled, key="tih_resume"
    ):
        _launch(resume=True)

    if resume_target is not None:
        cols[2].caption(f"A checkpoint exists: `{resume_target.name}` — Resume continues from it.")

    if disabled:
        st.caption("Disabled until every critical pre-flight check passes.")
    else:
        launched = svc.build_train_argv(_overrides())
        shown = " ".join(Path(a).name if a.endswith(".exe") else a for a in launched)
        st.caption(f"One click starts one official run: `{shown}`")


def _launch(*, resume: bool) -> None:
    """Start the official run. Reached only from a button click branch."""
    if st.session_state.get(_LAUNCHING):
        return
    st.session_state[_LAUNCHING] = True
    try:
        state, process = svc.start_training(_overrides(), resume=resume)
    except Exception as exc:
        _log.exception("could not start training")
        st.error(f"Could not start training: {exc}", icon="⚠️")
        st.session_state[_LAUNCHING] = False
        return
    st.session_state[_STATE] = state
    st.session_state[_PROCESS] = process
    st.session_state[_LAUNCHING] = False
    st.rerun()


def _resume_target(cfg: Any) -> Path | None:
    from plantdx.company.train import latest_checkpoint

    try:
        return latest_checkpoint(cfg.checkpoint_dir)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Live progress
# --------------------------------------------------------------------------- #


def _running_panel(cfg: Any, state: Any, process: Any) -> None:
    progress = svc.read_progress(cfg)
    stage = svc.last_stage(state.log_path)

    st.markdown("##### Training progress")
    st.info(f"**Status: TRAINING** — {stage}  ·  run id `{state.run_id or state.run_name}`")

    st.caption(f"Overall — step {progress.step} / {progress.total_steps or '?'}")
    st.progress(progress.fraction)
    st.caption(f"Epoch {progress.current_epoch} / {progress.total_epochs:g}")
    st.progress(progress.epoch_fraction)

    cols = st.columns(5)
    cols[0].metric("Step", f"{progress.step} / {progress.total_steps or '?'}")
    cols[1].metric("Training loss", f"{progress.loss:.4f}" if progress.loss is not None else "—")
    cols[2].metric(
        "Learning rate",
        f"{progress.learning_rate:.2e}" if progress.learning_rate is not None else "—",
    )
    cols[3].metric("Elapsed", _hms(state.elapsed))
    eta = progress.eta_seconds
    cols[4].metric("Remaining (est.)", _hms(eta) if eta is not None else "—")

    st.metric("Validation loss", f"{progress.validation_loss:.4f}" if progress.validation_loss is not None else "not evaluated yet")
    _gpu_panel(progress)

    cols = st.columns(2)
    cols[0].metric(
        "Latest checkpoint",
        svc.latest_checkpoint(state.checkpoint_dir or cfg.checkpoint_dir),
    )
    peak = progress.memory.get("max_reserved_gb")
    cols[1].metric("Peak GPU memory", f"{peak:.2f} GiB" if peak is not None else "not measured")

    points = [h for h in progress.history if h.get("loss") is not None]
    if len(points) > 1:
        st.markdown("**Training loss**")
        st.line_chart({"loss": [float(h["loss"]) for h in points]}, height=180)

    st.markdown("**Training log**")
    st.code(svc.read_log(state.log_path, lines=40) or "(waiting for output)", language="text")
    st.caption(f"Full log: `{state.log_path}`")

    cols = st.columns([1, 1, 3])
    if cols[0].button("Refresh now", width="stretch"):
        st.rerun()
    if cols[1].button("🛑 STOP TRAINING", width="stretch", key="tih_stop"):
        svc.stop_training(state)
        st.session_state[_STATE] = state
        st.session_state.pop(_PROCESS, None)
        st.warning(
            "Stop signal sent. Checkpoints already written are kept, and Resume "
            "continues from the newest one.",
            icon="🛑",
        )
        time.sleep(2)
        st.rerun()
    cols[2].caption("This panel refreshes every few seconds while training runs.")

    del process
    time.sleep(5)
    st.rerun()


def _gpu_panel(progress: Any) -> None:
    gpu = svc.gpu_sample()
    cols = st.columns(4)
    if not gpu.available:
        cols[0].metric("GPU", "nvidia-smi unavailable")
        return
    cols[0].metric(
        "GPU utilization",
        f"{gpu.utilization_percent:.0f}%" if gpu.utilization_percent is not None else "—",
    )
    cols[1].metric(
        "VRAM used",
        f"{gpu.memory_used_gb} / {gpu.memory_total_gb} GB"
        if gpu.memory_used_gb is not None
        else "—",
    )
    cols[2].metric(
        "VRAM utilization",
        f"{gpu.memory_percent:.0f}%" if gpu.memory_percent is not None else "—",
    )
    cols[3].metric(
        "GPU temperature",
        f"{gpu.temperature_c:.0f} °C" if gpu.temperature_c is not None else "—",
    )
    del progress


def _outcome_banner(current: str, state: Any) -> None:
    if state is None or current in (svc.IDLE, svc.RUNNING):
        return
    if current == svc.SUCCEEDED:
        st.success(
            f"**✅ TRAINING COMPLETE** — `{state.run_name}` finished in "
            f"{state.elapsed / 60:.1f} min.",
        )
    elif current == svc.STOPPED:
        st.warning(
            f"**🛑 INTERRUPTED** — `{state.run_name}` was stopped after "
            f"{state.elapsed / 60:.1f} min. Checkpoints and logs were kept; press "
            "Resume to continue.",
        )
    else:
        st.error("**⛔ TRAINING FAILED**")
        st.markdown(f"**Last successful stage:** {svc.last_stage(state.log_path)}")
        st.markdown("**Reason**")
        st.code(svc.failure_reason(state.log_path), language="text")
        with st.expander("Full log tail"):
            st.code(svc.read_log(state.log_path, lines=120) or "(empty)", language="text")


# --------------------------------------------------------------------------- #
# Results and activation
# --------------------------------------------------------------------------- #


def _results_panel(cfg: Any, state: Any, current: str) -> None:
    from plantdx.company.modeling import adapter_is_valid

    st.markdown("##### Trained adapters")
    runs = svc.list_runs()
    if not runs:
        st.caption("No checkpoints yet. The adapter appears here once a run completes.")
        return

    for run in runs:
        with st.container(border=True):
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**{run['run_name']}**")
            cols[0].caption(run["path"])
            validation = svc.read_metrics(run["run_name"], "validation")
            test = svc.read_metrics(run["run_name"], "test")
            cols[1].metric("Val macro F1", f"{validation.f1_macro:.3f}" if validation else "—")
            cols[2].metric("Test macro F1", f"{test.f1_macro:.3f}" if test else "—")

            ok, reason = adapter_is_valid(run["path"])
            if not ok:
                cols[3].warning("no adapter", icon="⚠️")
                cols[3].caption(reason)
                continue
            if cols[3].button("Use this model", key=f"use_{run['run_name']}", width="stretch"):
                try:
                    path = svc.activate(run["run_name"], run["path"])
                except Exception as exc:
                    st.error(f"Could not activate: {exc}", icon="⚠️")
                else:
                    st.success(f"Diagnose now serves `{run['run_name']}`. Registry: `{path}`")
                    st.cache_resource.clear()

            if validation is not None:
                _metrics_block(validation, run["run_name"], cfg, state)

    del current


def _metrics_block(result: Any, run_name: str, cfg: Any, state: Any) -> None:
    with st.expander(f"Validation metrics — {run_name}", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{result.accuracy * 100:.2f}%")
        cols[1].metric("Macro precision", f"{result.precision_macro:.4f}")
        cols[2].metric("Macro recall", f"{result.recall_macro:.4f}")
        cols[3].metric("Macro F1", f"{result.f1_macro:.4f}")
        cols[4].metric("Weighted F1", f"{result.f1_weighted:.4f}")

        st.markdown("**Per-class metrics**")
        st.dataframe(
            {
                "Class": [row.label for row in result.per_class],
                "Support": [row.support for row in result.per_class],
                "Precision": [f"{row.precision:.3f}" for row in result.per_class],
                "Recall": [f"{row.recall:.3f}" for row in result.per_class],
                "F1": [f"{row.f1:.3f}" for row in result.per_class],
            },
            hide_index=True,
            width="stretch",
        )

        if result.confusion_matrix:
            st.markdown("**Confusion matrix** — rows true, columns predicted")
            columns = [*result.labels, "unparseable"]
            table: dict[str, Any] = {"true \\ pred": list(result.labels)}
            for index, name in enumerate(columns):
                table[name] = [
                    row[index] if index < len(row) else 0 for row in result.confusion_matrix
                ]
            st.dataframe(table, hide_index=True, width="stretch")

        report_dir = cfg.report_dir
        st.markdown("**Artifacts**")
        st.code(
            "\n".join(
                [
                    f"Adapter        : {cfg.checkpoint_dir}",
                    f"Run config     : {cfg.checkpoint_dir / 'run_config.json'}",
                    f"Training log   : {state.log_path if state else cfg.log_dir / 'train.log'}",
                    f"Loss history   : {cfg.log_dir / 'loss_history.jsonl'}",
                    f"Metrics        : {report_dir / 'validation_finetuned' / 'metrics.json'}",
                    f"Predictions    : {report_dir / 'validation_finetuned' / 'predictions.jsonl'}",
                ]
            ),
            language="text",
        )


def _hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
