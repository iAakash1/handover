"""Evaluation and adapter panels for company (TIH) runs.

Reads the frozen artifacts written by ``plantdx company evaluate`` and never
recomputes them. Missing results are reported as missing, not as an error.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import training_service as svc
from app.company_runtime import active_entry, adapter_summary, backend_status


def company_evaluation_panel() -> None:
    """Base vs fine-tuned metrics, per-class table, and the confusion matrix."""
    runs = [run["run_name"] for run in svc.list_runs()]
    if not runs:
        st.caption("No company training runs yet. Train one from the Training tab.")
        return

    cols = st.columns([2, 1])
    run_name = cols[0].selectbox("Run", runs, key="tih_eval_run")
    split = cols[1].selectbox("Split", ["validation", "test"], key="tih_eval_split")

    tuned = svc.read_metrics(run_name, split, tuned=True)
    base = svc.read_metrics(run_name, split, tuned=False)

    if tuned is None and base is None:
        st.info(
            f"No evaluation for `{run_name}` on the {split} split yet. Run:\n\n"
            f"```bash\npython -m plantdx company evaluate --split {split}\n```",
            icon="📭",
        )
        return

    _headline(base, tuned, split)
    if split == "validation":
        _comparison(base, tuned)
    result = tuned or base
    if result is not None:
        _per_class(result)
        _confusion(result)


def _headline(base: Any, tuned: Any, split: str) -> None:
    result = tuned or base
    st.caption(
        f"Held-out **{split}** split — {result.samples} company tomato images, "
        f"{result.seconds_per_sample:.2f} s per image."
    )
    cols = st.columns(4)
    if tuned is not None:
        delta_acc = f"{(tuned.accuracy - base.accuracy) * 100:+.1f} pts vs base" if base else None
        delta_f1 = f"{tuned.f1_macro - base.f1_macro:+.3f} vs base" if base else None
        cols[0].metric("Accuracy", f"{tuned.accuracy * 100:.1f}%", delta=delta_acc)
        cols[1].metric("Macro F1", f"{tuned.f1_macro:.3f}", delta=delta_f1)
        cols[2].metric("Weighted F1", f"{tuned.f1_weighted:.3f}")
        cols[3].metric("Unparseable", tuned.unparseable)
    else:
        cols[0].metric("Accuracy (base)", f"{base.accuracy * 100:.1f}%")
        cols[1].metric("Macro F1 (base)", f"{base.f1_macro:.3f}")
        cols[2].metric("Weighted F1 (base)", f"{base.f1_weighted:.3f}")
        cols[3].metric("Unparseable", base.unparseable)


def _comparison(base: Any, tuned: Any) -> None:
    from plantdx.company.evaluate import compare

    if base is None or tuned is None:
        st.caption(
            "Run the other side of the comparison with "
            "`python -m plantdx company evaluate --split <split> --base` to see the delta."
        )
        return
    st.markdown("**Base Qwen2.5-VL vs fine-tuned**")
    rows = compare(base, tuned)
    st.dataframe(
        {
            "Metric": [r["metric"] for r in rows],
            "Base": [f"{r['base']:.4f}" if r["base"] is not None else "—" for r in rows],
            "Fine-tuned": [
                f"{r['fine_tuned']:.4f}" if r["fine_tuned"] is not None else "—" for r in rows
            ],
            "Delta": [f"{r['delta']:+.4f}" if r["delta"] is not None else "—" for r in rows],
        },
        hide_index=True,
        width="stretch",
    )


def _per_class(result: Any) -> None:
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
    best, worst = result.best_class, result.worst_class
    if best and worst:
        cols = st.columns(2)
        cols[0].success(f"Strongest: **{best.label}** (F1 {best.f1:.3f})", icon="🟢")
        cols[1].warning(f"Weakest: **{worst.label}** (F1 {worst.f1:.3f})", icon="🟠")
    if result.confusions:
        st.markdown("**Most confused pairs**")
        st.dataframe(
            {
                "True": [c["true"] for c in result.confusions],
                "Predicted": [c["predicted"] for c in result.confusions],
                "Count": [c["count"] for c in result.confusions],
            },
            hide_index=True,
            width="stretch",
        )


def _confusion(result: Any) -> None:
    if not result.confusion_matrix:
        return
    st.markdown("**Confusion matrix** — rows are true classes, columns predicted")
    columns = [*result.labels, "unparseable"]
    table: dict[str, Any] = {"true \\ pred": result.labels}
    for index, name in enumerate(columns):
        table[name] = [row[index] if index < len(row) else 0 for row in result.confusion_matrix]
    st.dataframe(table, hide_index=True, width="stretch")


def company_adapter_panel() -> None:
    """What model the app is serving right now, and its provenance."""
    ok, reason = backend_status()
    if ok:
        st.success("CUDA backend available (transformers + PEFT + bitsandbytes)", icon="✅")
    else:
        st.error(f"CUDA backend unavailable: {reason}", icon="⛔")

    entry = active_entry()
    if entry is None:
        st.warning(
            "No active adapter. Train a run, then press **Use this model** on the "
            "Training tab to serve it here.",
            icon="⚠️",
        )
        return

    path = entry.resolved_path()
    info = adapter_summary(path)
    if info["attached"]:
        st.success("LoRA adapter attached", icon="✅")
    elif path.is_dir():
        st.error("Adapter directory exists but holds no LoRA weights", icon="⛔")
    else:
        st.error(f"Registered adapter is missing: {path}", icon="⛔")

    params = info.get("trainable_params")
    rows = {
        "Base model": entry.base_model,
        "Fine-tuning": "QLoRA (4-bit NF4)",
        "Backend": entry.backend,
        "Dataset": entry.dataset,
        "Adapter run": entry.run_name,
        "Adapter path": str(path),
        "Status": "Valid" if info["attached"] else "Missing or corrupt",
        "LoRA rank": info.get("rank", "—"),
        "LoRA scale (alpha/r)": info.get("scale", "—"),
        "Adapted modules": info.get("num_target_modules", "—"),
        "Trainable params": f"{params:,}" if isinstance(params, int) else "—",
        "Weights checksum": info.get("weights_checksum", "—"),
        "Classes": f"{len(entry.classes)} — " + ", ".join(entry.classes[:4]) + " …",
        "Activated": entry.activated_at or "—",
        "Validation macro F1": _fmt(entry.metrics.get("f1_macro")),
        "Validation accuracy": _fmt(entry.metrics.get("accuracy")),
    }
    test = svc.read_metrics(entry.run_name, "test")
    rows["Test macro F1"] = _fmt(test.f1_macro) if test else "not evaluated"
    st.table({"Field": list(rows.keys()), "Value": [str(v) for v in rows.values()]})


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, int | float) else "—"
