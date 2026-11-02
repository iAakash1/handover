"""
scripts/create_plots.py

Reads json_results/model_summary.json + evaluation_master.json and
renders publication-quality PNG plots into plots/:

  avg_score_by_model.png             — average total judge score per model
  accuracy_by_model.png              — disease-label accuracy per model
  hallucination_rate_by_model.png    — hallucination rate per model
  latency_by_model.png               — average latency per model
  peak_memory_by_model.png           — average peak memory per model
  radar_comparison.png               — normalized multi-metric radar chart
  confusion_matrix_<model>.png       — one confusion matrix per model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def bar_chart(models, values, ylabel, title, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(models), 1)))
    plot_values = [v if v is not None else 0 for v in values]
    bars = ax.bar(models, plot_values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    for b, v in zip(bars, values):
        label = f"{v:.2f}" if v is not None else "N/A"
        ax.annotate(label, (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / filename)
    plt.close(fig)


def canonicalize_predicted(predicted, known_labels):
    """Map free-text predictions onto the known ground-truth label set so
    the confusion matrix stays readable instead of having one row per
    slightly-different phrasing the model used."""
    if not predicted or not predicted.strip():
        return "(no prediction)"
    for kl in known_labels:
        if utils.labels_match(predicted, kl):
            return kl
    return "Other / Unmatched"


def confusion_matrix_plot(model_key, rows, known_labels):
    pairs = [(r["ground_truth_disease"], r["predicted_disease"])
             for r in rows if r["model"] == model_key]
    if not pairs:
        return

    gts = [p[0] for p in pairs]
    preds = [canonicalize_predicted(p[1], known_labels) for p in pairs]
    labels = sorted(set(gts) | set(preds))
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)))
    for gt, pred in zip(gts, preds):
        mat[idx[gt], idx[pred]] += 1

    size = max(6, len(labels) * 0.75)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("Ground-truth disease")
    ax.set_title(f"Confusion matrix — {model_key}")
    vmax = mat.max() if mat.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                         color="white" if vmax and mat[i, j] > vmax / 2 else "black",
                         fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / f"confusion_matrix_{model_key}.png")
    plt.close(fig)


def radar_chart(models, metrics_dict, filename):
    metric_names = list(next(iter(metrics_dict.values())).keys())
    n = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
    for color, model_key in zip(colors, models):
        values = [metrics_dict[model_key][m] or 0 for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=model_key)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison — normalized scores")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12))
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / filename)
    plt.close(fig)


def main():
    config.ensure_directories()
    summary_path = config.JSON_RESULTS_DIR / "model_summary.json"
    master_path  = config.JSON_RESULTS_DIR / "evaluation_master.json"
    if not summary_path.exists() or not master_path.exists():
        utils.log("Missing model_summary.json / evaluation_master.json. Run evaluate.py first.")
        return

    summary = utils.load_json(summary_path)
    master  = utils.load_json(master_path)
    models  = list(summary.keys())
    if not models:
        utils.log("No models found in model_summary.json.")
        return

    bar_chart(models, [summary[m]["avg_total_judge_score"] for m in models],
              "Average judge score (0-100)", "Average Total Score by Model",
              "avg_score_by_model.png", ylim=(0, 100))

    bar_chart(models, [(summary[m]["accuracy"] or 0) * 100 if summary[m]["accuracy"] is not None else None
                        for m in models],
              "Accuracy (%)", "Disease Prediction Accuracy by Model",
              "accuracy_by_model.png", ylim=(0, 100))

    bar_chart(models, [(summary[m]["hallucination_rate"] or 0) * 100 if summary[m]["hallucination_rate"] is not None else None
                        for m in models],
              "Hallucination rate (%)", "Hallucination Rate by Model (lower = better)",
              "hallucination_rate_by_model.png", ylim=(0, 100))

    bar_chart(models, [summary[m]["avg_latency_sec"] for m in models],
              "Seconds / image", "Average Inference Latency by Model",
              "latency_by_model.png")

    bar_chart(models, [summary[m]["avg_peak_memory_gb"] for m in models],
              "GB", "Average Peak Memory Usage by Model",
              "peak_memory_by_model.png")

    known_labels = sorted({r["ground_truth_disease"] for r in master
                            if r["ground_truth_disease"] != "Unknown"})
    for m in models:
        confusion_matrix_plot(m, master, known_labels)

    metrics_dict = {}
    for m in models:
        s = summary[m]
        metrics_dict[m] = {
            "Accuracy":        (s["accuracy"] or 0),
            "Symptom Quality": (s["avg_symptom_quality_score"] or 0) / 20,
            "Disease Quality": (s["avg_disease_quality_score"] or 0) / 40,
            "Groundedness":    (s["avg_hallucination_score"] or 0) / 20,
            "Calibration":     (s["avg_confidence_calibration_score"] or 0) / 10,
        }
    radar_chart(models, metrics_dict, "radar_comparison.png")

    utils.log(f"Saved plots to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
