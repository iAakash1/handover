"""Stage 2 (analyze): compute every metric and write every report file.

Never runs inference; never touches mlx-vlm. Reads ONLY the frozen
`predictions.jsonl` + `metadata.json` artifact contract produced by stage 1.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from plantdx.core.exceptions import PlantDxError
from plantdx.evaluation.classification import (
    build_lexicon,
    compute_classification_metrics,
    extract_disease_id,
)
from plantdx.evaluation.clinical import build_clinical_lexicons, score_clinical_correctness
from plantdx.evaluation.config import resolve_crop
from plantdx.evaluation.hallucination import build_hallucination_lexicons, score_hallucinations
from plantdx.evaluation.latency import compute_latency_stats
from plantdx.evaluation.per_disease import compute_per_disease_table
from plantdx.evaluation.quality import score_response_quality, summarize_response_quality
from plantdx.evaluation.reproducibility import build_reproducibility_manifest
from plantdx.evaluation.samples import (
    build_sample_comparison,
    render_markdown_table,
    select_sample_indices,
)
from plantdx.evaluation.stats import compare_paired
from plantdx.evaluation.text_metrics import compute_text_metrics
from plantdx.evaluation.visualize import plot_confusion_matrix, plot_grouped_comparison
from plantdx.utils.io import ensure_dir, read_jsonl, write_json

# Metrics grouped by comparable scale, so no chart mixes a 0..1 score with an
# unbounded one on the same axis (CIDEr and BERTScore each get their own chart).
_BOUNDED_TEXT_METRICS = ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "meteor")
_MODELS = ("base", "finetuned")


def run_analysis(
    predictions_path: str | Path,
    *,
    output_dir: str | Path,
    model_path: str,
    adapter_path: str,
    dataset_dir: str,
    seed: int,
) -> dict[str, str]:
    """Run the full stage-2 analysis; return a dict of every written file path."""
    pred_path = Path(predictions_path)
    if not pred_path.is_file():
        raise PlantDxError(
            f"predictions file not found: {pred_path}. Run `plantdx evaluate "
            f"--stage inference` first (a different environment; see docs/EVALUATION.md)."
        )
    rows = list(read_jsonl(pred_path))
    if not rows:
        raise PlantDxError(f"predictions file is empty: {pred_path}")

    out = ensure_dir(output_dir)
    written: dict[str, str] = {}

    crop = _resolve_analysis_crop(pred_path.parent, dataset_dir)
    lexicon = build_lexicon(crop)
    hallucination_lex = build_hallucination_lexicons(crop)
    clinical_lex = build_clinical_lexicons(crop)

    per_model = {
        model: _analyze_one_model(rows, model, lexicon, hallucination_lex, clinical_lex)
        for model in _MODELS
    }

    written.update(_write_predictions_csv(rows, per_model, out))
    written.update(_write_classification_outputs(rows, per_model, lexicon, out))
    written.update(_write_per_disease_csv(rows, per_model, lexicon, out))
    written.update(_write_hallucinations_csv(rows, per_model, out))
    written.update(_write_metric_csvs(per_model, out))
    written.update(_write_latency_csv(rows, per_model, out))
    comparisons = _write_statistical_comparisons(per_model, out, seed=seed)
    written["statistical_comparisons"] = comparisons
    written.update(_write_figures(rows, per_model, lexicon, out))
    written.update(_write_sample_comparisons(rows, per_model, out, seed=seed))

    manifest = build_reproducibility_manifest(
        model_path=model_path,
        adapter_path=adapter_path,
        dataset_dir=dataset_dir,
        split="test",
        seed=seed,
    )
    system_info_path = out / "system_info.json"
    write_json(system_info_path, manifest.to_dict())
    written["system_info"] = str(system_info_path)

    written.update(_write_summary(rows, per_model, manifest.to_dict(), comparisons, out, crop))
    return written


def _resolve_analysis_crop(predictions_dir: Path, dataset_dir: str) -> str:
    """The crop this analysis run is for.

    Prefers the frozen ``metadata.json`` stage 1 writes next to
    ``predictions.jsonl`` (the artifact contract this module's own docstring
    promises to read -- always co-located, present even when ``dataset_dir``
    itself is not, e.g. ``analyze`` running on a different machine than the
    dataset lives on). Falls back to the dataset's own manifest for
    predictions produced before this field existed.
    """
    metadata_path = predictions_dir / "metadata.json"
    if metadata_path.is_file():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        crop = data.get("crop")
        if crop:
            return str(crop)
    return resolve_crop(dataset_dir)


# --------------------------------------------------------------------------- #
# Per-model analysis
# --------------------------------------------------------------------------- #


def _analyze_one_model(
    rows: list[dict[str, Any]],
    model: str,
    lexicon: Any,
    hallucination_lex: Any,
    clinical_lex: Any,
) -> dict[str, Any]:
    predictions_text = [str(r[f"{model}_prediction"]) for r in rows]
    ground_truths = [str(r["ground_truth"]) for r in rows]
    targets = [str(r["disease_id"]) for r in rows]
    confidences = [r.get(f"{model}_confidence") for r in rows]

    predicted_disease = [extract_disease_id(t, lexicon) for t in predictions_text]
    classification = compute_classification_metrics(predicted_disease, targets, lexicon.disease_ids)

    pairs = [
        (str(r["image_id"]), predictions_text[i], ground_truths[i]) for i, r in enumerate(rows)
    ]
    text_per_sample, text_corpus = compute_text_metrics(pairs)

    hallucinations = [
        score_hallucinations(predictions_text[i], targets[i], hallucination_lex)
        for i in range(len(rows))
    ]
    clinical = [
        score_clinical_correctness(predictions_text[i], targets[i], clinical_lex, hallucination_lex)
        for i in range(len(rows))
    ]
    quality = [score_response_quality(t) for t in predictions_text]

    per_disease = compute_per_disease_table(
        lexicon.disease_ids,
        predicted_disease,
        targets,
        predictions_text,
        confidences,
        [h.any for h in hallucinations],
    )

    latency = compute_latency_stats(
        [float(r[f"{model}_runtime_ms"]) for r in rows],
        [int(r[f"{model}_generation_tokens"]) for r in rows],
        [float(r[f"{model}_peak_memory_gb"]) for r in rows],
    )

    return {
        "predicted_disease": predicted_disease,
        "classification": classification,
        "text_per_sample": text_per_sample,
        "text_corpus": text_corpus,
        "hallucinations": hallucinations,
        "clinical": clinical,
        "quality": quality,
        "quality_summary": summarize_response_quality(quality),
        "per_disease": per_disease,
        "latency": latency,
    }


# --------------------------------------------------------------------------- #
# Writers -- one function per output file (or small family of files)
# --------------------------------------------------------------------------- #


def _write_predictions_csv(
    rows: list[dict[str, Any]], per_model: dict[str, Any], out: Path
) -> dict[str, str]:
    path = out / "predictions.csv"
    fieldnames = [
        "image_id",
        "disease_id",
        "class_name",
        "instruction",
        "ground_truth",
        "base_prediction",
        "base_predicted_disease",
        "finetuned_prediction",
        "finetuned_predicted_disease",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "disease_id": row["disease_id"],
                    "class_name": row["class_name"],
                    "instruction": row["instruction"],
                    "ground_truth": row["ground_truth"],
                    "base_prediction": row["base_prediction"],
                    "base_predicted_disease": per_model["base"]["predicted_disease"][i],
                    "finetuned_prediction": row["finetuned_prediction"],
                    "finetuned_predicted_disease": per_model["finetuned"]["predicted_disease"][i],
                }
            )
    return {"predictions_csv": str(path)}


def _write_classification_outputs(
    rows: list[dict[str, Any]], per_model: dict[str, Any], lexicon: Any, out: Path
) -> dict[str, str]:
    from sklearn.metrics import confusion_matrix

    written: dict[str, str] = {}
    report_path = out / "classification_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", *_fields(per_model["base"]["classification"])])
        for model in _MODELS:
            metrics = per_model[model]["classification"]
            writer.writerow([model, *asdict(metrics).values()])
    written["classification_report_csv"] = str(report_path)

    targets = [str(r["disease_id"]) for r in rows]
    for model in _MODELS:
        predicted = per_model[model]["predicted_disease"]
        matrix = confusion_matrix(targets, predicted, labels=list(lexicon.disease_ids))
        csv_path = out / f"confusion_matrix_{model}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["true\\pred", *lexicon.disease_ids])
            for label, row_values in zip(lexicon.disease_ids, matrix, strict=True):
                writer.writerow([label, *row_values.tolist()])
        written[f"confusion_matrix_{model}_csv"] = str(csv_path)
        png, svg = plot_confusion_matrix(
            matrix.tolist(),
            list(lexicon.disease_ids),
            normalized=False,
            title=f"Confusion Matrix ({model})",
            out_dir=out / "figures",
            filename=f"confusion_matrix_{model}",
        )
        written[f"confusion_matrix_{model}_png"] = str(png)
        written[f"confusion_matrix_{model}_svg"] = str(svg)
    return written


def _fields(dataclass_instance: Any) -> list[str]:
    return list(asdict(dataclass_instance).keys())


def _write_per_disease_csv(
    rows: list[dict[str, Any]], per_model: dict[str, Any], lexicon: Any, out: Path
) -> dict[str, str]:
    path = out / "per_disease.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "disease_id", *_fields(per_model["base"]["per_disease"][0])[1:]])
        for model in _MODELS:
            for pd_row in per_model[model]["per_disease"]:
                values = list(asdict(pd_row).values())
                writer.writerow([model, *values])
    return {"per_disease_csv": str(path)}


def _write_hallucinations_csv(
    rows: list[dict[str, Any]], per_model: dict[str, Any], out: Path
) -> dict[str, str]:
    path = out / "hallucinations.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["image_id", "model", "disease_id", *_fields(per_model["base"]["hallucinations"][0])]
        )
        for i, row in enumerate(rows):
            for model in _MODELS:
                flags = per_model[model]["hallucinations"][i]
                writer.writerow(
                    [row["image_id"], model, row["disease_id"], *asdict(flags).values()]
                )
    return {"hallucinations_csv": str(path)}


def _write_metric_csvs(per_model: dict[str, Any], out: Path) -> dict[str, str]:
    written: dict[str, str] = {}
    specs = (
        ("bleu_scores.csv", ("id", "bleu1", "bleu2", "bleu3", "bleu4")),
        ("rouge_scores.csv", ("id", "rouge_l")),
        ("meteor_scores.csv", ("id", "meteor")),
        ("cider_scores.csv", ("id", "cider")),
        ("bertscore.csv", ("id", "bertscore_f1")),
    )
    for filename, fields in specs:
        path = out / filename
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["model", *fields])
            for model in _MODELS:
                for sample in per_model[model]["text_per_sample"]:
                    writer.writerow([model, *(getattr(sample, f) for f in fields)])
        written[filename.removesuffix(".csv")] = str(path)
    return written


def _write_latency_csv(
    rows: list[dict[str, Any]], per_model: dict[str, Any], out: Path
) -> dict[str, str]:
    path = out / "latency.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", *_fields(per_model["base"]["latency"])])
        for model in _MODELS:
            writer.writerow([model, *asdict(per_model[model]["latency"]).values()])
    return {"latency_csv": str(path)}


def _write_statistical_comparisons(per_model: dict[str, Any], out: Path, *, seed: int) -> str:
    comparisons = []
    for metric in (*_BOUNDED_TEXT_METRICS, "cider", "bertscore_f1"):
        base_values = [getattr(s, metric) for s in per_model["base"]["text_per_sample"]]
        ft_values = [getattr(s, metric) for s in per_model["finetuned"]["text_per_sample"]]
        comparisons.append(asdict(compare_paired(metric, base_values, ft_values, seed=seed)))
    path = out / "statistical_comparisons.json"
    write_json(path, comparisons)
    return str(path)


def _write_figures(
    rows: list[dict[str, Any]], per_model: dict[str, Any], lexicon: Any, out: Path
) -> dict[str, str]:
    figures_dir = out / "figures"
    written: dict[str, str] = {}

    base_c, ft_c = per_model["base"]["classification"], per_model["finetuned"]["classification"]
    png, svg = plot_grouped_comparison(
        ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
        [base_c.accuracy, base_c.balanced_accuracy, base_c.f1_macro, base_c.f1_weighted],
        [ft_c.accuracy, ft_c.balanced_accuracy, ft_c.f1_macro, ft_c.f1_weighted],
        title="Classification Accuracy Comparison",
        ylabel="Score",
        out_dir=figures_dir,
        filename="accuracy_comparison",
    )
    written["accuracy_comparison_png"], written["accuracy_comparison_svg"] = str(png), str(svg)

    base_tc, ft_tc = per_model["base"]["text_corpus"], per_model["finetuned"]["text_corpus"]
    for group_name, metrics in (
        ("bounded_metric_comparison", _BOUNDED_TEXT_METRICS),
        ("cider_comparison", ("cider",)),
        ("bertscore_comparison", ("bertscore_f1",)),
    ):
        png, svg = plot_grouped_comparison(
            list(metrics),
            [getattr(base_tc, m) for m in metrics],
            [getattr(ft_tc, m) for m in metrics],
            title=f"{group_name.replace('_', ' ').title()}: Base vs Fine-tuned",
            ylabel="Score",
            out_dir=figures_dir,
            filename=group_name,
        )
        written[f"{group_name}_png"], written[f"{group_name}_svg"] = str(png), str(svg)

    disease_labels = list(lexicon.disease_ids)
    base_f1 = {r.disease_id: r.f1 for r in per_model["base"]["per_disease"]}
    ft_f1 = {r.disease_id: r.f1 for r in per_model["finetuned"]["per_disease"]}
    png, svg = plot_grouped_comparison(
        disease_labels,
        [base_f1[d] for d in disease_labels],
        [ft_f1[d] for d in disease_labels],
        title="Per-Disease F1: Base vs Fine-tuned",
        ylabel="F1",
        out_dir=figures_dir,
        filename="per_disease_f1",
    )
    written["per_disease_f1_png"], written["per_disease_f1_svg"] = str(png), str(svg)

    base_hr = {r.disease_id: r.hallucination_rate for r in per_model["base"]["per_disease"]}
    ft_hr = {r.disease_id: r.hallucination_rate for r in per_model["finetuned"]["per_disease"]}
    png, svg = plot_grouped_comparison(
        disease_labels,
        [base_hr[d] for d in disease_labels],
        [ft_hr[d] for d in disease_labels],
        title="Hallucination Rate by Disease: Base vs Fine-tuned",
        ylabel="Hallucination rate",
        out_dir=figures_dir,
        filename="hallucination_comparison",
    )
    written["hallucination_comparison_png"], written["hallucination_comparison_svg"] = (
        str(png),
        str(svg),
    )

    base_q, ft_q = per_model["base"]["quality_summary"], per_model["finetuned"]["quality_summary"]
    png, svg = plot_grouped_comparison(
        ["avg_word_count", "avg_sentence_count"],
        [base_q["avg_word_count"], base_q["avg_sentence_count"]],
        [ft_q["avg_word_count"], ft_q["avg_sentence_count"]],
        title="Response Length: Base vs Fine-tuned",
        ylabel="Count",
        out_dir=figures_dir,
        filename="response_length",
    )
    written["response_length_png"], written["response_length_svg"] = str(png), str(svg)

    base_l, ft_l = per_model["base"]["latency"], per_model["finetuned"]["latency"]
    png, svg = plot_grouped_comparison(
        ["mean_ms", "median_ms", "p95_ms"],
        [base_l.mean_ms, base_l.median_ms, base_l.p95_ms],
        [ft_l.mean_ms, ft_l.median_ms, ft_l.p95_ms],
        title="Inference Latency: Base vs Fine-tuned",
        ylabel="Milliseconds",
        out_dir=figures_dir,
        filename="latency_comparison",
    )
    written["latency_comparison_png"], written["latency_comparison_svg"] = str(png), str(svg)

    return written


def _write_sample_comparisons(
    rows: list[dict[str, Any]], per_model: dict[str, Any], out: Path, *, seed: int
) -> dict[str, str]:
    indices = select_sample_indices(len(rows), seed=seed, count=50)
    base_bleu1 = [s.bleu1 for s in per_model["base"]["text_per_sample"]]
    ft_bleu1 = [s.bleu1 for s in per_model["finetuned"]["text_per_sample"]]
    comparisons = [
        build_sample_comparison(
            image_path=rows[i]["image_path"],
            ground_truth_disease=rows[i]["disease_id"],
            instruction=rows[i]["instruction"],
            ground_truth_answer=rows[i]["ground_truth"],
            base_answer=rows[i]["base_prediction"],
            finetuned_answer=rows[i]["finetuned_prediction"],
            base_bleu1=base_bleu1[i],
            finetuned_bleu1=ft_bleu1[i],
        )
        for i in indices
    ]
    path = out / "sample_comparisons.md"
    path.write_text(
        "# Sample Comparisons (base vs fine-tuned)\n\n" + render_markdown_table(comparisons),
        encoding="utf-8",
    )
    return {"sample_comparisons_md": str(path)}


def _write_summary(
    rows: list[dict[str, Any]],
    per_model: dict[str, Any],
    manifest: dict[str, Any],
    comparisons_path: str,
    out: Path,
    crop: str,
) -> dict[str, str]:
    metrics_json = {
        model: {
            "classification": asdict(per_model[model]["classification"]),
            "text_corpus": asdict(per_model[model]["text_corpus"]),
            "quality_summary": per_model[model]["quality_summary"],
            "latency": asdict(per_model[model]["latency"]),
        }
        for model in _MODELS
    }
    metrics_path = out / "metrics.json"
    write_json(metrics_path, metrics_json)

    comparisons = json.loads(Path(comparisons_path).read_text(encoding="utf-8"))
    evaluation_json = {
        "sample_count": len(rows),
        "metrics": metrics_json,
        "statistical_comparisons": comparisons,
        "reproducibility": manifest,
    }
    evaluation_path = out / "evaluation.json"
    write_json(evaluation_path, evaluation_json)

    summary_path = out / "evaluation_summary.md"
    summary_path.write_text(
        _render_summary_markdown(rows, per_model, comparisons, crop), encoding="utf-8"
    )

    return {
        "metrics_json": str(metrics_path),
        "evaluation_json": str(evaluation_path),
        "evaluation_summary_md": str(summary_path),
    }


def _render_summary_markdown(
    rows: list[dict[str, Any]],
    per_model: dict[str, Any],
    comparisons: list[dict[str, Any]],
    crop: str,
) -> str:
    base_c, ft_c = per_model["base"]["classification"], per_model["finetuned"]["classification"]
    lines = [
        f"# Evaluation Summary: Base vs Fine-tuned (Qwen2.5-VL, {crop})",
        "",
        f"- Samples evaluated: **{len(rows)}**",
        f"- Base accuracy: **{base_c.accuracy:.3f}** | "
        f"Fine-tuned accuracy: **{ft_c.accuracy:.3f}**",
        f"- Base macro-F1: **{base_c.f1_macro:.3f}** | "
        f"Fine-tuned macro-F1: **{ft_c.f1_macro:.3f}**",
        "",
        "## Statistical significance (paired, base vs fine-tuned)",
        "| Metric | Base mean | Fine-tuned mean | Difference | t-test p | Significant (p<0.05) |",
        "|---|---|---|---|---|---|",
    ]
    for cmp in comparisons:
        lines.append(
            f"| {cmp['metric_name']} | {cmp['base']['mean']:.3f} | "
            f"{cmp['finetuned']['mean']:.3f} | {cmp['mean_difference']:+.3f} | "
            f"{cmp['t_p_value']:.4g} | {'yes' if cmp['significant_at_0_05'] else 'no'} |"
        )
    lines += [
        "",
        "See `metrics.json`, `per_disease.csv`, `hallucinations.csv`, and `figures/` for detail.",
    ]
    return "\n".join(lines) + "\n"
