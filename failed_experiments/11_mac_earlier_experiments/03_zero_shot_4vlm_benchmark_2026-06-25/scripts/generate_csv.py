"""
scripts/generate_csv.py

Reads json_results/evaluation_master.json + model_summary.json (produced
by evaluate.py) and writes flat CSV tables for spreadsheet review and for
create_plots.py / generate_report.py to consume:

  csv/per_image_results.csv   — one row per (model, image)
  csv/model_summary.csv       — one row per model, aggregated
  csv/confusion_data.csv      — (model, ground_truth, predicted) triples
  csv/latency_memory.csv      — per-image latency / tokens-per-sec / memory
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import utils


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    config.ensure_directories()
    master_path  = config.JSON_RESULTS_DIR / "evaluation_master.json"
    summary_path = config.JSON_RESULTS_DIR / "model_summary.json"

    if not master_path.exists() or not summary_path.exists():
        utils.log("Missing evaluation_master.json / model_summary.json. Run evaluate.py first.")
        return

    master  = utils.load_json(master_path)
    summary = utils.load_json(summary_path)

    # ── per_image_results.csv ───────────────────────────────────────────
    per_image_fields = [
        "model", "image", "ground_truth_species", "ground_truth_disease",
        "predicted_species", "predicted_disease", "confidence", "correct",
        "correct_in_alternatives", "latency_sec", "tokens_per_sec", "peak_memory_gb",
        "plant_species_score", "symptom_quality_score", "disease_quality_score",
        "hallucination_score", "confidence_calibration_score", "total_judge_score",
        "generation_error",
    ]
    per_image_rows = []
    for r in master:
        per_image_rows.append({
            "model": r["model"],
            "image": r["image"],
            "ground_truth_species": r["ground_truth_species"],
            "ground_truth_disease": r["ground_truth_disease"],
            "predicted_species": r["predicted_species"],
            "predicted_disease": r["predicted_disease"],
            "confidence": r["confidence"],
            "correct": r["correct"],
            "correct_in_alternatives": r["correct_in_alternatives"],
            "latency_sec": r["latency_sec"],
            "tokens_per_sec": r["tokens_per_sec"],
            "peak_memory_gb": r["peak_memory_gb"],
            "plant_species_score": r["judge"]["plant_species_score"],
            "symptom_quality_score": r["judge"]["symptom_quality_score"],
            "disease_quality_score": r["judge"]["disease_quality_score"],
            "hallucination_score": r["judge"]["hallucination_score"],
            "confidence_calibration_score": r["judge"]["confidence_calibration_score"],
            "total_judge_score": r["judge"]["total_score"],
            "generation_error": r["generation_error"],
        })
    write_csv(config.CSV_DIR / "per_image_results.csv", per_image_rows, per_image_fields)

    # ── model_summary.csv ────────────────────────────────────────────────
    summary_fields = [
        "model", "n_images", "n_errors", "n_resolvable_ground_truth", "accuracy",
        "accuracy_incl_alternatives", "avg_latency_sec", "avg_tokens_per_sec",
        "avg_peak_memory_gb", "avg_plant_species_score", "avg_symptom_quality_score",
        "avg_disease_quality_score", "avg_hallucination_score", "hallucination_rate",
        "avg_confidence_calibration_score", "avg_total_judge_score",
    ]
    summary_rows = []
    for model_key, s in summary.items():
        row = {"model": model_key}
        row.update({k: s.get(k) for k in summary_fields if k != "model"})
        summary_rows.append(row)
    write_csv(config.CSV_DIR / "model_summary.csv", summary_rows, summary_fields)

    # ── confusion_data.csv ───────────────────────────────────────────────
    confusion_rows = [
        {"model": r["model"], "ground_truth_disease": r["ground_truth_disease"],
         "predicted_disease": r["predicted_disease"]}
        for r in master if r["ground_truth_disease"] != "Unknown"
    ]
    write_csv(config.CSV_DIR / "confusion_data.csv", confusion_rows,
              ["model", "ground_truth_disease", "predicted_disease"])

    # ── latency_memory.csv ───────────────────────────────────────────────
    lm_rows = [
        {"model": r["model"], "image": r["image"], "latency_sec": r["latency_sec"],
         "tokens_per_sec": r["tokens_per_sec"], "peak_memory_gb": r["peak_memory_gb"]}
        for r in master
    ]
    write_csv(config.CSV_DIR / "latency_memory.csv", lm_rows,
              ["model", "image", "latency_sec", "tokens_per_sec", "peak_memory_gb"])

    utils.log(f"Wrote CSVs to {config.CSV_DIR}:")
    for name in ["per_image_results.csv", "model_summary.csv", "confusion_data.csv", "latency_memory.csv"]:
        utils.log(f"  - {name}")


if __name__ == "__main__":
    main()
