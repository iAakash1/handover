"""
scripts/evaluate.py

Merges the outputs of benchmark.py, extract_ground_truth.py, and
judge_outputs.py into one per-(model, image) master record, and computes
per-model aggregate metrics:

  - accuracy                 : programmatic fuzzy match between the
                                model's parsed "Most likely disease" and
                                the filename-derived ground truth
                                (independent of the LLM judge)
  - accuracy_incl_alternatives: same, but also counts a hit if the
                                correct disease appears in "Alternative
                                diagnoses"
  - calibration / ECE        : Expected Calibration Error computed
                                directly from each model's stated
                                confidence (0-100) vs. objective
                                correctness — NOT the judge's subjective
                                calibration score, a separate, independent
                                cross-check on the same dimension
  - avg_* judge scores        : straight averages of judge_outputs.py's
                                per-component scores (null if judging
                                was skipped)
  - latency / tokens-per-sec / peak memory averages

NOTE: this script is not mentioned in the original spec's run_all.sh
sequence, but generate_csv.py / create_plots.py / generate_report.py all
read its two output files below, so it's a required pipeline stage —
run_all.sh in this project calls it between judge_outputs.py and
generate_csv.py.

Outputs:
  json_results/evaluation_master.json   (one row per model x image)
  json_results/model_summary.json       (one row per model, aggregated)
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import utils


def compute_calibration(records):
    """
    Expected Calibration Error (ECE) over 10 equal-width confidence bins,
    using only records with a parsed numeric confidence AND a resolvable
    ground truth. Returns None if there's nothing usable.
    """
    usable = [r for r in records
              if r["confidence"] is not None and r["ground_truth_disease"] != "Unknown"]
    if not usable:
        return None

    n_bins = 10
    bins = [[] for _ in range(n_bins)]
    for r in usable:
        conf = max(0.0, min(100.0, r["confidence"])) / 100.0
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append(r)

    total = len(usable)
    ece = 0.0
    bin_stats = []
    for i, b in enumerate(bins):
        if not b:
            bin_stats.append({"bin": i, "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = sum((max(0.0, min(100.0, r["confidence"])) / 100.0) for r in b) / len(b)
        acc = sum(1 for r in b if r["correct"]) / len(b)
        ece += (len(b) / total) * abs(avg_conf - acc)
        bin_stats.append({"bin": i, "count": len(b), "avg_confidence": avg_conf, "accuracy": acc})

    return {"ece": ece, "n_used": total, "bins": bin_stats}


def _avg(records, key_path):
    vals = []
    for r in records:
        v = r
        ok = True
        for k in key_path:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                ok = False
                break
        if ok and isinstance(v, (int, float)):
            vals.append(v)
    return (sum(vals) / len(vals)) if vals else None


def main():
    config.ensure_directories()
    raw_path   = config.JSON_RESULTS_DIR / "all_raw_results.json"
    gt_path    = config.JSON_RESULTS_DIR / "ground_truth.json"
    judge_path = config.JSON_RESULTS_DIR / "judge_scores.json"

    if not raw_path.exists() or not gt_path.exists():
        utils.log("Missing benchmark.py / extract_ground_truth.py outputs. Run those first.")
        return

    raw_results  = utils.load_json(raw_path)
    ground_truth = utils.load_json(gt_path)
    judge_scores = utils.load_json(judge_path) if judge_path.exists() else []
    judge_lookup = {(j["model"], j["image"]): j for j in judge_scores}

    master = []
    for rec in raw_results:
        gt = ground_truth.get(rec["image"], {})
        gt_disease = gt.get("ground_truth_disease", "Unknown")
        gt_species = gt.get("ground_truth_species", "Unknown")
        parsed = utils.parse_model_output(rec.get("raw_output", ""))

        predicted_disease = parsed.get("most_likely_disease", "")
        correct = utils.labels_match(predicted_disease, gt_disease)
        correct_in_alternatives = utils.labels_match(
            parsed.get("alternative_diagnoses", ""), gt_disease
        )

        judge = judge_lookup.get((rec["model"], rec["image"]), {})

        master.append({
            "model": rec["model"],
            "image": rec["image"],
            "ground_truth_disease": gt_disease,
            "ground_truth_species": gt_species,
            "predicted_disease": predicted_disease,
            "predicted_species": parsed.get("plant_species", ""),
            "confidence": parsed.get("confidence"),
            "correct": bool(correct),
            "correct_in_alternatives": bool(correct_in_alternatives),
            "latency_sec": rec.get("latency_sec"),
            "tokens_per_sec": rec.get("tokens_per_sec"),
            "peak_memory_gb": rec.get("peak_memory_gb"),
            "generation_error": rec.get("error"),
            "parsed_fields": parsed,
            "raw_output": rec.get("raw_output", ""),
            "judge": {
                "plant_species_score": judge.get("plant_species_score"),
                "symptom_quality_score": judge.get("symptom_quality_score"),
                "disease_quality_score": judge.get("disease_quality_score"),
                "hallucination_score": judge.get("hallucination_score"),
                "confidence_calibration_score": judge.get("confidence_calibration_score"),
                "total_score": judge.get("total_score"),
                "justification": judge.get("justification"),
                "judge_error": judge.get("judge_error"),
            },
        })

    utils.save_json(master, config.JSON_RESULTS_DIR / "evaluation_master.json")

    by_model = defaultdict(list)
    for r in master:
        by_model[r["model"]].append(r)

    summary = {}
    for model_key, records in by_model.items():
        resolvable = [r for r in records if r["ground_truth_disease"] != "Unknown"]
        n_errors   = sum(1 for r in records if r["generation_error"])

        accuracy = (sum(1 for r in resolvable if r["correct"]) / len(resolvable)
                    if resolvable else None)
        accuracy_incl_alt = (
            sum(1 for r in resolvable if r["correct"] or r["correct_in_alternatives"]) / len(resolvable)
            if resolvable else None
        )
        hallucination_score_avg = _avg(records, ["judge", "hallucination_score"])
        hallucination_rate = (
            (1 - hallucination_score_avg / 20) if hallucination_score_avg is not None else None
        )

        summary[model_key] = {
            "n_images": len(records),
            "n_errors": n_errors,
            "n_resolvable_ground_truth": len(resolvable),
            "accuracy": accuracy,
            "accuracy_incl_alternatives": accuracy_incl_alt,
            "avg_latency_sec": _avg(records, ["latency_sec"]),
            "avg_tokens_per_sec": _avg(records, ["tokens_per_sec"]),
            "avg_peak_memory_gb": _avg(records, ["peak_memory_gb"]),
            "avg_plant_species_score": _avg(records, ["judge", "plant_species_score"]),
            "avg_symptom_quality_score": _avg(records, ["judge", "symptom_quality_score"]),
            "avg_disease_quality_score": _avg(records, ["judge", "disease_quality_score"]),
            "avg_hallucination_score": hallucination_score_avg,
            "hallucination_rate": hallucination_rate,
            "avg_confidence_calibration_score": _avg(records, ["judge", "confidence_calibration_score"]),
            "avg_total_judge_score": _avg(records, ["judge", "total_score"]),
            "calibration": compute_calibration(records),
        }

    utils.save_json(summary, config.JSON_RESULTS_DIR / "model_summary.json")

    utils.log("Saved evaluation_master.json and model_summary.json")
    for k, v in summary.items():
        utils.log(f"  {k}: accuracy={v['accuracy']}, avg_total_judge_score={v['avg_total_judge_score']}")


if __name__ == "__main__":
    main()
