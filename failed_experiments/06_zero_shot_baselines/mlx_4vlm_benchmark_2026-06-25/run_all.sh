#!/usr/bin/env bash
set -e

# Crop Disease VLM Benchmark Suite — full pipeline runner.
# Run from the project root after:
#   conda activate vlm
#   (copying images into ./images/)
#
# NOTE: evaluate.py is included between judge_outputs.py and
# generate_csv.py. It isn't optional — generate_csv.py / create_plots.py /
# generate_report.py all read evaluation_master.json and model_summary.json,
# which only evaluate.py produces (it merges benchmark.py's raw outputs,
# extract_ground_truth.py's labels, and judge_outputs.py's scores into one
# dataset and computes accuracy/calibration).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "vlm" ]; then
  echo "WARNING: conda environment 'vlm' does not appear to be active."
  echo "  Run: conda activate vlm"
  echo "Continuing in 5 seconds anyway (Ctrl+C to abort)..."
  sleep 5
fi

if [ -z "$(ls -A images 2>/dev/null | grep -v '.gitkeep')" ]; then
  echo "ERROR: images/ has no images. Copy your crop disease images into:"
  echo "  $PROJECT_ROOT/images/"
  exit 1
fi

echo "=================================================="
echo " 1/7  benchmark.py            — running all VLMs"
echo "=================================================="
python scripts/benchmark.py

echo "=================================================="
echo " 2/7  extract_ground_truth.py — inferring labels"
echo "=================================================="
python scripts/extract_ground_truth.py

echo "=================================================="
echo " 3/7  judge_outputs.py        — LLM-as-a-judge scoring"
echo "=================================================="
python scripts/judge_outputs.py

echo "=================================================="
echo " 4/7  evaluate.py             — merging results + metrics"
echo "=================================================="
python scripts/evaluate.py

echo "=================================================="
echo " 5/7  generate_csv.py         — writing CSV tables"
echo "=================================================="
python scripts/generate_csv.py

echo "=================================================="
echo " 6/7  create_plots.py         — rendering plots"
echo "=================================================="
python scripts/create_plots.py

echo "=================================================="
echo " 7/7  generate_report.py      — building final report"
echo "=================================================="
python scripts/generate_report.py

echo "=================================================="
echo " DONE."
echo " Report   : reports/final_report.md / final_report.pdf"
echo " CSVs     : csv/"
echo " Plots    : plots/"
echo " Raw JSON : json_results/"
echo "=================================================="
