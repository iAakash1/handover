# Crop Disease VLM Benchmark Suite

Compares multiple open-source vision-language models (run locally via
**MLX-VLM** on Apple Silicon) on zero-shot crop disease diagnosis from
leaf images — accuracy, symptom-description quality, hallucination rate,
confidence calibration, reasoning quality, latency, and peak memory.

This is a separate effort from the Qwen2.5-VL fine-tuning pipeline
(PlantDx): that project fine-tunes one model for production captioning;
this suite benchmarks several **off-the-shelf, non-fine-tuned** VLMs
against each other in a zero-shot setting.

---

## 1. Requirements

- macOS, Apple Silicon (designed/sized for an M4 Pro, 24GB unified memory)
- Python 3.12
- conda

## 2. Setup

```bash
conda create -n vlm python=3.12 -y
conda activate vlm
pip install -r requirements.txt
```

`mlx` and `mlx-vlm` move quickly; if model loading or `generate()` throws
a `TypeError` about unexpected arguments, see **Troubleshooting** below.

## 3. Add your images

Copy your crop disease images directly into:

```
crop_vlm_benchmark/images/
```

Flat folder, no subfolders — ground truth is inferred per-image from the
**filename**, not folder structure (see Section 5).

## 4. API key for the LLM judge (optional but recommended)

`judge_outputs.py` scores every model response using GPT-4.1 or
Gemini 2.5 Pro as an impartial judge. Set ONE of:

```bash
export OPENAI_API_KEY="sk-..."          # default provider: openai
# or
export VLM_JUDGE_PROVIDER="gemini"
export GEMINI_API_KEY="..."
```

If no key is set, judging is skipped automatically — you still get
accuracy, latency, and peak-memory results; judge-score columns/plots
will just show as N/A.

## 5. Run

```bash
conda activate vlm
bash run_all.sh
```

This runs, in order:

1. `benchmark.py` — every model over every image (raw outputs, latency, tokens/sec, peak memory)
2. `extract_ground_truth.py` — disease labels inferred from filenames
3. `judge_outputs.py` — LLM-as-a-judge scoring
4. `evaluate.py` — merges everything + computes accuracy / calibration
5. `generate_csv.py` — CSV tables
6. `create_plots.py` — PNG plots
7. `generate_report.py` — `reports/final_report.md` and `final_report.pdf`

You can also run any script individually, e.g. `python scripts/benchmark.py`,
as long as the scripts before it in the sequence have already run (each
stage reads the previous stage's JSON output from `json_results/`).

> **Note on the pipeline order:** the original spec's `run_all.sh` listed
> six scripts and didn't include `evaluate.py`. It's included here as
> step 4 because `generate_csv.py` / `create_plots.py` / `generate_report.py`
> all depend on `evaluation_master.json` and `model_summary.json`, which
> only `evaluate.py` produces — it's the merge step between raw judge/benchmark
> output and everything downstream, not an optional extra.

## 6. Outputs

| Path | Contents |
|---|---|
| `results/<model>/*.txt` | Raw per-image text output, for quick manual review |
| `json_results/*.json` | Structured intermediate + final data at every stage |
| `csv/*.csv` | Per-image results, model summary, confusion pairs, latency/memory |
| `plots/*.png` | Bar charts, confusion matrices, radar chart |
| `reports/final_report.md`, `final_report.pdf` | The full report |

In this repository (handover note): the June 2026 run's raw answers are in `results/` and `json_results/`, and its
report is in [`report/`](report/final_report.md) (`final_report.md`, `crop_vlm_benchmark_report.pdf`, `charts/`).
`csv/`, `plots/` and `reports/` are empty until you run the pipeline again.

## 7. Known limitations (read before trusting the numbers)

- **Ground truth is filename-derived and incomplete by design.** Many
  PlantVillage-style filenames encode the disease in a short research
  code (`RS_LB` → Late Blight, `JR_B.Spot` → Bacterial Spot), but some
  use pure site/photographer codes that don't encode the disease at all
  (e.g. a code like `PSU_CG` gives no usable signal). Unmatched images
  are labeled `"Unknown"` and excluded from accuracy/calibration scoring
  rather than guessed — `extract_ground_truth.py` reports what fraction
  of your images this affects. If you recognize more codes in your own
  data, add them to `SINGLE_TOKEN_MAP` / `BIGRAM_MAP` in that file.
- **`mlx-vlm`'s API changes across versions.** `scripts/utils.py::run_mlx_vlm_generate`
  tries the current documented call shape, falls back to an older one on
  `TypeError`, and recovers latency/tokens-per-sec/peak-memory from
  `mlx-vlm`'s own verbose stdout when possible — otherwise it computes
  its own latency/token estimate. If a model still fails outright, the
  error is recorded per-image and the benchmark continues with the next
  model rather than crashing.
- **24GB unified memory is shared, not dedicated VRAM.** Running a
  12B-parameter 4-bit model (gemma3) and an 11B-parameter 4-bit model
  (llama32vision) is feasible but leaves limited headroom. `benchmark.py`
  loads and frees one model at a time. Close other GPU-heavy apps before
  running, and if you hit memory pressure, comment out a model in
  `config.MODELS` and run the rest first.
- **The "hallucination" rubric component is a groundedness score** (0-20,
  higher = better / less hallucination) despite the name — this matches
  the points-out-of-100 framing in the rubric, but don't read a high
  `hallucination_score` as "more hallucination."
- **Confusion matrices canonicalize free-text predictions** onto the
  known ground-truth label set via fuzzy string matching (see
  `utils.labels_match`) so the matrix stays readable; predictions that
  don't fuzzy-match anything are bucketed as "Other / Unmatched."

## 8. Troubleshooting

**`generate()` raises `TypeError: unexpected keyword argument ...`**
Check your installed `mlx-vlm` version's actual signature:
```bash
python -c "import mlx_vlm, inspect; print(inspect.signature(mlx_vlm.generate))"
```
Update the call inside `scripts/utils.py::run_mlx_vlm_generate` to match.

**A model OOMs or fails to download**
It's recorded as an error for that model across all images; the rest of
the benchmark still runs. Check `json_results/<model>_results.json` for
the `error` field.

**PDF generation fails in `generate_report.py`**
The Markdown report (`reports/final_report.md`) is still written. The
script prints a `pandoc` fallback command to convert it manually.

**Judge scores are all `null`**
No `OPENAI_API_KEY` / `GEMINI_API_KEY` was set when `judge_outputs.py`
ran — see Section 4. Everything downstream still works without them.
