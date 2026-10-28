# Evaluation Pipeline (Milestone 6) — Base vs. Fine-tuned

Compares a fine-tuned QLoRA adapter against the base
**Qwen2.5-VL-7B-Instruct-4bit** model on the frozen `test.jsonl` split, using
identical prompts and deterministic (temperature 0, no sampling) decoding for
both models. Never retrains, never regenerates the dataset, never touches the
DKB/ontology/vocabulary/concepts/templates/corpus/training pipeline — all of
that is frozen input to this milestone.

**Crop is never a flag.** It is read from `<dataset_dir>/manifest.json`'s
`crop` field (the same manifest `plantdx train`/`prepare-training` writes),
recorded into stage 1's `metadata.json`, and used to build the disease
lexicon/hallucination/clinical lexicons for stage 2 — so `--adapter
checkpoints/qwen25vl_mango_qlora --dataset artifacts/training/qwen25vl_mango_qlora/dataset`
evaluates against mango's own disease IDs and vocabulary automatically, and a
future crop needs no evaluation-code change at all.

```
Dataset  ->  Manifest  ->  Crop Resolution  ->  Inference  ->  Analysis  ->  Report
```

`<dataset_dir>` (the frozen `train/validation/test.jsonl`) carries a
`manifest.json` recording the crop it was built for; crop resolution reads
that field; stage 1 (inference) uses it to load the right label map and stage
2 (analysis) uses it to load the right disease lexicon; the report is written
to `reports/<run_name>/`.

## Two-stage architecture (and why)

Inference needs **mlx-vlm** (Apple Silicon only); the metrics stack needs
**matplotlib / scikit-learn / scipy / nltk / pycocoevalcap / rouge-score /
bert-score+torch** (heavy, CPU-portable, and never installed alongside mlx-vlm).
These two dependency sets are never installed in the same environment. The
pipeline is therefore two independent stages that communicate **exclusively**
through one frozen artifact file, `predictions.jsonl` (+ its `metadata.json`):

```bash
# Stage 1 — inference. Run in the mlx-vlm environment. Never computes metrics.
plantdx evaluate --stage inference --output-dir reports/qwen25vl_tomato_qlora/evaluation

# Stage 2 — analyze. Run in the [eval] environment. Never touches mlx-vlm.
plantdx evaluate --stage analyze  --output-dir reports/qwen25vl_tomato_qlora/evaluation

# Convenience — both, in one process, only if the current env supports both.
plantdx evaluate --stage all      --output-dir reports/qwen25vl_tomato_qlora/evaluation
```

Use the **same `--output-dir`** for both stages: stage 1 writes
`<output_dir>/raw/predictions.jsonl`; stage 2 reads exactly that path.

## One-time setup for the analyze environment

```bash
make install-eval          # pip install -e ".[eval]" + cache WordNet + BERTScore model
```

This is the **only** point where the metrics stack touches the network — it
downloads and caches WordNet/OMW (`~/nltk_data`) and the BERTScore backbone
model (`~/.cache/huggingface`) once. `plantdx evaluate --stage analyze` never
downloads anything; if a resource is missing it fails closed with a message
pointing back to this command. See `scripts/setup_eval_env.sh`.

## Official reference implementations (never approximated)

| Metric | Library | Note |
|---|---|---|
| BLEU-1..4 | `pycocoevalcap.bleu.bleu.Bleu` | cumulative n-gram precision × BP, the MS-COCO caption-eval convention |
| CIDEr | `pycocoevalcap.cider.cider.Cider` | scored over the **full** batch at once — TF-IDF needs a multi-document corpus for a non-degenerate score |
| ROUGE-L | `rouge_score.rouge_scorer` | Google Research's reference implementation |
| METEOR | `nltk.translate.meteor_score` + WordNet | exact-match + WordNet-synonym alignment, fragmentation penalty |
| BERTScore | `bert_score.score` | default backbone `roberta-large` |
| Sentence similarity | TF-IDF cosine (`scikit-learn`) | explicitly a lexical-overlap similarity, not a neural embedding — BERTScore covers that role |
| Classification metrics, confusion matrix | `scikit-learn` | |
| Paired significance, bootstrap CI | `scipy.stats` (`ttest_rel`, `wilcoxon`, `bootstrap`) | |

Disease-label extraction, hallucination detection, and clinical-correctness
checks are **deterministic keyword/phrase matching** grounded in the frozen DKB
and the compiled `artifacts/vocabulary/` artifacts (agent names, symptom
lexicon, forbidden terms) — never an LLM judge, consistent with the project's
"no LLM/VLM in the pipeline" invariant. This trades some recall (a paraphrased
symptom may not match the verbatim DKB phrase) for full reproducibility.

## Split integrity

Before inference, the pipeline reads `train.jsonl` and the target split's
`.jsonl` (paths only, never writing to either) and hard-fails
(`InvariantViolation`) if any image path appears in both. On the real frozen
tomato dataset this passes with 0 overlap (16,201 train / 910 test images).

## CLI reference

```
plantdx evaluate [--stage inference|analyze|all] [--adapter PATH] [--dataset DIR]
                  [--split test|validation|train] [--model REPO_ID]
                  [--output-dir DIR] [--batch-size N] [--max-samples N]
                  [--seed N] [--device auto|cpu|gpu]
```

All flags default to the frozen tomato/Qwen2.5-VL run
(`configs/train/qwen25vl_tomato.yaml`'s outputs); `plantdx evaluate` with no
flags evaluates that exact adapter against `test.jsonl`. `--output-dir`
itself defaults to `reports/<run_name>/evaluation`, where `<run_name>` is
`--adapter`'s own directory name — passing `--adapter
checkpoints/qwen25vl_mango_qlora` alone is enough to also route the report
directory correctly, with no separate `--output-dir` needed.

## Outputs (`<output_dir>/`)

`evaluation_summary.md`, `evaluation.json`, `metrics.json`,
`classification_report.csv`, `per_disease.csv`, `hallucinations.csv`,
`predictions.csv`, `confusion_matrix_{base,finetuned}.csv`,
`{bleu,rouge,meteor,cider}_scores.csv`, `bertscore.csv`, `latency.csv`,
`system_info.json`, `sample_comparisons.md` (50 deterministic samples),
`statistical_comparisons.json`, and `figures/` (11 PNG+SVG pairs: confusion
matrices, accuracy/metric/per-disease-F1/hallucination/response-length/latency
comparisons). Every figure uses a fixed, CVD-validated color system (never a
default-cycled or rainbow palette) and never mixes metrics of different scales
on one axis (CIDEr and BERTScore each get their own chart).

## Reproducibility manifest

`system_info.json` records the git commit, package versions (mlx/mlx-vlm/
mlx-lm/torch/transformers), adapter/corpus/ontology/vocabulary checksums, the
seed, and hardware (CPU count, memory, macOS version) — everything needed to
reproduce the exact run.

## Troubleshooting

### BERTScore: `ImportError: numpy.core.multiarray failed to import`

A **pre-existing, environment-specific** conflict, not a PlantDx bug: an old
`numba` build (compiled against NumPy 1.x) that `transformers`' model-loading
path touches indirectly via optional audio support (`librosa` → `numba`), in
an environment where NumPy has since been upgraded to 2.x. Two fixes:

```bash
# 1. Upgrade the conflicting packages (safe, reversible)
pip install -U "numba>=0.59" "llvmlite>=0.42"

# 2. Or install [eval] into a fresh virtualenv that never had numba/librosa
python3 -m venv .venv-eval && source .venv-eval/bin/activate
./scripts/setup_eval_env.sh
```

`plantdx evaluate --stage analyze` detects this specific failure and prints
the same guidance rather than a raw traceback. When BERTScore cannot load,
every other metric still computes normally — only `bertscore_f1` is affected.

### `predictions file not found`

Run `--stage inference` first, with the **same `--output-dir`** you're about
to pass to `--stage analyze`.

### CIDEr looks unexpectedly low on a tiny sample

CIDEr's TF-IDF weighting needs a multi-document corpus; on very small
`--max-samples` runs (a handful of images) the IDF statistics are too sparse
for a meaningful score. This is expected CIDEr behavior, not a bug — it
converges as the sample count grows toward the real ~910-image test split.
