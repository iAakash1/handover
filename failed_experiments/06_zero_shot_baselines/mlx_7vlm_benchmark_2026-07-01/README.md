# PlantDx — Zero-Shot VLM Benchmark (Tomato)

Runs one fixed prompt across seven local MLX vision-language models on a
set of tomato leaf images, and records every model's response — raw
text, parsed fields, timing, memory, token throughput, and full
provenance metadata. **This phase only collects standardized
predictions.** It does not score them, does not compare anything
against folder names, and does not compute accuracy. Ground-truth
comparison against expert annotations and report generation are a
separate, later phase.

## What this does

For every image in `dataset/`, every model listed in `scripts/config.py`
is sent the exact same prompt (below) and its response is recorded as
one self-contained JSON record containing: the exact prompt sent, the
raw response, the response parsed into fields, generation metrics,
timestamps, image metadata (including a SHA-256 hash for later
duplicate detection), and model metadata.

Folder names inside `dataset/` (e.g. `early_blight/`) exist only to
organize the images on disk. They are never sent to a model, and never
read or compared anywhere in this codebase.

## Prompt (identical for every model, every image, unchanged from spec)

```
You are an expert agricultural plant pathologist.
This image is of a TOMATO crop.
Analyze the leaf carefully.
Return ONLY the following fields.
Crop:
Visible Symptoms:
Disease:
Alternative Diagnoses:
Severity:
Confidence (0-100):
Reasoning:
Do not use external knowledge.
Only describe what can be visually observed.
If uncertain, explicitly say so instead of guessing.
```

## Models

- `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`
- `mlx-community/Qwen3-VL-8B-Instruct-4bit`
- `mlx-community/gemma-3-12b-it-4bit`
- `mlx-community/InternVL3-8B-MLX-4bit`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`
- `mlx-community/Molmo-7B-D-0924-4bit`
- `mlx-community/deepseek-vl2-small-4bit`

Edit the `MODELS` list in `scripts/config.py` to add, remove, or reorder.

## Project structure

```
benchmark/
├── README.md
├── requirements.txt
├── dataset/               # tomato images, one subfolder per class
│   ├── bacterial_spot/
│   ├── early_blight/
│   ├── leaf_mold/
│   ├── mosaic_virus/
│   ├── target_spot/
│   └── yellowleaf_curl_virus/
├── scripts/
│   ├── config.py           # models, prompt, paths, generation settings, seed
│   ├── dataset_utils.py    # dataset traversal + image metadata
│   ├── metadata_utils.py   # image metadata (hash/dims) + model metadata parsing
│   ├── parser.py           # extracts labeled fields from raw model output
│   ├── mem_utils.py        # cross-version peak-memory + RNG seeding helpers
│   ├── logging_utils.py    # per-model console + file logging
│   ├── model_runner.py     # MLX-VLM load / generate wrapper
│   ├── validate.py         # structural integrity checks -> validation_report.json
│   ├── aggregate.py        # merges per-model JSON into combined JSON + CSV
│   ├── summary.py          # run statistics + environment -> summary.json
│   └── run_benchmark.py    # main entry point
├── outputs/                # JSON: per-model files, combined file, summary, validation
├── results/                # predictions.csv — all models, all images, flattened
└── logs/                   # one .log per model, plus run.log
```

## Prerequisites

- macOS on Apple Silicon
- Python 3.10+
- The seven models already available to `mlx-vlm` locally. This project
  does not download or manage models — it assumes they're already
  cached, exactly as described in the task.

## Setup

```bash
cd benchmark
pip install -r requirements.txt
```

## Run

Run everything (all 7 models × all 30 images):

```bash
python scripts/run_benchmark.py
```

Run a subset of models:

```bash
python scripts/run_benchmark.py --models mlx-community/Qwen2.5-VL-7B-Instruct-4bit,mlx-community/gemma-3-12b-it-4bit
```

Quick smoke test (first image of each class only):

```bash
python scripts/run_benchmark.py --limit 1
```

Re-run everything, ignoring any previously completed model outputs:

```bash
python scripts/run_benchmark.py --force
```

Rebuild derived outputs from whatever is currently in `outputs/`,
without running any models:

```bash
python scripts/validate.py    # -> outputs/validation_report.json
python scripts/aggregate.py   # -> outputs/all_predictions.json, results/predictions.csv
python scripts/summary.py     # -> outputs/summary.json
```

`run_benchmark.py` calls all three automatically, in that order, at the
end of every invocation.

## Output

**`outputs/predictions_<model>.json`** — one file per model, written
incrementally as each image finishes, so a crash partway through a model
doesn't lose the images it already completed.

**`outputs/all_predictions.json`** and **`results/predictions.csv`** —
every (image, model) prediction combined into one place, sorted by image
then model. The CSV is the same data flattened into columns (nested
sections become prefixed columns, e.g. `metrics.peak_memory_mb` →
`metric_peak_memory_mb`).

**`outputs/validation_report.json`** — structural integrity check run
before aggregation: malformed JSON files, records missing expected
fields, duplicate (image, model) records, and internally inconsistent
records (e.g. a `"success"` record with an empty `raw_output`). This is
**not** a correctness check — it never looks at whether a prediction is
right, only whether the collected data is well-formed.

**`outputs/summary.json`** — run statistics: number of models, images,
completed/failed runs (broken down by failure type), average runtime,
the most recent run's wall-clock duration, and an environment snapshot
(Python/mlx/mlx-vlm versions, seed, temperature, max tokens) for
reproducibility.

**`logs/<model>.log`** and **`logs/run.log`** — persistent, timestamped
logs mirroring console output: load status, per-image progress,
warnings, and errors.

### Per-record JSON schema

```json
{
  "image": "/absolute/path/to/image.JPG",
  "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
  "prompt": "You are an expert agricultural plant pathologist...",
  "status": "success",
  "error": null,
  "raw_output": "Crop: Tomato\nVisible Symptoms: ...",
  "parsed_output": {
    "crop": "Tomato",
    "visible_symptoms": "...",
    "disease": "Early Blight",
    "alternative_diagnoses": "...",
    "severity": "Moderate",
    "confidence": "72",
    "reasoning": "..."
  },
  "metrics": {
    "prompt_tokens": null,
    "generated_tokens": null,
    "prompt_tokens_per_sec": null,
    "generation_tokens_per_sec": null,
    "peak_memory_mb": 5423.71,
    "inference_time_sec": 6.184
  },
  "timestamps": {
    "start": "2026-07-01T12:00:00.000000+00:00",
    "finish": "2026-07-01T12:00:06.184000+00:00",
    "runtime_sec": 6.184
  },
  "image_metadata": {
    "filename": "1.JPG",
    "relative_path": "early_blight/1.JPG",
    "class_folder": "early_blight",
    "width": 256,
    "height": 256,
    "file_size_bytes": 10761,
    "sha256": "bb8cf07350baac37f7fdfa2a9e54587363c8a8806a28e8d0b085c1de9ab4a3cb"
  },
  "model_metadata": {
    "model_id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "model_name": "Qwen2.5-VL-7B-Instruct-4bit",
    "model_family": "Qwen",
    "parameters": "7B",
    "quantization": "4bit",
    "mlx_model": true
  }
}
```

`status` is one of:

| status | meaning |
|---|---|
| `success` | generation and parsing both completed |
| `generation_failed` | the model call itself raised — `raw_output` and `parsed_output` are null, `error` has the exception message |
| `parse_failed` | generation succeeded but parsing raised (rare) — `raw_output` is still preserved in full, only `parsed_output` is null |
| `model_load_failed` | the model failed to load at all — every field is null except `error`, `image_metadata`, and `model_metadata` (both of which are knowable without the model ever loading) |

`metrics.prompt_tokens`, `generated_tokens`, `prompt_tokens_per_sec`,
and `generation_tokens_per_sec` are `null` whenever the installed
`mlx-vlm` version's result object doesn't expose them — see Design notes.

## Design notes

- **Models load once, not once per image.** The benchmark loops
  model-outer / image-inner (load a model, run it over every image,
  unload, load the next model) instead of reloading a multi-GB model 210
  times. Every image is still run through every model — the resulting
  data is identical either way; this order just avoids ~200 unnecessary
  model loads.
- **Resume behavior.** A model is skipped on the next run only if its
  `predictions_<model>.json` has a record for every image *and* the
  model loaded successfully at least once. If a model failed to load
  entirely (every record is `model_load_failed`), it is **not** treated
  as complete, and the next run retries loading it automatically — a
  load failure is often fixable (model not yet downloaded, a missing
  dependency) and worth retrying without needing `--force`. A model that
  loaded fine but had a few failed images is still treated as complete,
  since those failures were recorded, not silently dropped — pass
  `--force` to redo it anyway.
- **Failure isolation.** A failure at any single image, or a parsing
  failure on an otherwise-successful generation, is caught, logged, and
  recorded as its own failed record — it never stops the remaining
  images or models. A failure to load a model at all produces a
  placeholder failed record for every image (see the `status` table
  above) rather than silently producing no data for that model.
- **Parsing recognizes common label variants, but never guesses.** A
  field is populated only when the model's response uses one of a
  curated set of accepted labels for it (see `FIELD_VARIANTS` in
  `scripts/parser.py`) — e.g. `Confidence`, `Confidence Score`,
  `Confidence Level`, and `Certainty` all resolve to the same
  `confidence` field, case-insensitively, with or without a trailing
  `(0-100)`/`(%)` qualifier. A label that isn't on the list for any
  field is still recognized as *a* field boundary (so its text can't
  bleed into a neighboring recognized field) but leaves that field
  `null` rather than guessing at a match. `raw_output` always has the
  complete, untouched text regardless, so nothing is ever lost — and a
  model's tendency to deviate from the requested labels is itself
  useful signal for the evaluation phase, not something to paper over.
- **Reproducibility.** `TEMPERATURE = 0.0` (greedy decoding) is
  deterministic on its own for most models; `mx.random.seed()` is also
  set at startup as a defensive extra layer, in case any model's code
  path samples instead of taking the argmax. `summary.json` records the
  Python/mlx/mlx-vlm versions and generation settings used for a given
  run, so a result can be traced back to the environment that produced
  it. The exact prompt is stored on every single record for the same
  reason — reproducibility shouldn't depend on cross-referencing a
  separate config file.
- **Token/throughput metrics are best-effort.** They're read from
  `mlx-vlm`'s generation result object via several candidate attribute
  names, since this has varied across versions of the library. If your
  installed version doesn't expose them, they're `null` — never
  estimated or fabricated. `peak_memory_mb` and `inference_time_sec` are
  measured directly by this project either way, so they're always
  populated on a successful generation.
- **A model's first image may run slightly slower** than the rest — MLX
  does one-time kernel compilation on a model's first generation call.
  That shows up in `inference_time_sec` for that one image and is
  expected, not a bug.
- **`peak_memory_mb` may be `null`** if your installed `mlx` version
  doesn't expose a peak-memory API that `mem_utils.py` recognizes.
  Everything else continues to run normally.
- **Model metadata is parsed from the repo id string, never memorized.**
  `model_family`, `parameters`, and `quantization` are all derived by
  pattern-matching the model's own MLX repo name (e.g. `"7B"` in
  `Qwen2.5-VL-7B-Instruct-4bit`) in `scripts/metadata_utils.py`. If a
  model's name doesn't spell out a parameter count —
  `deepseek-vl2-small-4bit` has no explicit number — `parameters` is
  `null` rather than an invented figure.
- **This phase does not evaluate correctness.** No comparison against
  folder names, and no accuracy/F1/etc. of any kind is computed anywhere
  in this codebase — that's intentionally left for a later phase once
  expert ground-truth annotations exist.
