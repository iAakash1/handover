# PlantDx Demo App (Streamlit)

A polished, presentation-only inference app over the **already-trained** tomato
and mango QLoRA adapters. It reuses `plantdx.training.inference` /
`plantdx.evaluation` unchanged and never trains, evaluates, or regenerates
anything — everything new lives in `app/` and writes only to gitignored
`uploads/`, `predictions/`, and `logs/`.

## Launch

Use the **absolute path** to the Python interpreter that has a working ML stack
+ mlx-vlm (the same one training/evaluation use):

```bash
~/miniforge3/envs/vlm/bin/python -m streamlit run streamlit_app.py
```

> Plain `streamlit run …` (and even `conda activate vlm`) can silently pick a
> different Python. On a machine where the Framework Python has an old
> `numba` vs NumPy 2.x, that interpreter can't import mlx-vlm — the app then
> shows a clear banner with this exact command instead of crashing. See
> [`EVALUATION.md`](EVALUATION.md#troubleshooting).

Dependencies: `pip install -r app/requirements.txt` (just Streamlit) into the
same environment; mlx-vlm comes from `pip install -e ".[train]"`.

## What it does

Upload one or more leaf images (JPG/JPEG/PNG, multiple allowed) → the selected
crop's adapter captions each image → the caption is classified to a DKB disease
→ a grounded result card is shown, the original is filed by predicted class, and
a record + log line are written.

Three tabs:

- **🔬 Diagnose** — upload, per-image result cards, downloads (JSON / Markdown).
- **🧬 Adapter & model** — adapter verification (proves the LoRA adapter, not the
  base model, is attached) + runtime status (warm/cold, load time).
- **📊 Evaluation** — the crop's held-out `plantdx evaluate` results
  (base-vs-fine-tuned accuracy / macro-F1 / per-disease), read from
  `reports/<run>/evaluation/`.

## Confidence, thresholds, and unknowns

The one-shot `generate()` only returns the **last** token's full-vocabulary
logprobs, which is not a usable confidence (it averages to ~1e-9). The app
instead iterates **`stream_generate`** and averages the probability the model
assigned to **its own emitted tokens** — a real per-generation confidence
(≈0.85–0.94 on correct, in-distribution predictions).

Each prediction is bucketed:

| Status | When | UI |
|---|---|---|
| **confident** | a disease is named and confidence ≥ threshold | green, symptoms shown, filed under the class |
| **low_confidence** | a disease is named but confidence < threshold | orange, tentative wording, filed under `unknown/` |
| **unknown** | the caption names no DKB disease | grey, "differs from training distribution", filed under `unknown/` |

The **confidence threshold** is a sidebar slider (default 0.55). It is a
heuristic proxy for generation certainty, **not** a calibrated
out-of-distribution detector — casual field/phone photos differ from the
PlantVillage training data and legitimately score lower.

## Robust classification

`app.classification.classify` first calls the frozen production extractor
(`plantdx.evaluation.classification`); only on `unclassified` does it retry with
a normalized, alias-aware, **crop-scoped** matcher built from the same DKB
fields. This fixes real gaps (`dieback` vs `die back`, `sooty mold` vs `sooty
mould`, acronyms like `TYLCV`, punctuation) without ever changing production
behavior or the evaluation numbers, and never returns another crop's disease.

## Adapter verification

The **Adapter & model** tab reads `adapter_config.json` + `adapters.safetensors`
and shows: base model, adapter run/checkpoint, fine-tune type, LoRA rank/scale,
number of adapted modules, **trainable-parameter count** (~40.4M for tomato),
adapter tensor count, and a weights checksum — with an explicit **"LoRA adapter
attached ✅"** confirmation so you can be sure the fine-tuned adapter (not the
base model) is in use.

## Storage & logs (all gitignored)

```
uploads/<crop>/<predicted_class|unknown>/<timestamp>_<uuid>_<name>.<ext>   # immutable original
uploads/<crop>/<...>/<timestamp>_<uuid>_<name>.json                        # metadata sidecar
predictions/<timestamp>.json                                               # full record (reopenable from History)
predictions/history.json                                                   # compact sidebar index
logs/predictions.jsonl                                                     # append-only debug log
logs/plantdx_app.log                                                       # model-load / inference / error events
```

Originals are written **once**, never overwritten (timestamp + uuid filenames),
and filed by predicted class. The `predictions.jsonl` line carries timestamp,
filename, crop, prediction, status, confidence, generation, adapter run, latency,
and model.

## Performance

- **Model loads once** per crop via `st.cache_resource(max_entries=1)` and stays
  warm for the whole session — predictions never reload it. Load time (~5 s) is
  shown in the Adapter tab.
- `max_entries=1` keeps only one crop's 7B model resident; switching crops evicts
  the other rather than holding two in 24 GB.
- MLX's Metal buffer cache is bounded (`set_cache_limit(4 GB)`) and cleared after
  every inference, so memory stays flat (~5.8 GB active) across many predictions
  — the earlier segfault-under-memory-pressure is gone.

## Module map (`app/`)

| Module | Responsibility |
|---|---|
| `streamlit_app.py` | Entry point; page config; last-resort guard. |
| `utils.py` | Paths, crop profiles, thresholds, filename/format helpers. |
| `classification.py` | Robust crop-scoped disease extraction (wraps production). |
| `inference.py` | Cached model load, real-confidence `stream_generate`, adapter verification, status. |
| `storage.py` | Organized immutable uploads, sidecars, records, JSONL log. |
| `history.py` | History index + reopen. |
| `evaluation_view.py` | Reads `reports/<run>/evaluation/` for the Evaluation tab. |
| `logging_setup.py` | File logger. |
| `components.py` | Result cards, confidence states, adapter/evaluation panels. |
| `ui.py` | Sidebar + tabs + the upload→infer→save→log flow. |

Tests live in `tests/unit/app/` (classification, upload organization, logging,
history, evaluation reading, confidence/status logic, adapter parsing). Tests
needing Streamlit skip cleanly where it isn't installed, matching the repo's
optional-dependency pattern.

## Honest limitations

- The model was trained/evaluated **only** on PlantVillage-style single-leaf
  images (93.7% tomato / 82.0% mango on their held-out test splits). Casual field
  or phone photos are out-of-distribution and will misfire more often; the app
  surfaces this via low-confidence/unknown states rather than asserting a
  confident wrong answer.
- Confidence is a generation-certainty heuristic, not a calibrated OOD score.
- Real inference requires Apple Silicon + mlx-vlm; without it the app still runs
  (upload/history/UI) and shows a friendly message.
