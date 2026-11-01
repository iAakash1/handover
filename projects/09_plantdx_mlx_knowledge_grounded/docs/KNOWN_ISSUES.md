# Known Issues

Issues that are real, understood, and **intentionally deferred** rather than fixed
immediately — either because fixing them now would mean touching finished,
behavior-frozen milestones, or because the fix belongs to a different concern
than the one currently in scope. Each entry states what happens, why, why it is
safe to leave as-is, and where the eventual fix belongs.

---

## 1. Repeated `plantdx normalize` runs report 0 images on an already-normalized dataset

**Status:** known, deferred. **Component:** `src/plantdx/normalization/`. **Severity:** low (cosmetic/reporting only — no data loss, no corruption).

### What happens

Running `plantdx normalize --dataset mango` a second time (after a first run has
already populated `datasets/mango/processed/`) produces:

```
Normalization complete: 0 images across 1 crop(s) into datasets/
  mango: 0 images, 0 classes, checksum e3b0c44298fc
```

`datasets/mango/manifest.json`, `class_mapping.json`, `dataset_card.md`, and
`normalization_report.json` are all rewritten to report `0` images and `0`
classes, and `datasets/mango/processed/` **is not touched or modified** — it
still contains every previously normalized file, correctly named and organized.
`e3b0c44298fc` is the well-known SHA-256 of an empty string, confirming the
checksum, too, is computed over an (in this run) empty image set.

### Why it happens

`normalize_crop` (`src/plantdx/normalization/engine.py`) computes each source
file's SHA-256 and calls `_place_file`, which checks whether a file with the
same name already exists at the destination:

- If the destination file exists **and its content hash matches**, the source
  is classified `"duplicate"` and is deliberately **not** re-copied (this is
  the correct, intentional behavior — it is what makes re-running `normalize`
  idempotent and fast, and it is what the raw-dataset-immutability +
  checksum-verification guarantees of the Normalization Engine design rely on).
- The bug is one level up: `normalize_crop`'s `images` list — the list that
  `manifest.json`/`class_mapping.json`/`dataset_card.md` are built from — only
  accumulates files that were *freshly placed in this run*. A `"duplicate"`
  status causes an early `continue` (`engine.py`, `_build_condition`-adjacent
  loop in `normalize_crop`) that skips appending to `images` entirely. So the
  **manifest describes "what this run did," not "the current full contents of
  `processed/`."** On a fully-idempotent re-run, "what this run did" is
  correctly "nothing" — but the manifest should instead always describe the
  full, current state of `processed/`, regardless of which files were touched
  in this particular invocation.

### Why it is not dangerous

- **No files are deleted, moved, renamed, or corrupted.** `processed/` is
  strictly additive/idempotent; this is a reporting-artifact staleness issue,
  not a data-integrity issue.
- **The raw datasets are never touched** — the immutability guarantee (the
  Normalization Engine's core invariant) holds regardless of this bug.
- **Every file that** ***is*** **freshly copied in a run is still fully
  verified** (checksum-compared against its source) before being counted —
  the defect is purely about what happens to files that were *already*
  correctly in place from a prior run.
- A **fresh** normalize run (empty `datasets/<crop>/processed/`, e.g. right
  after `git clone` or after clearing `datasets/`) is entirely unaffected:
  nothing is a "duplicate" yet, so every file is freshly placed, counted, and
  correctly reported. This is the common case (a first-time setup) and it
  works correctly.

### Why it was intentionally deferred

This was discovered during CI-stabilization work that was explicitly scoped
to **style-only fixes with zero behavior change** (Ruff/mypy/pytest cleanup).
Fixing it requires changing `normalize_crop`'s control flow — specifically,
making the manifest/report always reflect a full re-scan of `processed/`
rather than only this run's newly-placed files — which is a genuine, if small,
**behavior change to the Normalization Engine**, explicitly out of scope for a
style-preservation pass. The Normalization Engine is feature-complete and
behavior-frozen per the current milestone plan; this fix belongs to a
dedicated Normalization Engine bugfix task, not to a CI-stabilization or
Vocabulary Builder milestone.

### Suggested eventual fix (not implemented; for the future task)

When a "duplicate" is detected, still append its `NormalizedImage` record to
`images` (with its already-existing destination path) instead of only
`continue`-ing past it. This makes `images` — and therefore every report
derived from it — always reflect the true, current contents of `processed/`,
while preserving the existing (correct) behavior of never re-copying or
re-verifying bytes that are already known-good on disk.

### How to work around it today

If you need an accurate manifest right now, clear the crop's processed output
before normalizing: `rm -rf datasets/<crop>` then re-run `plantdx normalize
--dataset <crop>`. This forces every file to be freshly placed and correctly
counted. (This is a manual workaround, not a fix — do this only if you
understand you are regenerating, not editing, `datasets/<crop>/`.)

---

## Reporting a new known issue

## 2. The July adapter weights are lost (resolved by retraining, 1 Nov 2026)

**Status:** resolved: both models were retrained and the new weights download with `setup/download_models.py`.
**Component:** `checkpoints/`. **Severity:** none once the weights are downloaded.

### What happened

The July runs wrote `adapters.safetensors` (see the last lines of `logs/<run>/train.log`), but the files were no
longer on the Mac when the project was handed over; only their SHA-256 checksums survive in
`reports/<run>/evaluation/system_info.json` (tomato `9d4b21f4…`, mango `c944488f…`). Without weights,
`plantdx infer`, `plantdx evaluate --stage inference` and the demo's model backend stop with "malformed adapter
checkpoint … missing adapters.safetensors", and
`tests/unit/evaluation/test_eval_checkpoint.py::test_real_frozen_tomato_checkpoint_resolves` fails.

### How it was resolved

Both models were retrained on the same Mac (Apple M4 Pro, 24 GB) from 31 Oct to 1 Nov 2026 with the same code
(commit `9f08d99`), the same training rows, config and split seed, and the same `mlx` / `mlx-vlm` versions.
`mlx_vlm.lora` takes no seed of its own (example order, LoRA initialisation and dropout are random), so the retrained
weights have new checksums and somewhat different scores (tomato 86.9%, mango 86.0%; July 93.7% and 82.0%) (tomato `a4872a5c…`, mango `f10dea4c…`). They are stored
in `github.com/iAakash1/TIH-IoT` (Git LFS) and listed in `setup/models_manifest.json`; from the repository root,
`python setup/download_models.py 09_plantdx` puts them in `checkpoints/<run>/`. Their training logs and evaluation
are in `retrained_2026-11/`. `scripts/verify_project.py` reports which weights are present (MISSING / July weights /
retrained).

---

## 3. The July training datasets list images by their absolute Mac paths

**Status:** known, by design. **Component:** `artifacts/training/<run>/dataset/*.jsonl`. **Severity:** low.

### What happens

Every row of the committed `train/validation/test.jsonl` points at
`/Users/aakashjawle/Desktop/experiments/datasets/<crop>/processed/...`, which exists only on the original Mac.

### Why it happens

`training/data/discovery.py` writes each image as an absolute path (`path.resolve()`), and `mlx_vlm.lora` reads
the paths exactly as written.

### Why it is not dangerous

`plantdx normalize` followed by `plantdx prepare-training` rewrites the files for the current machine with the same
rows in the same order (verified for both crops, `docs/REPRODUCTION.md`). The committed copies are the record of
what was trained on.

### Why it was intentionally deferred / suggested eventual fix

Changing the path format would change the frozen training artifacts. If relative paths are ever wanted, write
them relative to the dataset directory and have the runner resolve them before calling `mlx_vlm.lora`.

---

Add a new `##`-level section above, following the same structure: **Status /
Component / Severity**, **What happens**, **Why it happens**, **Why it is not
dangerous**, **Why it was intentionally deferred**, and (if known) a
**Suggested eventual fix**.
