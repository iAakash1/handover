# Dataset Audit Engine (Milestone 2)

A CPU-only engine that inspects every image in the configured datasets and writes
a reproducibility report. It never trains, never uses a GPU/MLX, and never decodes
full images — it reads header metadata and hashes bytes.

## Run it

From the repository root (`experiments/`):

```bash
plantdx audit                     # audit all configured datasets
plantdx audit --dataset mango     # audit one dataset (config key)
plantdx audit --reports-dir out   # override the output directory
python -m plantdx audit           # equivalent, without the console script
```

Configuration comes from `configs/`:
- **Datasets & reports directory** — `configs/paths.yaml` (`paths.datasets`, `paths.reports_dir`).
- **Audit tunables** — `configs/audit.yaml` (`workers`, `supported_extensions`,
  `near_duplicates`, `ahash_size`, `imbalance_warn_ratio`).

## What it reports

Written to `reports/` (gitignored):

| File | Contents |
|------|----------|
| `dataset_card.md` | Human-readable summary per dataset (counts, dimensions, imbalance, issues). |
| `<dataset>_summary.json` | Full machine-readable summary for one dataset. |
| `class_distribution.csv` | Per (dataset, class) image counts. |
| `image_statistics.csv` | One row per image: dimensions, mode, format, size, aspect ratio, SHA-256. |
| `duplicate_images.csv` | Exact-duplicate groups (by SHA-256). |
| `corrupt_images.csv` | Corrupt / unreadable images. |
| `near_duplicate_images.csv` | Near-duplicate groups (only when `near_duplicates: true`). |
| `audit_manifest.json` | Dataset version manifest: checksums, totals, settings, split status. |
| `audit.log` | Run log. |

## Design notes

- **Deterministic.** Images are processed in sorted order and all rows are sorted,
  so reports and checksums are identical across runs regardless of `workers`. Only
  the manifest's `generated_at` timestamp varies, and it is excluded from every checksum.
- **Metadata only.** Each file's bytes are read once (for SHA-256), opened from
  memory, and validated with `Image.verify()` — no full decode.
- **Robust.** A single bad image is recorded (`ok=false` + `error`) and the audit
  continues; failures are summarized in `corrupt_images.csv` and the manifest.
- **Structure-agnostic discovery.** A class is the immediate parent folder of each
  image, so flat (`root/Class/img`) and nested (`root/split/Class/img`) layouts both
  work. No disease names are hardcoded; the audit *reports* mismatches against the
  configured class count instead of assuming a structure.
- **Perceptual hashing (opt-in).** `near_duplicates` enables average-hash (aHash)
  grouping — ~10 deterministic lines, Pillow-only, O(n) by exact aHash bucket. It is
  off by default because computing aHash requires decoding pixels (slower).
- **Splitting is deferred.** The audit only inventories; the manifest records
  `splits.status = "not_performed"`.
