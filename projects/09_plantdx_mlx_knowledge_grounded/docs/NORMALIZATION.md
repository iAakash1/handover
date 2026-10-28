# Dataset Normalization Engine (Milestone 2.1)

Filesystem-only. Copies (or symlinks) the **tomato** and **mango** classes from the
immutable raw datasets into one canonical structure that every later stage
consumes. **The raw datasets are never modified.**

## Run it

From the repository root (`experiments/`):

```bash
plantdx normalize                    # normalize all configured datasets (copy)
plantdx normalize --dataset mango    # one dataset
plantdx normalize --mode link        # symlink instead of copy
python -m plantdx normalize          # equivalent, without the console script
```

## Output layout

Written under `paths.processed_dir` (default `datasets/`, gitignored):

```
datasets/
  normalization_report.json          # combined run report
  tomato/
    class_mapping.json               # raw folder -> canonical class + ignored folders
    manifest.json                    # every image: source, normalized path, checksum, split
    dataset_card.md                  # source, license, citation, URL, counts, checksum, limitations
    processed/<class>/<images>
  mango/
    ... (same)
```

## Configuration

Everything comes from the existing config system:
- **Raw locations** — `configs/paths.yaml` (`paths.datasets.<crop>.root`).
- **Output root** — `configs/paths.yaml` (`paths.processed_dir`).
- **Class map, mode, source metadata** — `configs/normalization.yaml`
  (`normalization.sources.<crop>.class_map`, `.mode`, `.license`, `.citation`, `.url`).

The `class_map` maps each raw folder (e.g. `Tomato___Early_blight`) to a canonical
class (`early_blight`). Folders not in the map (non-tomato PlantVillage crops) are
recorded in `class_mapping.json:ignored_folders` and **not** copied.

## Design notes

- **Structure-agnostic layout detection.** A *class directory* directly contains
  image files; its *split* is the parent folder name when that parent is not the
  root. This handles `root/class` and `root/train|val/class` with no hardcoded split names.
- **train/val are merged** into single class directories; each image's split is
  preserved in `manifest.json`.
- **Filenames preserved.** On a genuine collision (same name, different bytes across
  splits) the split is prefixed (`val__name.jpg`); byte-identical duplicates are skipped.
- **Verified.** Every placed file's SHA-256 is checked against its source. No Pillow
  is needed — verification is byte-level.
- **Deterministic.** Folders and files are processed in sorted order; the per-crop
  `dataset_checksum` is content-based and stable across runs.
- **No images are resized, recompressed, renamed (except on collision), or altered.**
