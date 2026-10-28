"""
scripts/extract_ground_truth.py

Infers a ground-truth disease (and best-effort plant species) label for
every image in images/ purely from its filename — no class subfolders are
assumed, since the benchmark expects a single flat images/ directory.

PlantVillage-style filenames embed a short research/site code after the
image UUID, e.g.:
    0051e5e8-...-d1c4___RS_LB 4640.JPG          -> Late Blight
    0022d6b7-...-ae9a___JR_B.Spot 8964.JPG      -> Bacterial Spot
    00694db7-...-b4da___Crnl_L.Mold 6923.JPG    -> Leaf Mold

KNOWN LIMITATION (by design, not an oversight):
Not every code is disease-indicative. Some PlantVillage filenames use pure
site/photographer codes (e.g. "PSU_CG", "Com.G_FL") that don't encode the
disease at all. Rather than guess and silently corrupt the benchmark's
ground truth, unmatched images are labeled "Unknown" and excluded from
accuracy / calibration scoring downstream (they still get LLM-judge
symptom-quality scoring where possible). If you recognize more codes from
your own dataset, add them to SINGLE_TOKEN_MAP / BIGRAM_MAP below — that's
the intended extension point.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from scripts import utils

# Single filename token (split on any non-alphanumeric char) -> disease.
SINGLE_TOKEN_MAP = {
    "lb":          "Late Blight",
    "hl":          "Healthy",
    "healthy":     "Healthy",
    "ylcv":        "Yellow Leaf Curl Virus",
    "tgs":         "Target Spot",
    "spm":         "Spider Mites",
    "st":          "Septoria Leaf Spot",
    "septoria":    "Septoria Leaf Spot",
    "mosaic":      "Mosaic Virus",
    "variegation": "Leaf Variegation",
    "redding":     "Leaf Redding",
    "curl":        "Yellow Leaf Curl Virus",
}

# Two ADJACENT tokens concatenated (e.g. "Early" + "B" -> "earlyb") -> disease.
BIGRAM_MAP = {
    "lateb":           "Late Blight",
    "earlyb":          "Early Blight",
    "erlyb":           "Early Blight",
    "bactsp":          "Bacterial Spot",
    "bspot":           "Bacterial Spot",
    "lmold":           "Leaf Mold",
    "leafvariegation": "Leaf Variegation",
    "leafredding":     "Leaf Redding",
    "curlvirus":       "Yellow Leaf Curl Virus",
    "mosaicvirus":     "Mosaic Virus",
}

SPECIES_KEYWORDS = [
    "tomato", "potato", "bell pepper", "pepper", "cotton", "sugarcane",
    "maize", "corn", "cassava", "grape", "apple", "peach", "strawberry",
    "cherry", "orange", "squash", "soybean", "blueberry", "raspberry",
]


def _tokenize(stem: str):
    return [t for t in re.split(r"[^a-zA-Z0-9]+", stem) if t]


def infer_disease(filename: str):
    stem = Path(filename).stem
    tokens = [t.lower() for t in _tokenize(stem)]
    flat = "".join(tokens)

    # 1. Exact single-token match.
    for tok in tokens:
        if tok in SINGLE_TOKEN_MAP:
            return SINGLE_TOKEN_MAP[tok], f"token:{tok}"

    # 2. Adjacent bigram match.
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + tokens[i + 1]
        if bigram in BIGRAM_MAP:
            return BIGRAM_MAP[bigram], f"bigram:{tokens[i]}+{tokens[i+1]}"

    # 3. Literal substring fallback (covers codes typed without separators,
    #    e.g. "leafredding" or "curlvirus" appearing directly in a filename).
    for key, disease in BIGRAM_MAP.items():
        if key in flat:
            return disease, f"substring:{key}"
    for key, disease in SINGLE_TOKEN_MAP.items():
        if len(key) >= 4 and key in flat:  # skip short keys to avoid false positives
            return disease, f"substring:{key}"

    return "Unknown", "no_match"


def infer_species(filename: str):
    low = filename.lower().replace("_", " ").replace(".", " ")
    for sp in SPECIES_KEYWORDS:
        if sp in low:
            return sp.title()
    return "Unknown"


def main():
    config.ensure_directories()
    images = utils.list_images(config.IMAGES_DIR)
    if not images:
        utils.log(f"No images found in {config.IMAGES_DIR}. "
                   f"Copy your crop images there and re-run.")
        return

    ground_truth = {}
    unknown_count = 0
    for img_path in images:
        disease, matched_on = infer_disease(img_path.name)
        species = infer_species(img_path.name)
        if disease == "Unknown":
            unknown_count += 1
        ground_truth[img_path.name] = {
            "ground_truth_disease": disease,
            "ground_truth_species": species,
            "matched_on": matched_on,
        }

    out_path = config.JSON_RESULTS_DIR / "ground_truth.json"
    utils.save_json(ground_truth, out_path)

    utils.log(f"Processed images : {len(images)}")
    utils.log(f"Resolved labels  : {len(images) - unknown_count}")
    pct_unknown = unknown_count / len(images) * 100
    utils.log(f"Unknown labels   : {unknown_count} ({pct_unknown:.1f}%) "
              f"— see module docstring for why some filenames can't be resolved")
    utils.log(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
