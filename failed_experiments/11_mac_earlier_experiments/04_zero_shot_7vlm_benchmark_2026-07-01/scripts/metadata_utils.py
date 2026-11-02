"""
Metadata derivation for images and models.

Everything here is computed either directly from the file on disk (image
metadata) or parsed out of a model's own MLX repo identifier string
(model metadata) — nothing is looked up from external knowledge or a
memorized model card. If a value can't be safely read this way, it is
left as None rather than guessed at, for the same reason parser.py
refuses to guess at field values: an incorrect fabricated fact is worse
than an honest null in a research-grade dataset.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional, TypedDict

from PIL import Image


class ImageMetadata(TypedDict):
    filename: str
    relative_path: str
    class_folder: str
    width: Optional[int]
    height: Optional[int]
    file_size_bytes: int
    sha256: str


class ModelMetadata(TypedDict):
    model_id: str
    model_name: str
    model_family: Optional[str]
    parameters: Optional[str]
    quantization: Optional[str]
    mlx_model: bool


def compute_image_metadata(image_path: Path, dataset_root: Path, class_folder: str) -> ImageMetadata:
    """
    Compute metadata for one image file: identity (filename, relative
    path, class folder), dimensions, size, and a SHA-256 content hash for
    later duplicate detection. Computed once per image and reused across
    all models, since none of this depends on which model is being run.
    """
    image_path = Path(image_path)
    file_bytes = image_path.read_bytes()
    sha256 = hashlib.sha256(file_bytes).hexdigest()

    width: Optional[int]
    height: Optional[int]
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        # Corrupt/unreadable image: don't fail dataset discovery over it —
        # dimensions are left null and the file will still surface as a
        # generation failure later, with a clear error, when a model tries
        # to actually load it.
        width, height = None, None

    return {
        "filename": image_path.name,
        "relative_path": str(image_path.relative_to(dataset_root)),
        "class_folder": class_folder,
        "width": width,
        "height": height,
        "file_size_bytes": len(file_bytes),
        "sha256": sha256,
    }


# Family keywords are matched against the lowercased model id. Order
# doesn't matter since these are architecture names and never overlap.
_FAMILY_KEYWORDS = (
    ("qwen", "Qwen"),
    ("gemma", "Gemma"),
    ("internvl", "InternVL"),
    ("llama", "Llama"),
    ("molmo", "Molmo"),
    ("deepseek", "DeepSeek"),
)

# Parameter count: digits directly followed by "B"/"b", NOT followed by
# another letter (so "4bit" doesn't get misread as "4B", and "Qwen3" /
# "Llama-3.2" version numbers aren't mistaken for a parameter count since
# they aren't immediately followed by a B).
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")

# Quantization: digits followed by "bit" (optionally hyphenated), e.g.
# "4bit" or "4-bit".
_QUANT_RE = re.compile(r"(\d+)\s*-?bit", re.IGNORECASE)


def derive_model_metadata(model_id: str) -> ModelMetadata:
    """
    Derive lightweight model metadata directly from an MLX repo id such
    as "mlx-community/Qwen2.5-VL-7B-Instruct-4bit". Every field is parsed
    from the identifier itself:

    - model_family: matched via a small keyword list against the id.
    - parameters:   the "<N>B" token in the id, e.g. "7B" — left None if
                     the id doesn't spell out a parameter count (e.g.
                     "deepseek-vl2-small" has no explicit number, so
                     parameters is None rather than an invented figure).
    - quantization: the "<N>bit" token in the id, e.g. "4bit".

    No field here is ever pulled from memorized knowledge about a
    specific model's architecture — only from the string mlx-community
    itself published as the repo name.
    """
    basename = model_id.split("/")[-1]

    family = None
    lowered = basename.lower()
    for keyword, label in _FAMILY_KEYWORDS:
        if keyword in lowered:
            family = label
            break

    param_match = _PARAM_RE.search(basename)
    parameters = f"{param_match.group(1)}B" if param_match else None

    quant_match = _QUANT_RE.search(basename)
    quantization = f"{quant_match.group(1)}bit" if quant_match else None

    return {
        "model_id": model_id,
        "model_name": basename,
        "model_family": family,
        "parameters": parameters,
        "quantization": quantization,
        "mlx_model": "mlx" in model_id.lower(),
    }
