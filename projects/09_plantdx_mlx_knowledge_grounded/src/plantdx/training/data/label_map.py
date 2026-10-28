"""The one filesystem<->knowledge coupling: processed folder name -> disease_id.

Loaded from ``assets/metadata/label_map.json`` (authored, version-controlled).
Kept tiny and explicit so a domain expert can verify it by reading one file.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.core.exceptions import ConfigError
from plantdx.utils.io import read_json

_DEFAULT_ASSET = Path("assets/metadata/label_map.json")


def load_label_map(crop: str, *, asset_path: str | Path = _DEFAULT_ASSET) -> dict[str, str]:
    """Return ``{processed_class_folder: disease_id}`` for ``crop``.

    Raises :class:`ConfigError` if the asset is missing or the crop is absent —
    the training pipeline must never guess a label.
    """
    path = Path(asset_path)
    if not path.is_file():
        raise ConfigError(f"label map asset not found: {path}")
    data = read_json(path)
    crops = data.get("crops", {}) if isinstance(data, dict) else {}
    mapping = crops.get(crop)
    if not isinstance(mapping, dict) or not mapping:
        raise ConfigError(f"label map has no entries for crop {crop!r} in {path}")
    return {str(k): str(v) for k, v in mapping.items()}
