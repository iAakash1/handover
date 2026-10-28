"""Builders for synthetic raw datasets used by normalization tests.

The normalization engine never decodes images (it only hashes bytes and copies),
so these fixtures use tiny byte blobs with image extensions — no Pillow needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build_plantvillage(root: Path) -> dict[str, Any]:
    """A nested PlantVillage-like tree (train/val, two tomato classes + one other crop)."""
    root = Path(root)
    train = root / "train"
    val = root / "val"

    _write(train / "Tomato___Early_blight" / "a.jpg", b"AAAA")
    _write(train / "Tomato___Early_blight" / "b.jpg", b"BBBB")
    _write(train / "Tomato___Early_blight" / "same.jpg", b"SAME")
    _write(train / "Tomato___Early_blight" / "dup.jpg", b"DUPX")
    _write(train / "Tomato___healthy" / "h.jpg", b"HHHH")
    _write(train / "Corn_(maize)___healthy" / "c.jpg", b"CORN")  # non-tomato -> ignored

    _write(val / "Tomato___Early_blight" / "same.jpg", b"SAME")  # identical -> duplicate, skipped
    _write(val / "Tomato___Early_blight" / "dup.jpg", b"DUPY")  # collision -> disambiguated
    _write(val / "Tomato___Early_blight" / "v.jpg", b"VVVV")
    _write(val / "Tomato___healthy" / "h2.jpg", b"HH2")

    class_map = {
        "Tomato___Early_blight": "early_blight",
        "Tomato___healthy": "healthy",
    }
    return {
        "root": root,
        "class_map": class_map,
        "dataset": "PlantVillage",
        "layout": "nested (split/class)",
        "image_count": 8,  # early_blight: 6 (same.jpg dup skipped), healthy: 2
        "class_counts": {"early_blight": 6, "healthy": 2},
        "ignored_count": 1,  # Corn_(maize)___healthy
        "duplicates_skipped": 1,  # val/.../same.jpg
        "disambiguated": 1,  # val__dup.jpg
    }


def build_mango(root: Path) -> dict[str, Any]:
    """A flat MangoLeafBD-like tree (class folders directly under root, incl. a space)."""
    root = Path(root)
    _write(root / "Anthracnose" / "x.jpg", b"ANTH")
    _write(root / "Bacterial Canker" / "b.jpg", b"BCAN")
    _write(root / "Healthy" / "y.jpg", b"HLTY")

    class_map = {
        "Anthracnose": "anthracnose",
        "Bacterial Canker": "bacterial_canker",
        "Healthy": "healthy",
    }
    return {
        "root": root,
        "class_map": class_map,
        "dataset": "MangoLeafBD",
        "layout": "flat (class)",
        "image_count": 3,
        "class_counts": {"anthracnose": 1, "bacterial_canker": 1, "healthy": 1},
    }
