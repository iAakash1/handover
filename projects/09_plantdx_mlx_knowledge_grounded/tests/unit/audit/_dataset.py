"""Builds a small synthetic dataset for audit tests.

Structure created under ``root``::

    A/solid1.png       gradient image (PNG)
    A/solid1_dup.png   byte-for-byte copy of solid1.png  -> EXACT duplicate
    A/solid1_bmp.bmp   same pixels as solid1, BMP encoding -> NEAR duplicate
    A/other.png        a different gradient
    B/b1.jpg           a valid JPEG
    B/broken.jpg       garbage bytes with a .jpg extension -> CORRUPT
    B/notes.txt        a text file -> UNSUPPORTED
    Empty/             an empty directory -> EMPTY FOLDER
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image


def _gradient(size: tuple[int, int], fn: Callable[[int, int], int]) -> Image.Image:
    image = Image.new("L", size)
    pixels = image.load()
    width, height = size
    for y in range(height):
        for x in range(width):
            pixels[x, y] = fn(x, y) % 256
    return image


def build_sample_dataset(root: Path) -> dict[str, Any]:
    """Create the synthetic dataset under ``root`` and return expected counts."""
    root = Path(root)
    class_a, class_b, empty = root / "A", root / "B", root / "Empty"
    for directory in (class_a, class_b, empty):
        directory.mkdir(parents=True, exist_ok=True)

    gradient = _gradient((16, 16), lambda x, y: x * 13 + y * 7)
    gradient.save(class_a / "solid1.png")
    shutil.copyfile(class_a / "solid1.png", class_a / "solid1_dup.png")  # exact duplicate
    gradient.save(class_a / "solid1_bmp.bmp")  # same pixels, different bytes -> near duplicate
    _gradient((16, 16), lambda x, y: 255 - (x * 13 + y * 7)).save(class_a / "other.png")

    Image.new("RGB", (20, 10), (0, 128, 255)).save(class_b / "b1.jpg")
    (class_b / "broken.jpg").write_bytes(b"this is not a valid image")
    (class_b / "notes.txt").write_text("not an image", encoding="utf-8")

    return {
        "root": root,
        "num_images": 6,  # 4 in A + 2 in B (broken.jpg counts as an image, just corrupt)
        "num_ok": 5,
        "num_corrupt": 1,
        "classes": ["A", "B"],
        "exact_groups": 1,
        "near_groups": 1,
        "unsupported": 1,
        "empty_dir": "Empty",
    }
