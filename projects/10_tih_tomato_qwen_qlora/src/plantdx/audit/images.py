"""Per-image inspection with Pillow (metadata only) and SHA-256 hashing.

Design choices required by the milestone brief:
  * Read each file's bytes once, hash them with SHA-256 (reproducible identity),
    and open them from memory so we never read the file from disk twice.
  * Read only header metadata (``size``/``mode``/``format``) — we do NOT decode
    the full image. ``Image.verify()`` provides a cheap integrity check.
  * A single bad image must never crash the audit: any failure is caught and
    recorded on the :class:`ImageRecord` (``ok=False`` + ``error``).

Average (perceptual) hashing is optional and off by default because it must
decode the pixels to resize them; see :func:`average_hash`.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from plantdx.audit.models import ImageRecord
from plantdx.utils.hashing import sha256_bytes


def average_hash(data: bytes, size: int) -> str:
    """Compute an average hash (aHash) of the image, as a hex string.

    aHash is the simplest reliable perceptual hash: convert to grayscale, resize
    to ``size x size``, then set each bit to 1 where the pixel is >= the mean.
    Two images that are visually the same but byte-different (rescaled or
    re-encoded) get the same aHash, which exact SHA-256 cannot detect. It is
    ~10 lines, deterministic, and Pillow-only. It requires decoding the pixels,
    which is why near-duplicate detection is opt-in.
    """
    with Image.open(io.BytesIO(data)) as image:
        small = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        pixels = list(small.getdata())
    average = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | (1 if pixel >= average else 0)
    hex_width = (size * size + 3) // 4
    return format(bits, f"0{hex_width}x")


def inspect_image(
    path: Path,
    dataset: str,
    class_name: str,
    root: Path,
    *,
    compute_ahash: bool,
    ahash_size: int,
) -> ImageRecord:
    """Inspect one image and return an :class:`ImageRecord` (never raises)."""
    relpath = str(path.relative_to(root))

    try:
        data = path.read_bytes()
    except OSError as exc:
        return ImageRecord(
            dataset,
            class_name,
            relpath,
            ok=False,
            file_size=0,
            sha256="",
            error=f"unreadable: {exc}",
        )

    sha = sha256_bytes(data)
    file_size = len(data)

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            mode, fmt = image.mode, image.format
        with Image.open(io.BytesIO(data)) as image:
            image.verify()  # integrity check without a full decode
    except Exception as exc:
        return ImageRecord(
            dataset,
            class_name,
            relpath,
            ok=False,
            file_size=file_size,
            sha256=sha,
            error=f"corrupt: {type(exc).__name__}: {exc}",
        )

    ahash: str | None = None
    if compute_ahash:
        try:
            ahash = average_hash(data, ahash_size)
        except Exception:
            ahash = None

    return ImageRecord(
        dataset,
        class_name,
        relpath,
        ok=True,
        file_size=file_size,
        sha256=sha,
        width=width,
        height=height,
        mode=mode,
        format=fmt,
        ahash=ahash,
    )
