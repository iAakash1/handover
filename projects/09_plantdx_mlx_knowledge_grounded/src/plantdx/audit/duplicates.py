"""Duplicate and near-duplicate detection.

Exact duplicates use SHA-256 (deterministic, reliable, already computed during
inspection). Near-duplicates group images by identical average hash (aHash) that
have *different* SHA-256 values — i.e. visually the same but byte-different
(rescaled or re-encoded copies). Both are O(n) grouping by hash; there is no
O(n^2) pairwise comparison.
"""

from __future__ import annotations

from collections import defaultdict

from plantdx.audit.models import ImageRecord


def exact_duplicate_groups(records: list[ImageRecord]) -> list[tuple[str, list[str]]]:
    """Group readable images by SHA-256; return ``(sha256, [relpaths])`` for size > 1."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.ok:
            by_hash[record.sha256].append(record.relpath)
    groups = [(sha, sorted(paths)) for sha, paths in by_hash.items() if len(paths) > 1]
    return sorted(groups)


def near_duplicate_groups(records: list[ImageRecord]) -> list[tuple[str, list[str]]]:
    """Group readable images by identical aHash with differing SHA-256.

    Returns ``(ahash, [relpaths])`` for groups that are visually identical but
    not byte-identical. Requires that aHash was computed (near-dup detection on).
    """
    by_ahash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.ok and record.ahash:
            by_ahash[record.ahash].append(record)
    groups = []
    for ahash, group in by_ahash.items():
        if len(group) > 1 and len({r.sha256 for r in group}) > 1:
            groups.append((ahash, sorted(r.relpath for r in group)))
    return sorted(groups)
