"""Company dataset preparation: audit, deduplicate, split, emit JSONL.

The raw dataset is read-only throughout: nothing here renames, rewrites, moves,
resizes, or deletes a source file. Every emitted row points at an absolute path
inside the company dataset root, and :func:`check_provenance` fails closed if a
path ever escapes it.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plantdx.company.classes import ClassEntry, CompanyDataset
from plantdx.company.config import CompanyTrainConfig
from plantdx.core.exceptions import PlantDxError
from plantdx.utils.hashing import sha256_hex, stable_json_hash
from plantdx.utils.io import read_json, write_json, write_jsonl

SPLITS = ("train", "validation", "test")

# Path fragments that must never reach a company training manifest. Guards the
# experiment against the other datasets that live in this repository.
FORBIDDEN_FRAGMENTS = (
    "plantvillage",
    "plant_village",
    "mango",
    "tomato/raw/plantvillage",
)


class DatasetError(PlantDxError):
    """The company dataset could not be prepared."""


@dataclass(frozen=True)
class ImageRecord:
    """One usable source image."""

    path: str
    relative_path: str
    class_id: str
    label: str
    sha256: str
    width: int
    height: int


@dataclass
class DatasetAudit:
    """What a scan of the raw dataset found."""

    root: str
    total_files: int = 0
    usable: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    per_class_raw: dict[str, int] = field(default_factory=dict)
    extensions: dict[str, int] = field(default_factory=dict)
    corrupt: list[dict[str, str]] = field(default_factory=list)
    skipped_extension: list[str] = field(default_factory=list)
    unmapped_dirs: list[str] = field(default_factory=list)
    empty_dirs: list[str] = field(default_factory=list)
    duplicate_groups: int = 0
    duplicate_extra_files: int = 0
    cross_class_duplicate_groups: list[list[str]] = field(default_factory=list)
    dropped_duplicates: int = 0
    dropped_ambiguous: int = 0
    image_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return asdict(self)

    @property
    def imbalance_ratio(self) -> float:
        """Largest class size divided by smallest; 0.0 when there are no classes."""
        if not self.per_class:
            return 0.0
        counts = [c for c in self.per_class.values() if c > 0]
        return round(max(counts) / min(counts), 2) if counts else 0.0


@dataclass(frozen=True)
class PreparedDataset:
    """The result of preparing one company dataset."""

    dataset_dir: Path
    audit: DatasetAudit
    counts: dict[str, int]
    per_split_per_class: dict[str, dict[str, int]]
    manifest_path: Path
    manifest_hash: str
    labels: tuple[str, ...]


def scan_dataset(
    dataset: CompanyDataset,
    *,
    verify: bool = True,
    max_per_class: int | None = None,
) -> tuple[list[ImageRecord], DatasetAudit]:
    """Walk the raw dataset and return usable records plus an audit.

    Args:
        dataset: The resolved company dataset profile.
        verify: Decode each image header and hash its bytes. Disable for a fast
            existence-only pass.
        max_per_class: Stop after this many usable images per class (smoke tests).
    """
    from PIL import Image

    audit = DatasetAudit(root=str(dataset.root))
    if not dataset.root.is_dir():
        raise DatasetError(f"company dataset root does not exist: {dataset.root}")

    audit.unmapped_dirs = dataset.unmapped_dirs()
    allowed = set(dataset.extensions)
    records: list[ImageRecord] = []
    sizes: Counter[str] = Counter()

    for entry in dataset.classes:
        class_dir = dataset.root / entry.raw
        if not class_dir.is_dir():
            audit.empty_dirs.append(entry.raw)
            audit.per_class_raw[entry.class_id] = 0
            audit.per_class[entry.class_id] = 0
            continue

        files = sorted(p for p in class_dir.rglob("*") if p.is_file())
        audit.total_files += len(files)
        audit.per_class_raw[entry.class_id] = len(files)
        kept = 0
        for file_path in files:
            audit.extensions[file_path.suffix] = audit.extensions.get(file_path.suffix, 0) + 1
            if file_path.suffix.lower() not in allowed:
                audit.skipped_extension.append(str(file_path.relative_to(dataset.root)))
                continue
            if max_per_class is not None and kept >= max_per_class:
                continue
            if not verify:
                records.append(_record(file_path, dataset.root, entry, "", 0, 0))
                kept += 1
                continue
            try:
                data = file_path.read_bytes()
                with Image.open(file_path) as image:
                    image.verify()
                with Image.open(file_path) as image:
                    width, height = image.size
            except Exception as exc:
                audit.corrupt.append(
                    {
                        "path": str(file_path.relative_to(dataset.root)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            digest = hashlib.sha256(data).hexdigest()
            sizes[f"{width}x{height}"] += 1
            records.append(_record(file_path, dataset.root, entry, digest, width, height))
            kept += 1

        if kept == 0:
            audit.empty_dirs.append(entry.raw)

    if verify:
        records = _deduplicate(records, audit)

    audit.usable = len(records)
    audit.per_class = dict(Counter(r.class_id for r in records))
    for entry in dataset.classes:
        audit.per_class.setdefault(entry.class_id, 0)
    audit.image_sizes = dict(sizes.most_common(15))
    return records, audit


def _record(
    path: Path, root: Path, entry: ClassEntry, digest: str, width: int, height: int
) -> ImageRecord:
    return ImageRecord(
        path=str(path.resolve()),
        relative_path=path.relative_to(root).as_posix(),
        class_id=entry.class_id,
        label=entry.label,
        sha256=digest,
        width=width,
        height=height,
    )


def _deduplicate(records: list[ImageRecord], audit: DatasetAudit) -> list[ImageRecord]:
    """Drop byte-identical copies; drop groups whose copies carry different labels.

    Keeping one representative per content hash is what guarantees no image can
    appear in two splits. A hash that spans two classes has a contradictory
    label, so the whole group is excluded and reported.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)

    kept: list[ImageRecord] = []
    for digest, members in groups.items():
        if len(members) > 1:
            audit.duplicate_groups += 1
            audit.duplicate_extra_files += len(members) - 1
        classes = {m.class_id for m in members}
        if len(classes) > 1:
            audit.cross_class_duplicate_groups.append(sorted({m.relative_path for m in members}))
            audit.dropped_ambiguous += len(members)
            continue
        members.sort(key=lambda r: r.relative_path)
        kept.append(members[0])
        audit.dropped_duplicates += len(members) - 1
        del digest
    kept.sort(key=lambda r: (r.class_id, r.relative_path))
    return kept


def split_records(
    records: list[ImageRecord],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    """Deterministic class-stratified split.

    Ordering inside a class is a pure function of ``(seed, image sha256)``, so
    the same dataset and seed always produce the same split regardless of
    filesystem order. Classes with at least three images always contribute at
    least one validation and one test image.
    """
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_id].append(record)

    out: dict[str, list[ImageRecord]] = {name: [] for name in SPLITS}
    for class_id in sorted(by_class):
        members = sorted(
            by_class[class_id],
            key=lambda r: (sha256_hex(seed, r.sha256 or r.relative_path), r.relative_path),
        )
        total = len(members)
        n_val = int(total * val_ratio)
        n_test = int(total * test_ratio)
        if total >= 3:
            n_val = max(n_val, 1) if val_ratio > 0 else 0
            n_test = max(n_test, 1) if test_ratio > 0 else 0
        n_train = total - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = total, 0, 0
        out["train"].extend(members[:n_train])
        out["validation"].extend(members[n_train : n_train + n_val])
        out["test"].extend(members[n_train + n_val :])
    return out


def build_rows(
    records: list[ImageRecord],
    instruction: str,
    *,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Turn image records into training rows: image, question, answer.

    Passing ``seed`` interleaves the classes in a deterministic order, so a
    truncated evaluation (``--limit``) still covers every class instead of only
    the first one.
    """
    ordered = records
    if seed is not None:
        ordered = sorted(
            records, key=lambda r: sha256_hex(seed, "order", r.sha256 or r.relative_path)
        )
    return [
        {
            "image": record.path,
            "question": instruction,
            "answer": record.label,
            "class_id": record.class_id,
            "sha256": record.sha256,
        }
        for record in ordered
    ]


def check_provenance(rows: list[dict[str, Any]], root: Path) -> list[str]:
    """Return one message per row whose image is not a clean company-dataset path.

    Fails closed on two conditions: the path is not inside the company dataset
    root, or it contains a fragment belonging to another dataset in this repo.
    """
    problems: list[str] = []
    resolved_root = root.resolve()
    for index, row in enumerate(rows):
        raw = str(row.get("image", ""))
        if not raw:
            problems.append(f"row {index}: empty image path")
            continue
        path = Path(raw)
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError:
            problems.append(f"row {index}: image outside the company dataset root: {raw}")
            continue
        lowered = raw.replace("\\", "/").lower()
        for fragment in FORBIDDEN_FRAGMENTS:
            if fragment in lowered:
                problems.append(f"row {index}: forbidden path fragment {fragment!r}: {raw}")
                break
    return problems


def prepare_dataset(
    cfg: CompanyTrainConfig,
    dataset: CompanyDataset,
    *,
    max_per_class: int | None = None,
    dataset_dir: str | Path | None = None,
) -> PreparedDataset:
    """Scan, split, and write the company training dataset.

    Raises:
        DatasetError: No usable images, or a row failed the provenance check.
    """
    target = Path(dataset_dir) if dataset_dir is not None else cfg.dataset_dir
    if (target / "manifest.json").is_file():
        manifest = validate_prepared(cfg, dataset, target)
        return PreparedDataset(
            target,
            DatasetAudit(**manifest["audit"]),
            manifest["split"]["counts"],
            manifest["split"]["per_class"],
            target / "manifest.json",
            stable_json_hash(manifest),
            dataset.labels,
        )
    if any((target / f"{split}.jsonl").exists() for split in SPLITS):
        raise DatasetError("Incomplete prepared dataset; refusing to overwrite split files")
    records, audit = scan_dataset(dataset, verify=True, max_per_class=max_per_class)
    if not records:
        raise DatasetError(f"no usable images found under {dataset.root}")

    splits = split_records(
        records,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.data.split_seed,
    )

    target = Path(dataset_dir) if dataset_dir is not None else cfg.dataset_dir
    target.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    per_split_per_class: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        rows = build_rows(splits[split], cfg.data.instruction, seed=cfg.data.split_seed)
        problems = check_provenance(rows, dataset.root)
        if problems:
            raise DatasetError(
                "company dataset provenance check failed:\n  - " + "\n  - ".join(problems[:10])
            )
        write_jsonl(target / f"{split}.jsonl", rows)
        counts[split] = len(rows)
        per_split_per_class[split] = dict(
            sorted(Counter(r.class_id for r in splits[split]).items())
        )

    overlap = _split_overlap(splits)
    if overlap:
        raise DatasetError(f"image appears in more than one split: {sorted(overlap)[:5]}")

    manifest = {
        "run_name": cfg.run_name,
        "dataset": {
            "name": dataset.name,
            "crop": dataset.crop,
            "root": str(dataset.root),
            "profile": str(dataset.profile_path),
        },
        "instruction": cfg.data.instruction,
        "labels": list(dataset.labels),
        "class_ids": list(dataset.class_ids),
        "class_map": [
            {"raw": c.raw, "class_id": c.class_id, "label": c.label} for c in dataset.classes
        ],
        "split": {
            "seed": cfg.data.split_seed,
            "train_ratio": cfg.data.train_ratio,
            "val_ratio": cfg.data.val_ratio,
            "test_ratio": cfg.data.test_ratio,
            "counts": counts,
            "per_class": per_split_per_class,
        },
        "audit": audit.to_dict(),
        "content_checksum": stable_json_hash(
            [{"p": r.relative_path, "h": r.sha256, "c": r.class_id} for r in records]
        ),
    }
    manifest_path = target / "manifest.json"
    write_json(manifest_path, manifest)

    return PreparedDataset(
        dataset_dir=target,
        audit=audit,
        counts=counts,
        per_split_per_class=per_split_per_class,
        manifest_path=manifest_path,
        manifest_hash=stable_json_hash(manifest),
        labels=dataset.labels,
    )


def _split_overlap(splits: dict[str, list[ImageRecord]]) -> set[str]:
    seen: dict[str, str] = {}
    overlap: set[str] = set()
    for split, records in splits.items():
        for record in records:
            key = record.sha256 or record.path
            if key in seen and seen[key] != split:
                overlap.add(record.relative_path)
            seen[key] = split
    return overlap


def load_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Read a prepared dataset's manifest.

    Raises:
        DatasetError: The manifest is missing or malformed.
    """
    path = Path(dataset_dir) / "manifest.json"
    if not path.is_file():
        raise DatasetError(f"dataset manifest not found: {path} (run `plantdx company prepare`)")
    data = read_json(path)
    if not isinstance(data, dict):
        raise DatasetError(f"malformed dataset manifest: {path}")
    return data


def split_path(dataset_dir: str | Path, split: str) -> Path:
    """Path to one split's JSONL file.

    Raises:
        DatasetError: Unknown split name.
    """
    if split not in SPLITS:
        raise DatasetError(f"unknown split {split!r}; expected one of {SPLITS}")
    return Path(dataset_dir) / f"{split}.jsonl"


def validate_prepared(cfg, dataset, target=None, *, verify_images=True):
    """Verify frozen rows, labels, content hashes and cross-split isolation."""
    import json

    target = Path(target) if target is not None else cfg.dataset_dir
    manifest = load_manifest(target)
    policy = manifest["split"]
    if (
        policy["seed"] != cfg.data.split_seed
        or any(
            policy[k] != getattr(cfg.data, k) for k in ("train_ratio", "val_ratio", "test_ratio")
        )
        or manifest["instruction"] != cfg.data.instruction
        or manifest["labels"] != list(dataset.labels)
    ):
        raise DatasetError("Prepared split/config mismatch; refusing to reshuffle existing data")
    seen = set()
    for split in SPLITS:
        rows = [
            json.loads(line)
            for line in split_path(target, split).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or len(rows) != policy["counts"][split]:
            raise DatasetError(f"Invalid {split} count")
        problems = check_provenance(rows, dataset.root)
        if problems:
            raise DatasetError(problems[0])
        counts = Counter()
        for row in rows:
            entry = dataset.by_class_id(row["class_id"])
            path = Path(row["image"])
            if (
                entry is None
                or row["answer"] != entry.label
                or path.parent.name != entry.raw
                or row["question"] != cfg.data.instruction
            ):
                raise DatasetError(f"Invalid class/prompt: {path}")
            if not path.is_file():
                raise DatasetError(f"Missing image: {path}")
            digest = row.get("sha256")
            if not digest or digest in seen:
                raise DatasetError(f"Duplicate/missing content hash in {split}: {path}")
            if verify_images and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise DatasetError(f"Image changed after preparation: {path}")
            seen.add(digest)
            counts[row["class_id"]] += 1
        if dict(counts) != policy["per_class"][split]:
            raise DatasetError(f"Class counts changed in {split}")
    return manifest
