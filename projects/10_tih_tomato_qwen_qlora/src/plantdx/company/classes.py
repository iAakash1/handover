"""Company dataset profile: raw folder names to training labels.

Loads ``configs/company_tomato_classes.yaml`` and resolves the immutable raw
dataset root. A raw folder that is absent from the mapping is reported as
unmapped and excluded; it never receives a label by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import ConfigError
from plantdx.utils.io import read_yaml

DEFAULT_PROFILE = "configs/company_tomato_classes.yaml"


@dataclass(frozen=True)
class ClassEntry:
    """One company class: its raw folder, class id, and display label."""

    raw: str
    class_id: str
    label: str


@dataclass(frozen=True)
class CompanyDataset:
    """The resolved company dataset profile."""

    name: str
    crop: str
    root: Path
    extensions: tuple[str, ...]
    classes: tuple[ClassEntry, ...]
    profile_path: Path

    @property
    def class_ids(self) -> tuple[str, ...]:
        """Canonical class ids in profile order."""
        return tuple(c.class_id for c in self.classes)

    @property
    def labels(self) -> tuple[str, ...]:
        """Training labels in profile order."""
        return tuple(c.label for c in self.classes)

    def by_raw(self, raw: str) -> ClassEntry | None:
        """Return the entry for a raw folder name, or ``None`` if unmapped."""
        for entry in self.classes:
            if entry.raw == raw:
                return entry
        return None

    def by_class_id(self, class_id: str) -> ClassEntry | None:
        """Return the entry for a canonical class id, or ``None``."""
        for entry in self.classes:
            if entry.class_id == class_id:
                return entry
        return None

    def by_label(self, label: str) -> ClassEntry | None:
        """Return the entry whose training label matches ``label`` exactly."""
        for entry in self.classes:
            if entry.label == label:
                return entry
        return None

    def raw_dirs(self) -> list[Path]:
        """Existing raw class directories, in profile order."""
        return [self.root / c.raw for c in self.classes if (self.root / c.raw).is_dir()]

    def unmapped_dirs(self) -> list[str]:
        """Directory names under the dataset root that the profile does not map."""
        if not self.root.is_dir():
            return []
        mapped = {c.raw for c in self.classes}
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name not in mapped)


def repo_root() -> Path:
    """The repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[3]


def load_company_dataset(
    profile_path: str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
    root_override: str | Path | None = None,
) -> CompanyDataset:
    """Load the company dataset profile and resolve its raw root.

    Args:
        profile_path: Profile YAML; defaults to ``configs/company_tomato_classes.yaml``
            under ``base_dir``.
        base_dir: Repository root used to resolve relative paths. Defaults to the
            repository this package lives in, so the caller's CWD does not matter.
        root_override: Explicit dataset root, overriding the profile.

    Raises:
        ConfigError: The profile is missing, malformed, or the dataset root
            cannot be resolved.
    """
    base = Path(base_dir).resolve() if base_dir is not None else repo_root()
    path = Path(profile_path) if profile_path is not None else base / DEFAULT_PROFILE
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise ConfigError(f"company dataset profile not found: {path}")

    raw = read_yaml(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    meta = raw.get("dataset")
    if not isinstance(meta, dict):
        raise ConfigError(f"{path}: missing `dataset:` section")

    entries = _parse_classes(raw.get("classes"), path)
    extensions = _parse_extensions(meta.get("extensions"), path)
    root = _resolve_root(meta, entries, base, root_override, path)

    return CompanyDataset(
        name=str(meta.get("name") or "company_tomato"),
        crop=str(meta.get("crop") or "tomato"),
        root=root,
        extensions=extensions,
        classes=entries,
        profile_path=path,
    )


def _parse_classes(value: object, path: Path) -> tuple[ClassEntry, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}: `classes:` must be a non-empty list")
    entries: list[ClassEntry] = []
    seen_raw: set[str] = set()
    seen_id: set[str] = set()
    seen_label: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: classes[{index}] must be a mapping")
        raw_name = str(item.get("raw") or "").strip()
        class_id = str(item.get("class_id") or "").strip()
        label = str(item.get("label") or "").strip()
        if not raw_name or not class_id or not label:
            raise ConfigError(f"{path}: classes[{index}] needs `raw`, `class_id` and `label`")
        for name, bucket in ((raw_name, seen_raw), (class_id, seen_id), (label, seen_label)):
            if name in bucket:
                raise ConfigError(f"{path}: duplicate class entry {name!r}")
            bucket.add(name)
        entries.append(
            ClassEntry(
                raw=raw_name,
                class_id=class_id,
                label=label,
            )
        )
    return tuple(entries)


def _parse_extensions(value: object, path: Path) -> tuple[str, ...]:
    if value is None:
        return (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}: `dataset.extensions` must be a non-empty list")
    out: list[str] = []
    for item in value:
        ext = str(item).strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        out.append(ext)
    return tuple(sorted(set(out)))


def _resolve_root(
    meta: dict[str, object],
    entries: tuple[ClassEntry, ...],
    base: Path,
    root_override: str | Path | None,
    path: Path,
) -> Path:
    tried: list[str] = []
    if root_override is not None:
        candidate = _absolute(root_override, base)
        if candidate.is_dir():
            return candidate
        raise ConfigError(f"dataset root does not exist: {candidate}")

    configured = meta.get("root")
    candidates: list[Path] = []
    if configured:
        candidates.append(_absolute(str(configured), base))
    raw_candidates = meta.get("candidate_roots")
    if isinstance(raw_candidates, list):
        candidates.extend(_absolute(str(item), base) for item in raw_candidates)

    mapped = {c.raw for c in entries}
    for candidate in candidates:
        tried.append(str(candidate))
        if not candidate.is_dir():
            continue
        if any((candidate / name).is_dir() for name in mapped):
            return candidate

    raise ConfigError(
        f"{path}: could not locate the company dataset root. Tried: "
        + ("; ".join(tried) if tried else "(no candidates configured)")
        + ". Set `dataset.root` in the profile or pass an explicit root."
    )


def _absolute(value: str | Path, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return candidate.resolve()
    except OSError:  # pragma: no cover - unreachable on supported platforms
        return candidate
