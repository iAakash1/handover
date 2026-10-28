"""Active-adapter registry.

A pointer, not a copy: records which trained adapter the app should serve for a
crop. Existing adapters are never moved or overwritten to activate one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plantdx.company.classes import repo_root
from plantdx.core.exceptions import ConfigError
from plantdx.utils.io import read_json, write_json

REGISTRY_PATH = "artifacts/runtime/active_adapter.json"


@dataclass(frozen=True)
class ActiveAdapter:
    """One activated adapter and enough provenance to identify it."""

    crop: str
    run_name: str
    adapter_path: str
    base_model: str
    dataset: str
    backend: str
    classes: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    activated_at: str = ""

    def resolved_path(self, base_dir: str | Path | None = None) -> Path:
        """Absolute adapter directory."""
        base = Path(base_dir) if base_dir is not None else repo_root()
        candidate = Path(self.adapter_path).expanduser()
        return candidate if candidate.is_absolute() else base / candidate

    def exists(self, base_dir: str | Path | None = None) -> bool:
        """Whether the adapter directory is present on disk."""
        return self.resolved_path(base_dir).is_dir()

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        payload = asdict(self)
        payload["classes"] = list(self.classes)
        return payload


def registry_path(base_dir: str | Path | None = None) -> Path:
    """Absolute path to the registry file."""
    base = Path(base_dir) if base_dir is not None else repo_root()
    return base / REGISTRY_PATH


def read_registry(base_dir: str | Path | None = None) -> dict[str, ActiveAdapter]:
    """Read every activated adapter, keyed by crop. Missing file yields ``{}``."""
    path = registry_path(base_dir)
    if not path.is_file():
        return {}
    try:
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"active adapter registry is unreadable: {path} ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"active adapter registry must be a mapping: {path}")
    entries: dict[str, ActiveAdapter] = {}
    for crop, value in raw.items():
        if isinstance(value, dict):
            entries[str(crop)] = _from_dict(str(crop), value)
    return entries


def read_active_adapter(crop: str, base_dir: str | Path | None = None) -> ActiveAdapter | None:
    """Return the activated adapter for ``crop``, or ``None`` if unset."""
    return read_registry(base_dir).get(crop)


def write_active_adapter(entry: ActiveAdapter, base_dir: str | Path | None = None) -> Path:
    """Record ``entry`` as the active adapter for its crop and return the file path."""
    path = registry_path(base_dir)
    current = read_registry(base_dir)
    current[entry.crop] = entry
    write_json(path, {crop: value.to_dict() for crop, value in sorted(current.items())})
    return path


def clear_active_adapter(crop: str, base_dir: str | Path | None = None) -> bool:
    """Remove ``crop`` from the registry. Returns whether anything was removed."""
    current = read_registry(base_dir)
    if crop not in current:
        return False
    del current[crop]
    write_json(
        registry_path(base_dir), {c: value.to_dict() for c, value in sorted(current.items())}
    )
    return True


def _from_dict(crop: str, raw: dict[str, Any]) -> ActiveAdapter:
    metrics_raw = raw.get("metrics") or {}
    metrics: dict[str, float] = {}
    if isinstance(metrics_raw, dict):
        for key, value in metrics_raw.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    classes_raw = raw.get("classes") or []
    classes = tuple(str(c) for c in classes_raw) if isinstance(classes_raw, list) else ()
    return ActiveAdapter(
        crop=str(raw.get("crop") or crop),
        run_name=str(raw.get("run_name") or ""),
        adapter_path=str(raw.get("adapter_path") or ""),
        base_model=str(raw.get("base_model") or ""),
        dataset=str(raw.get("dataset") or ""),
        backend=str(raw.get("backend") or ""),
        classes=classes,
        metrics=metrics,
        activated_at=str(raw.get("activated_at") or ""),
    )
