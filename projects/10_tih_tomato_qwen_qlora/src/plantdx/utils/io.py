"""I/O helpers: JSON / JSONL / YAML read-write.

Thin, typed wrappers so the pipeline reads and writes artifacts consistently
(UTF-8, newline-delimited JSONL, atomic writes via temp-file rename).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (a directory) if missing and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> Any:
    """Read a YAML file into Python objects."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as pretty JSON, atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate records from a JSONL file (one JSON object per line)."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records to a JSONL file, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one record to a JSONL file (creating it if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
