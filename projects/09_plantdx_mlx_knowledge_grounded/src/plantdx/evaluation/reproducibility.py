"""Reproducibility manifest: everything needed to reproduce this exact run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import DerivationError
from plantdx.evaluation.checkpoint import adapter_weights_path, resolve_adapter_dir
from plantdx.evaluation.latency import capture_system_info
from plantdx.utils.hashing import sha256_bytes


@dataclass(frozen=True)
class ReproducibilityManifest:
    """Everything needed to reproduce this exact evaluation run."""

    model_path: str
    adapter_path: str
    adapter_checksum: str | None
    dataset_dir: str
    split: str
    corpus_checksum: str | None
    ontology_checksum: str | None
    vocabulary_checksum: str | None
    seed: int
    system_info: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-writable dict."""
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "adapter_checksum": self.adapter_checksum,
            "dataset_dir": self.dataset_dir,
            "split": self.split,
            "corpus_checksum": self.corpus_checksum,
            "ontology_checksum": self.ontology_checksum,
            "vocabulary_checksum": self.vocabulary_checksum,
            "seed": self.seed,
            "system_info": self.system_info,
        }


def build_reproducibility_manifest(
    *,
    model_path: str,
    adapter_path: str,
    dataset_dir: str,
    split: str,
    seed: int,
    repo_root: str | Path = ".",
) -> ReproducibilityManifest:
    """Assemble the full reproducibility manifest from frozen artifacts."""
    root = Path(repo_root)
    return ReproducibilityManifest(
        model_path=model_path,
        adapter_path=adapter_path,
        adapter_checksum=_adapter_checksum(adapter_path),
        dataset_dir=dataset_dir,
        split=split,
        corpus_checksum=_dataset_manifest_field(Path(dataset_dir), "corpus_checksum"),
        ontology_checksum=_read_checksum_file(root / "artifacts/ontology/ontology_checksum.txt"),
        vocabulary_checksum=_read_checksum_file(root / "artifacts/vocabulary/checksum.txt"),
        seed=seed,
        system_info=capture_system_info(repo_root=root),
    )


def _adapter_checksum(adapter_path: str) -> str | None:
    """Checksum the adapter's actual weights file.

    Gracefully degrades to ``None`` if the checkpoint can't be resolved from
    here (e.g. the analyze stage running on a different machine than the
    checkpoint lives on) -- unlike stage 1's load, a missing checksum here
    must not fail the run.
    """
    try:
        weights_path = adapter_weights_path(resolve_adapter_dir(adapter_path))
    except DerivationError:
        return None
    return _file_checksum(weights_path)


def _file_checksum(path: Path) -> str | None:
    if not path.is_file():
        return None
    return f"sha256:{sha256_bytes(path.read_bytes())}"


def _dataset_manifest_field(dataset_dir: Path, field: str) -> str | None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = data.get(field)
    return str(value) if value is not None else None


def _read_checksum_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()
