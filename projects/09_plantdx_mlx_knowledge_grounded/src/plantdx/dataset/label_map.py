"""Folder → disease_id resolver (doc 04 §2).

The only place dataset folder strings are coupled to the pipeline. Loads
``assets/metadata/label_map.json`` and reconciles it against the real
directories under the dataset roots; an unmapped folder is a hard error.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.core.types import DiseaseLabel


class LabelResolver:
    """Resolves a dataset folder name to a :class:`DiseaseLabel`.

    Args:
        label_map_path: Path to ``label_map.json``.
    """

    def __init__(self, label_map_path: str | Path) -> None:
        """Initialize the resolver with the path to label_map.json."""
        self.label_map_path = Path(label_map_path)

    def load(self) -> None:
        """Parse the label map."""
        raise NotImplementedError("Milestone 4: label-map loading")

    def resolve(self, dataset: str, folder_name: str) -> DiseaseLabel:
        """Return the label for a folder.

        Raises:
            plantdx.core.exceptions.PlantDxError: If the folder is unmapped.
        """
        raise NotImplementedError("Milestone 4: label resolution")

    def reconcile(self, dataset_roots: dict[str, str | Path]) -> None:
        """Verify every on-disk folder maps to exactly one disease_id (doc 04 §2)."""
        raise NotImplementedError("Milestone 4: label-map reconciliation")
