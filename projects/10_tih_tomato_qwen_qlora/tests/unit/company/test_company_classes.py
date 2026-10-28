"""Company class mapping: resolution, fail-closed behavior, unmapped folders."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plantdx.company.classes import load_company_dataset
from plantdx.core.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_PROFILE = REPO_ROOT / "configs" / "company_tomato_classes.yaml"


@pytest.mark.unit
def test_profile_loads_and_maps_every_class(company_dataset) -> None:
    assert company_dataset.name == "Test_Tomato"
    assert company_dataset.crop == "tomato"
    assert company_dataset.class_ids == ("aphid", "late_blight", "healthy_leaf")
    assert company_dataset.labels == ("Aphid", "Late Blight", "Healthy Leaf")


@pytest.mark.unit
def test_raw_folder_maps_to_canonical_class(company_dataset) -> None:
    entry = company_dataset.by_raw("Late_Blight")
    assert entry is not None
    assert entry.class_id == "late_blight"
    assert entry.label == "Late Blight"
    assert set(entry.__dataclass_fields__) == {"raw", "class_id", "label"}


@pytest.mark.unit
def test_unknown_raw_class_returns_none_not_a_guess(company_dataset) -> None:
    assert company_dataset.by_raw("Septoria_Leaf_Spot") is None
    assert company_dataset.by_class_id("septoria_leaf_spot") is None
    assert company_dataset.by_label("Septoria Leaf Spot") is None


@pytest.mark.unit
def test_unmapped_folder_is_reported_not_absorbed(
    company_dataset,
    raw_dataset: Path,
) -> None:
    (raw_dataset / "Mystery_Class").mkdir()
    assert company_dataset.unmapped_dirs() == ["Mystery_Class"]
    assert company_dataset.by_raw("Mystery_Class") is None


@pytest.mark.unit
def test_duplicate_class_entry_fails_closed(tmp_path: Path, raw_dataset: Path) -> None:
    payload = {
        "dataset": {"name": "x", "crop": "tomato", "root": str(raw_dataset)},
        "classes": [
            {"raw": "Aphid", "class_id": "aphid", "label": "Aphid"},
            {"raw": "Aphid", "class_id": "aphid2", "label": "Aphid 2"},
        ],
    }
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate class entry"):
        load_company_dataset(path)


@pytest.mark.unit
def test_missing_required_field_fails_closed(tmp_path: Path, raw_dataset: Path) -> None:
    payload = {
        "dataset": {"name": "x", "crop": "tomato", "root": str(raw_dataset)},
        "classes": [{"raw": "Aphid", "class_id": "aphid"}],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="needs `raw`, `class_id` and `label`"):
        load_company_dataset(path)


@pytest.mark.unit
def test_unresolvable_root_raises_and_lists_what_it_tried(tmp_path: Path) -> None:
    payload = {
        "dataset": {
            "name": "x",
            "crop": "tomato",
            "root": None,
            "candidate_roots": ["nowhere", "also_nowhere"],
        },
        "classes": [{"raw": "Aphid", "class_id": "aphid", "label": "Aphid"}],
    }
    path = tmp_path / "noroot.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="could not locate the company dataset root"):
        load_company_dataset(path)


@pytest.mark.unit
def test_root_override_wins(profile_path: Path, raw_dataset: Path, tmp_path: Path) -> None:
    other = tmp_path / "Elsewhere"
    (other / "Aphid").mkdir(parents=True)
    dataset = load_company_dataset(profile_path, root_override=other)
    assert dataset.root == other.resolve()


@pytest.mark.unit
def test_extensions_are_normalized_lowercase_with_dot(company_dataset) -> None:
    assert all(ext.startswith(".") and ext.islower() for ext in company_dataset.extensions)


@pytest.mark.unit
def test_shipped_profile_has_only_class_mapping() -> None:
    dataset = load_company_dataset(REAL_PROFILE)
    assert len(dataset.classes) == 11
    for entry in dataset.classes:
        assert set(entry.__dataclass_fields__) == {"raw", "class_id", "label"}


@pytest.mark.unit
def test_shipped_profile_class_ids_are_unique_and_snake_case() -> None:
    dataset = load_company_dataset(REAL_PROFILE)
    ids = list(dataset.class_ids)
    assert len(ids) == len(set(ids))
    assert all(i == i.lower() and " " not in i for i in ids)
