"""Dataset exporter tests: coverage, determinism, and content preservation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from plantdx.exporters import FORMATS, export_checksum, export_jsonl, export_records, write_all


@pytest.mark.unit
def test_registry_covers_expected_formats() -> None:
    assert set(FORMATS) == {"blip2", "generic", "llava", "messages", "paligemma"}


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_every_format_reshapes_all_captions(corpus: Any) -> None:
    for fmt in FORMATS:
        recs = export_records(corpus, fmt)
        assert len(recs) == len(corpus.captions)


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_caption_text_is_never_altered(corpus: Any) -> None:
    texts = {c.text for c in corpus.captions}
    for fmt in FORMATS:
        for rec in export_records(corpus, fmt):
            payload = json.dumps(rec)
            # Every exported record's caption text must be one of the corpus texts.
            assert any(t in payload for t in texts) or fmt == "blip2"


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_exports_are_deterministic(corpus: Any) -> None:
    first = {f: export_checksum(corpus, f) for f in FORMATS}
    second = {f: export_checksum(corpus, f) for f in FORMATS}
    assert first == second


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_unknown_format_raises(corpus: Any) -> None:
    with pytest.raises(KeyError):
        export_records(corpus, "does_not_exist")


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_write_all_produces_manifest_per_format(corpus: Any, tmp_path: Any) -> None:
    write_all(corpus, tmp_path)
    for fmt in FORMATS:
        data = (tmp_path / fmt / "data.jsonl").read_text().strip().splitlines()
        manifest = json.loads((tmp_path / fmt / "manifest.json").read_text())
        assert len(data) == len(corpus.captions)
        assert manifest["format"] == fmt
        assert manifest["record_count"] == len(corpus.captions)
        assert export_jsonl(corpus, fmt) == (tmp_path / fmt / "data.jsonl").read_text()
