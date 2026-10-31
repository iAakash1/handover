"""Tests for setup/verify_repository.py.  Run: python -m pytest setup/tests"""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "verify_repository.py"
_spec = importlib.util.spec_from_file_location("verify_repository", _SCRIPT)
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def test_markdown_links_ignores_urls_anchors_and_code(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("[x](b.md) [y](https://x.org) [z](#top) [w](mailto:a@b.c)\n```\n[no](c.md)\n```\n[s](d%20e.md#h)\n",
                  encoding="utf-8")
    assert vr.markdown_links(md) == ["b.md", "d e.md"]


def test_broken_links_resolves_relative_to_the_file(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ok.md").write_text("", encoding="utf-8")
    md = tmp_path / "docs" / "index.md"
    md.write_text("[ok](ok.md) [up](../docs/ok.md) [bad](missing.md)", encoding="utf-8")
    assert vr.broken_links([md], root=tmp_path) == ["docs/index.md -> missing.md"]


def test_manifest_problems_flags_bad_entries():
    good = {"path": "a", "source": "s", "bytes": 1, "sha256": "0" * 64}
    manifest = {"files": [good, dict(good), {"path": "b", "source": "s", "bytes": 1, "sha256": "xyz"}, {"path": "c"}]}
    problems = vr.manifest_problems(manifest)
    assert "a: listed twice" in problems
    assert "b: sha256 is not 64 hex characters" in problems
    assert any(p.startswith("entry 3: missing source") for p in problems)


def test_the_real_manifest_is_well_formed():
    import json
    manifest = json.loads((_SCRIPT.parent / "models_manifest.json").read_text(encoding="utf-8"))
    assert vr.manifest_problems(manifest) == []
