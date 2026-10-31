"""Repository-wide consistency check: links, JSON/YAML, Python syntax and the model manifest.

Needs only Python 3.9+ (PyYAML is used if installed). Nothing is trained or downloaded.

    python setup/verify_repository.py                    # repository-wide checks
    python setup/verify_repository.py --with-projects    # also run projects/09's own verify_project.py

Each check prints PASS, FAIL or NOT RUN; the exit code is 1 only if something FAILs. `archive/` holds unedited
originals whose links point into the old TIH-IoT layout on purpose, so its links are not checked.
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
SKIP_LINKS = ("archive/",)                       # frozen originals, see the docstring
SKIP_DATA = ("projects/09_plantdx_mlx_knowledge_grounded/tomato/", "projects/09_plantdx_mlx_knowledge_grounded/mango/")
# Original records kept byte-for-byte although they are not valid JSON; each is explained in the README named here.
KNOWN_INVALID_JSON = {
    "failed_experiments/01_florence2_finetuning/captions.json": "failed_experiments/01_florence2_finetuning/README.md",
}
results = []


def report(status, check, detail=""):
    results.append(status)
    print(f"  {status:8} {check}" + (f" - {detail}" if detail else ""))


def tracked(*patterns):
    """Files git tracks under the repository root (falls back to a directory walk without git)."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", *patterns],
                             capture_output=True, check=True).stdout.decode("utf-8")
        files = [ROOT / f for f in out.split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        files = [p for pat in patterns for p in ROOT.rglob(pat.lstrip("*/"))]
    return [f for f in files if not f.relative_to(ROOT).as_posix().startswith(SKIP_DATA)]


def markdown_links(md_file):
    """Relative link targets in a Markdown file; URLs, anchors and fenced code are skipped."""
    text = re.sub(r"```.*?```", "", md_file.read_text(encoding="utf-8", errors="replace"), flags=re.S)
    targets = []
    for m in LINK_RE.finditer(text):
        target = m.group(1)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue
        path = urllib.parse.unquote(target.split("#")[0].split("?")[0])
        if path:
            targets.append(path)
    return targets


def broken_links(md_files, root=ROOT):
    broken = []
    for md in md_files:
        for target in markdown_links(md):
            base = root if target.startswith("/") else md.parent
            if not (base / target.lstrip("/")).exists():
                broken.append(f"{md.relative_to(root).as_posix()} -> {target}")
    return broken


def manifest_problems(manifest):
    """Structural problems of setup/models_manifest.json entries (no file access)."""
    problems, seen = [], set()
    for i, entry in enumerate(manifest.get("files", [])):
        missing = [k for k in ("path", "source", "bytes", "sha256") if k not in entry]
        if missing:
            problems.append(f"entry {i}: missing {', '.join(missing)}")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            problems.append(f"{entry['path']}: sha256 is not 64 hex characters")
        if entry["path"] in seen:
            problems.append(f"{entry['path']}: listed twice")
        seen.add(entry["path"])
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-projects", action="store_true", help="also run projects/09 scripts/verify_project.py")
    args = ap.parse_args()
    print(f"Verifying {ROOT}")

    print("Documentation links")
    md = [f for f in tracked("*.md") if not f.relative_to(ROOT).as_posix().startswith(SKIP_LINKS)]
    broken = broken_links(md)
    report("FAIL" if broken else "PASS", f"relative links in {len(md)} Markdown files (archive/ excluded)",
           "; ".join(broken[:15]) + (f" (+{len(broken) - 15} more)" if len(broken) > 15 else ""))

    print("Data files parse")
    bad, known = [], []
    js = [f for f in tracked("*.json") if f.stat().st_size < 50_000_000]
    for f in js:
        try:
            json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            rel = f.relative_to(ROOT).as_posix()
            (known if rel in KNOWN_INVALID_JSON else bad).append(rel if rel in KNOWN_INVALID_JSON else f"{rel}: {exc}")
    detail = "; ".join(bad[:10])
    if known:
        detail += ("; " if detail else "") + "known, documented: " + ", ".join(f"{k} (see {KNOWN_INVALID_JSON[k]})" for k in known)
    report("FAIL" if bad else "PASS", f"{len(js)} JSON files", detail)
    try:
        import yaml
    except ImportError:
        report("NOT RUN", "YAML files", "PyYAML not installed in this Python")
    else:
        bad = []
        ys = tracked("*.yaml", "*.yml")
        for f in ys:
            try:
                list(yaml.safe_load_all(f.read_text(encoding="utf-8")))
            except Exception as exc:  # collect all
                bad.append(f"{f.relative_to(ROOT).as_posix()}: {exc}")
        report("FAIL" if bad else "PASS", f"{len(ys)} YAML files", "; ".join(bad[:10]))

    print("Python sources compile")
    bad = []
    py = tracked("*.py")
    for f in py:
        try:
            compile(f.read_text(encoding="utf-8", errors="replace"), str(f), "exec")
        except SyntaxError as exc:
            bad.append(f"{f.relative_to(ROOT).as_posix()}:{exc.lineno}")
    report("FAIL" if bad else "PASS", f"{len(py)} .py files", ", ".join(bad[:10]))

    print("Model manifest")
    manifest = json.loads((ROOT / "setup" / "models_manifest.json").read_text(encoding="utf-8"))
    problems = manifest_problems(manifest)
    entries = manifest["files"]
    report("FAIL" if problems else "PASS", f"setup/models_manifest.json ({len(entries)} entries)", "; ".join(problems[:10]))
    present = [e for e in entries if (ROOT / e["path"]).exists()]
    wrong_size = [e["path"] for e in present if (ROOT / e["path"]).stat().st_size != e["bytes"]]
    report("FAIL" if wrong_size else "PASS", "downloaded model files have the manifest size",
           f"{len(present)} of {len(entries)} present (python setup/download_models.py [--all] fetches the rest)"
           + (f"; wrong size: {', '.join(wrong_size[:5])}" if wrong_size else ""))

    if args.with_projects:
        print("Project checks")
        script = ROOT / "projects" / "09_plantdx_mlx_knowledge_grounded" / "scripts" / "verify_project.py"
        sys.stdout.flush()  # keep this script's output ahead of the child's
        rc = subprocess.run([sys.executable, str(script), "--quick"], cwd=script.parents[1]).returncode
        report("FAIL" if rc else "PASS", "projects/09 scripts/verify_project.py --quick")

    counts = {s: results.count(s) for s in ("PASS", "FAIL", "NOT RUN")}
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
