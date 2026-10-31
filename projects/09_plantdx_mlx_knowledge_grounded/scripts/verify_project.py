#!/usr/bin/env python3
"""Check that this project is complete and wired together, without training anything.

Runs on any OS (macOS, Linux, Windows) with plain Python 3.10+; PyYAML and the plantdx package are used when
installed and the checks that need them are reported as NOT RUN otherwise. Nothing here needs MLX or a GPU.

    python scripts/verify_project.py            # from projects/09_plantdx_mlx_knowledge_grounded
    python scripts/verify_project.py --quick    # skip the per-subcommand CLI help calls

Every check prints PASS, FAIL, MISSING (an artifact that is known to be absent, e.g. the adapter weights) or
NOT RUN (a dependency is not installed here). The exit code is 1 only when something FAILs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    # run name -> (crop, config, expected rows train/validation/test of the July run, raw image folder)
    "qwen25vl_tomato_qlora": ("tomato", "configs/train/qwen25vl_tomato.yaml", (16201, 895, 910), "tomato/raw/PlantVillage"),
    "qwen25vl_mango_qlora": ("mango", "configs/train/qwen25vl_mango.yaml", (3600, 200, 200), "mango/raw/MangoLeafBD"),
}
RAW_IMAGES = {"tomato": 18160, "mango": 4000}
SUBCOMMANDS = ("audit", "normalize", "ontology", "vocabulary", "concepts", "templates", "generate", "validate",
               "corpus", "dataset", "qa", "prepare-training", "train", "infer", "evaluate")
REQUIRED = (
    "pyproject.toml", "requirements-mlx.txt", "src/plantdx/cli.py", "configs/config.yaml", "configs/paths.yaml",
    "configs/training.yaml", "configs/models/qwen25vl.yaml", "configs/lora/qlora.yaml",
    "knowledge_base/dkb.json", "artifacts/corpus/captions.jsonl", "assets/training/instructions.json",
    "assets/training/instructions_mango.json", "assets/metadata/label_map.json", "streamlit_app.py",
    "models/README.md", "models/zero_shot_benchmarks/README.md", "scripts/setup_eval_env.sh",
)
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")

results: list[tuple[str, str, str]] = []


def report(status: str, check: str, detail: str = "") -> None:
    results.append((status, check, detail))
    print(f"  {status:8} {check}" + (f" - {detail}" if detail else ""))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative_to_datasets(image_path: str) -> str | None:
    """'/Users/x/experiments/datasets/tomato/processed/a/b.JPG' -> 'datasets/tomato/processed/a/b.JPG'."""
    p = image_path.replace("\\", "/")
    i = p.find("/datasets/")
    return p[i + 1:] if i >= 0 else None


def markdown_links(md_file: Path) -> list[str]:
    """Relative link targets in a Markdown file (code blocks, URLs and anchors are skipped)."""
    text = re.sub(r"```.*?```", "", md_file.read_text(encoding="utf-8", errors="replace"), flags=re.S)
    out = []
    for m in LINK_RE.finditer(text):
        target = m.group(1)
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        path = urllib.parse.unquote(target.split("#")[0])
        if path:
            out.append(path)
    return out


def check_required() -> None:
    print("Required files")
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    for run, (_, cfg, _, _) in RUNS.items():
        for f in (cfg, f"checkpoints/{run}/adapter_config.json", f"logs/{run}/train.log",
                  f"reports/{run}/training_plan.json", f"reports/{run}/evaluation/metrics.json",
                  f"reports/{run}/evaluation/system_info.json", f"models/{run}/README.md",
                  *(f"artifacts/training/{run}/dataset/{s}.jsonl" for s in ("train", "validation", "test"))):
            if not (ROOT / f).exists():
                missing.append(f)
    report("FAIL" if missing else "PASS", "project files", ", ".join(missing) or f"{len(REQUIRED) + 10 * len(RUNS)} present")


def check_parse() -> None:
    print("Config and JSON files parse")
    try:
        import yaml
    except ImportError:
        report("NOT RUN", "YAML configs", "PyYAML not installed")
        yaml = None
    if yaml is not None:
        bad = []
        for f in sorted((ROOT / "configs").rglob("*.y*ml")):
            try:
                yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception as exc:  # report every broken file, not just the first
                bad.append(f"{f.relative_to(ROOT)}: {exc}")
        report("FAIL" if bad else "PASS", "YAML configs", "; ".join(bad) or "all configs/*.yaml parse")
    bad, n = [], 0
    for pattern in ("checkpoints/*/adapter_config.json", "reports/*/evaluation/*.json", "datasets/**/*.json",
                    "artifacts/training/*/dataset/manifest.json"):
        for f in ROOT.glob(pattern):
            n += 1
            try:
                json.loads(f.read_text(encoding="utf-8"))
            except ValueError as exc:
                bad.append(f"{f.relative_to(ROOT)}: {exc}")
    report("FAIL" if bad else "PASS", "JSON records", "; ".join(bad) or f"{n} files parse")


def check_package() -> None:
    print("plantdx package and configs")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from plantdx.config import load_config
        from plantdx.training.config import load_training_config
    except ImportError as exc:
        report("NOT RUN", "plantdx import", f"{exc} (pip install -e .)")
        return
    report("PASS", "plantdx import")
    try:
        load_config(ROOT / "configs/config.yaml")
        report("PASS", "configs/config.yaml loads through plantdx.config.load_config")
    except Exception as exc:
        report("FAIL", "configs/config.yaml", str(exc))
    for run, (crop, cfg, _, _) in RUNS.items():
        try:
            tc = load_training_config(ROOT / cfg, base_dir=ROOT)
            ok = tc.run_name == run and tc.data.crop == crop
            report("PASS" if ok else "FAIL", f"{cfg}", f"run_name={tc.run_name}, crop={tc.data.crop}, "
                   f"lora r{tc.lora.rank}/alpha {tc.lora.alpha}, seed {tc.seed}")
        except Exception as exc:
            report("FAIL", cfg, str(exc))


def check_cli(quick: bool) -> None:
    print("CLI entry points")
    code = "import sys; sys.path.insert(0, 'src'); from plantdx.cli import main; sys.argv = ['plantdx'] + sys.argv[1:]; main()"
    try:
        import plantdx  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(ROOT / "src"))
    probe = subprocess.run([sys.executable, "-c", code, "--help"], cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0 and "No module named" in probe.stderr:
        report("NOT RUN", "plantdx --help", probe.stderr.strip().splitlines()[-1])
        return
    report("PASS" if probe.returncode == 0 else "FAIL", "plantdx --help")
    if quick:
        report("NOT RUN", "subcommand --help", "--quick")
        return
    failed = [s for s in SUBCOMMANDS
              if subprocess.run([sys.executable, "-c", code, s, "--help"], cwd=ROOT, capture_output=True).returncode != 0]
    report("FAIL" if failed else "PASS", f"{len(SUBCOMMANDS)} subcommands --help", ", ".join(failed))


def check_checkpoints() -> None:
    print("Checkpoints")
    try:
        import yaml
        lora = yaml.safe_load((ROOT / "configs/lora/qlora.yaml").read_text(encoding="utf-8"))
    except Exception:
        lora = None
    for run in RUNS:
        cfg_file = ROOT / f"checkpoints/{run}/adapter_config.json"
        if not cfg_file.exists():
            report("FAIL", f"{run} adapter_config.json", "missing")
            continue
        params = json.loads(cfg_file.read_text(encoding="utf-8"))["lora_parameters"]
        detail = f"rank {params['rank']}, scale {params['scale']}, {len(params['keys'])} modules"
        consistent = lora is None or (params["rank"] == lora["rank"] and params["scale"] == lora["alpha"] / lora["rank"])
        report("PASS" if consistent else "FAIL", f"{run} adapter_config matches configs/lora/qlora.yaml", detail)
        weights = cfg_file.with_name("adapters.safetensors")
        recorded = json.loads((ROOT / f"reports/{run}/evaluation/system_info.json").read_text(encoding="utf-8"))
        expected = recorded["adapter_checksum"].removeprefix("sha256:")
        if not weights.exists():
            report("MISSING", f"{run} adapters.safetensors",
                   f"not in git; July sha256 {expected[:8]}..., retrain (models/{run}/README.md)")
        elif sha256(weights) == expected:
            report("PASS", f"{run} adapters.safetensors", "identical to the July weights")
        else:
            report("PASS", f"{run} adapters.safetensors", "present (retrained: checksum differs from July)")


def check_datasets() -> None:
    print("Datasets")
    for crop, expected in RAW_IMAGES.items():
        folder = next(v[3] for v in RUNS.values() if v[0] == crop)
        n = sum(1 for p in (ROOT / folder).rglob("*") if p.suffix.lower() in IMAGE_EXT)
        report("PASS" if n == expected else "FAIL", f"{crop} raw images ({folder})", f"{n} (expected {expected})")
    for run, (crop, _, counts, _) in RUNS.items():
        rows, here, mapped, n_rows = 0, 0, 0, []
        processed = ROOT / "datasets" / crop / "processed"
        for split in ("train", "validation", "test"):
            lines = (ROOT / f"artifacts/training/{run}/dataset/{split}.jsonl").read_text(encoding="utf-8").splitlines()
            n_rows.append(len(lines))
            for line in lines:
                image = json.loads(line)["image"]
                rows += 1
                here += Path(image).exists()
                rel = relative_to_datasets(image)
                mapped += bool(rel and processed.is_dir() and (ROOT / rel).exists())
        report("PASS" if tuple(n_rows) == counts else "FAIL", f"{run} split sizes", "/".join(map(str, n_rows)))
        if here == rows:
            report("PASS", f"{run} image paths", "every path resolves on this machine")
        elif processed.is_dir():
            report("PASS" if mapped == rows else "FAIL", f"{run} images found under datasets/{crop}/processed",
                   f"{mapped}/{rows} (paths in the July files are the Mac's; run `plantdx prepare-training` to rewrite them)")
        else:
            report("NOT RUN", f"{run} image paths", f"July files use Mac paths; run `plantdx normalize` first")


def check_python() -> None:
    print("Python sources compile")
    bad, n = [], 0
    for folder in ("src", "app", "tests", "scripts", "earlier_experiments"):
        for f in (ROOT / folder).rglob("*.py"):
            n += 1
            try:
                compile(f.read_text(encoding="utf-8"), str(f), "exec")
            except SyntaxError as exc:
                bad.append(f"{f.relative_to(ROOT)}:{exc.lineno}")
    report("FAIL" if bad else "PASS", f"{n} .py files", ", ".join(bad))


def check_links() -> None:
    print("Documentation links")
    broken, n = [], 0
    for md in ROOT.rglob("*.md"):
        rel = md.relative_to(ROOT).as_posix()
        if rel.startswith(("tomato/", "mango/", "datasets/tomato/processed", "datasets/mango/processed")):
            continue
        for target in markdown_links(md):
            n += 1
            if not (md.parent / target).exists():
                broken.append(f"{rel} -> {target}")
    report("FAIL" if broken else "PASS", f"{n} relative links", "; ".join(broken[:10]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="skip the per-subcommand --help calls")
    args = ap.parse_args()
    print(f"Verifying {ROOT}")
    for check in (check_required, check_parse, check_package, lambda: check_cli(args.quick), check_checkpoints,
                  check_datasets, check_python, check_links):
        check()
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "FAIL", "MISSING", "NOT RUN")}
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
