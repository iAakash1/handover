"""
Shared entry-point plumbing for the numbered scripts.

The scripts are deliberately thin: they parse arguments, load a config, call the
library, and print. All logic lives in plantvlm/ so it can be tested, reused, and
imported from a notebook without running a CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plantvlm.config import Config, load_config          # noqa: E402

DEFAULT_CONFIG = "configs/qwen25vl_lora_tomato.yaml"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"experiment YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value",
                        help="override a config value, repeatable "
                             "(e.g. --set train.epochs=12)")


def parse_overrides(items) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"bad override '{item}' — expected section.key=value")
        key, _, value = item.partition("=")
        out[key.strip()] = value.strip()
    return out


def load(args) -> Config:
    cfg = load_config(args.config, parse_overrides(args.overrides), PROJECT_ROOT)
    return cfg


def banner(title: str, cfg: Config, extra: Tuple[Tuple[str, str], ...] = ()) -> None:
    print("=" * 74)
    print(f"  {title}")
    print(f"  config : {Path(cfg.source_file).name}   fold {cfg.split.fold}   "
          f"seed {cfg.seed}")
    for label, value in extra:
        print(f"  {label:<7}: {value}")
    print("=" * 74)
