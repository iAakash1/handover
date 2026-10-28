"""Configuration loader, validator, and nested dictionary wrapper."""

import os
from pathlib import Path
from typing import Any, Dict, Union
import yaml


class ConfigDict(dict):
    """Dictionary supporting dot notation attribute access and recursion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to standard Python dict."""
        res = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                res[k] = v.to_dict()
            else:
                res[k] = v
        return res


def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """Load and validate YAML configuration file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dictionary in config file, got {type(raw)}")

    config = ConfigDict(raw)
    validate_config(config)
    return config


def validate_config(config: ConfigDict) -> None:
    """Ensure essential configuration sections and fields are defined."""
    required_sections = ["dataset", "model", "training", "loss", "checkpoint", "prototype", "system"]
    for sec in required_sections:
        if sec not in config:
            raise ValueError(f"Missing required configuration section: '{sec}'")

    # Dataset validation
    if "root" not in config.dataset:
        raise ValueError("config.dataset.root must be specified")

    # Model validation
    if "framework" not in config.model:
        raise ValueError("config.model.framework must be specified ('open_clip' or 'siglip')")

    framework = config.model.framework.lower()
    if framework not in ("open_clip", "siglip"):
        raise ValueError(f"Unsupported model framework '{framework}'. Must be 'open_clip' or 'siglip'")

    if framework == "open_clip":
        if "architecture" not in config.model:
            raise ValueError("config.model.architecture must be specified for open_clip")
    elif framework == "siglip":
        if "name" not in config.model:
            raise ValueError("config.model.name must be specified for siglip")
