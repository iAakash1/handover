"""
Typed configuration loaded from YAML.

WHY THIS EXISTS
    In the previous design, configuration was a Python module of module-level
    constants. That made two things impossible: running two experiments without
    editing source, and knowing afterwards which settings produced a checkpoint.

    Here a config is a value, not a global. It is loaded from YAML, can be
    overridden on the command line, is validated on construction, and is frozen
    into every run directory so a result can always be traced back to the exact
    settings that produced it.

USAGE
    cfg = load_config("configs/qwen25vl_lora_tomato.yaml",
                      overrides={"split.fold": 2, "train.epochs": 10})
    cfg.paths.image_root          # resolved absolute Path
    cfg.to_dict()                 # for freezing into a run directory
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ── Sections ─────────────────────────────────────────────────────────────────

@dataclass
class SourceConfig:
    """Where the validated captions come from. See plantvlm.data.sources."""
    type: str = "excel"
    excel_path: str = "1stJuly/Revised_captions.xlsx"
    sheet: str = "revised_captions"
    image_root: str = "1stJuly/Dataset"
    disease_column: str = "Disease"
    image_column: str = "Image No"
    caption_columns: List[str] = field(default_factory=lambda: [
        "Caption 1", "Caption 2", "Caption 3", "Caption 4", "Caption 5"])
    # Excel class name -> image sub-folder. The only place this mapping lives.
    class_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    instruction: str = "Describe the visible condition of this tomato leaf."
    expected_images_per_class: int = 5
    expected_captions_per_image: int = 5
    valid_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"])
    # Optional provenance check against the dataset the images were taken from.
    provenance_root: Optional[str] = None
    provenance_map: Dict[str, str] = field(default_factory=dict)
    output_dir: str = "data"


@dataclass
class SplitConfig:
    """
    Grouped by image, stratified by class, deterministic.

    'fold' rotates which image of each class is held out, so running folds
    0..N-1 evaluates every image exactly once.
    """
    strategy: str = "grouped_rotation"
    fold: int = 0
    train_per_class: int = 3
    val_per_class: int = 1
    test_per_class: int = 1


@dataclass
class ModelConfig:
    adapter: str = "qwen2_5_vl"          # key in plantvlm.models.ADAPTERS
    name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype: str = "bfloat16"
    # Vision resolution, expressed in 28x28 patches so the token count is explicit.
    # 256 -> ~448x448 -> ~256 visual tokens. Training and inference share these.
    min_pixels_factor: int = 256
    max_pixels_factor: int = 512
    max_length: int = 512


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_suffixes: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"])
    # Substring identifying the vision tower. Modules matching it are never
    # adapted, and the trainer refuses to start if any become trainable.
    exclude_substring: str = "visual"


@dataclass
class TrainConfig:
    epochs: int = 8
    micro_batch: int = 2
    grad_accum: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    grad_clip: float = 1.0
    patience: int = 2
    num_workers: int = 0                 # Windows: keep at 0
    augment: bool = False                # see docs: captions are spatially grounded
    save_every_epoch: bool = True


@dataclass
class GenerateConfig:
    max_new_tokens: int = 120
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    num_samples: int = 1


@dataclass
class EvalConfig:
    memorisation_threshold: float = 0.85
    retrieval_ngram_max: int = 2
    retrieval_stop_words: Optional[str] = "english"
    cross_image_phrases: List[str] = field(default_factory=lambda: [
        "other leaves", "previous leaves", "this group", "this set",
        "rest of the group", "compared with the other", "unlike some of"])


@dataclass
class Paths:
    """All paths resolved to absolute, once, at load time."""
    root: Path
    excel: Path
    image_root: Path
    data_dir: Path
    runs_dir: Path
    provenance_root: Optional[Path]

    @property
    def pairs(self) -> Path: return self.data_dir / "pairs.jsonl"

    @property
    def train(self) -> Path: return self.data_dir / "train.jsonl"

    @property
    def val(self) -> Path: return self.data_dir / "val.jsonl"

    @property
    def test(self) -> Path: return self.data_dir / "test.jsonl"

    @property
    def manifest(self) -> Path: return self.data_dir / "manifest.json"

    @property
    def defects(self) -> Path: return self.data_dir / "caption_defects.json"

    def split_file(self, split: str) -> Path:
        return {"train": self.train, "val": self.val, "test": self.test}[split]


@dataclass
class Config:
    project: str = "plantvlm"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runs_dir: str = "runs"

    paths: Paths = field(init=False, repr=False)
    source_file: Optional[str] = None

    # ── Derived ──────────────────────────────────────────────────────────────

    def resolve(self, root: Path) -> "Config":
        root = Path(root).resolve()
        self.paths = Paths(
            root=root,
            excel=root / self.data.source.excel_path,
            image_root=root / self.data.source.image_root,
            data_dir=root / self.data.output_dir,
            runs_dir=root / self.runs_dir,
            provenance_root=(root / self.data.provenance_root
                             if self.data.provenance_root else None),
        )
        return self

    @property
    def effective_batch(self) -> int:
        return self.train.micro_batch * self.train.grad_accum

    @property
    def min_pixels(self) -> int:
        return self.model.min_pixels_factor * 28 * 28

    @property
    def max_pixels(self) -> int:
        return self.model.max_pixels_factor * 28 * 28

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        return {k: (asdict(v) if is_dataclass(v) else v) for k, v in out.items()}

    def freeze_to(self, path: Path) -> None:
        """Write the exact configuration used by a run. Never read back for logic."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


# ── Loading ──────────────────────────────────────────────────────────────────

_SECTIONS = {
    "data": DataConfig, "split": SplitConfig, "model": ModelConfig,
    "lora": LoRAConfig, "train": TrainConfig, "generate": GenerateConfig,
    "eval": EvalConfig,
}


def _build_section(cls, payload: Dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown key(s) {sorted(unknown)}")
    kwargs = dict(payload)
    if cls is DataConfig and "source" in kwargs:
        src = kwargs["source"]
        src_unknown = set(src) - {f.name for f in fields(SourceConfig)}
        if src_unknown:
            raise ValueError(f"SourceConfig: unknown key(s) {sorted(src_unknown)}")
        kwargs["source"] = SourceConfig(**src)
    return cls(**kwargs)


def _apply_override(cfg: Config, dotted: str, value: Any) -> None:
    """Apply 'train.epochs=10' style overrides with type coercion."""
    section_name, _, key = dotted.partition(".")
    if not key:
        raise ValueError(f"override '{dotted}' must be section.key")
    section = getattr(cfg, section_name, None)
    if section is None or not is_dataclass(section):
        raise ValueError(f"unknown config section '{section_name}'")
    if not hasattr(section, key):
        raise ValueError(f"unknown key '{key}' in section '{section_name}'")
    current = getattr(section, key)
    if isinstance(current, bool) and isinstance(value, str):
        value = value.lower() in ("1", "true", "yes")
    elif isinstance(current, int) and not isinstance(current, bool):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    setattr(section, key, value)


def load_config(path: str | Path,
                overrides: Optional[Dict[str, Any]] = None,
                root: Optional[Path] = None) -> Config:
    """
    Load YAML into a validated Config.

    root defaults to the project directory (the parent of plantvlm/), so every
    relative path in the YAML resolves the same way regardless of the working
    directory a script was launched from.
    """
    path = Path(path)
    if not path.is_absolute():
        path = (root or Path(__file__).resolve().parent.parent) / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    top = {k: v for k, v in payload.items() if k not in _SECTIONS}
    unknown = set(top) - {f.name for f in fields(Config) if f.init}
    if unknown:
        raise ValueError(f"Config: unknown top-level key(s) {sorted(unknown)}")

    cfg = Config(**top, **{name: _build_section(cls, payload.get(name, {}))
                           for name, cls in _SECTIONS.items()})
    cfg.source_file = str(path)

    for dotted, value in (overrides or {}).items():
        _apply_override(cfg, dotted, value)

    cfg.resolve(root or Path(__file__).resolve().parent.parent)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Fail loudly at load time rather than three minutes into a run."""
    s = cfg.split
    per_class = s.train_per_class + s.val_per_class + s.test_per_class
    if per_class != cfg.data.expected_images_per_class:
        raise ValueError(
            f"split sizes ({s.train_per_class}+{s.val_per_class}+{s.test_per_class}"
            f"={per_class}) must sum to expected_images_per_class "
            f"({cfg.data.expected_images_per_class})")
    if not 0 <= s.fold < cfg.data.expected_images_per_class:
        raise ValueError(f"fold must be in [0, {cfg.data.expected_images_per_class})")
    if not cfg.data.source.class_map:
        raise ValueError("data.source.class_map is empty — nothing to load")
    if len(cfg.data.source.caption_columns) != cfg.data.expected_captions_per_image:
        raise ValueError("caption_columns length != expected_captions_per_image")
    if cfg.train.micro_batch < 1 or cfg.train.grad_accum < 1:
        raise ValueError("micro_batch and grad_accum must be >= 1")
    if cfg.model.min_pixels_factor > cfg.model.max_pixels_factor:
        raise ValueError("min_pixels_factor must be <= max_pixels_factor")
