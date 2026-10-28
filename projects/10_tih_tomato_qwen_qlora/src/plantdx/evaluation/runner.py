"""Stage dispatcher for `plantdx evaluate`: inference | analyze | all.

Communication between stages is exclusively through the frozen
`predictions.jsonl` artifact (stage 1's output, stage 2's only input) -- never
in-process state. `--stage all` runs both stages in the current process and
therefore requires an environment with both mlx-vlm AND the `[eval]` extra;
if either is missing it fails closed with that library's own actionable error
rather than silently skipping a stage.
"""

from __future__ import annotations

from plantdx.evaluation.config import EvalConfig
from plantdx.evaluation.integrity import IntegrityReport, check_split_integrity
from plantdx.evaluation.report import run_analysis


def run_evaluation(cfg: EvalConfig) -> dict[str, str]:
    """Dispatch to the requested stage(s); return every written file path."""
    written: dict[str, str] = {}

    if cfg.stage in ("inference", "all"):
        report = check_split_integrity(cfg.dataset_dir, cfg.split)
        written["integrity_check"] = _describe_integrity(report)
        from plantdx.evaluation.inference_runner import run_inference

        predictions_path = run_inference(cfg)
        written["predictions_path"] = str(predictions_path)

    if cfg.stage in ("analyze", "all"):
        analysis = run_analysis(
            cfg.predictions_path,
            output_dir=cfg.output_dir,
            model_path=cfg.model_path,
            adapter_path=cfg.adapter_path,
            dataset_dir=cfg.dataset_dir,
            seed=cfg.seed,
        )
        written.update(analysis)

    return written


def _describe_integrity(report: IntegrityReport) -> str:
    return (
        f"OK: {report.train_image_count} train / {report.eval_image_count} eval images, 0 overlap"
    )
