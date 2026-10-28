"""Evaluation package (Milestone 6): base vs. fine-tuned comparison.

Compares a fine-tuned model against the base model on the frozen test split of
whichever crop's dataset is configured (crop is read from the dataset's own
manifest, never hardcoded here). Public surface:
- ``EvalConfig`` / ``resolve_eval_config`` -- resolved, validated run settings.
- ``run_evaluation`` -- the CLI's stage dispatcher (inference | analyze | all).
- ``run_inference`` -- stage 1 (lazy mlx-vlm; writes predictions.jsonl).
- ``run_analysis`` -- stage 2 (lazy metrics stack; reads predictions.jsonl,
  writes every report file).
- ``check_split_integrity`` -- the train/eval leakage guard.

The M6-era ``ComparisonReporter`` / ``ZeroShotEvaluator`` / stub
``classification_metrics`` / ``caption_metrics`` interfaces are retained but
superseded by the modules above (same "new impl supersedes old stub" pattern
as ``concepts/`` vs ``ontology/{builder,models}``) -- they described a
four-model zero-shot comparison matrix, a broader scope than this milestone's
single base-vs-fine-tuned model pair.
"""

from __future__ import annotations

from plantdx.evaluation.compare import ComparisonReporter
from plantdx.evaluation.config import EvalConfig, resolve_eval_config
from plantdx.evaluation.inference_runner import run_inference
from plantdx.evaluation.integrity import check_split_integrity
from plantdx.evaluation.metrics import caption_metrics, classification_metrics
from plantdx.evaluation.report import run_analysis
from plantdx.evaluation.runner import run_evaluation
from plantdx.evaluation.zero_shot import ZeroShotEvaluator

__all__ = [
    "ComparisonReporter",
    "EvalConfig",
    "ZeroShotEvaluator",
    "caption_metrics",
    "check_split_integrity",
    "classification_metrics",
    "resolve_eval_config",
    "run_analysis",
    "run_evaluation",
    "run_inference",
]
