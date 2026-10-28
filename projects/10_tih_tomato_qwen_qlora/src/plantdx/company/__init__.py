"""Company (TIH) tomato training path.

A self-contained, CUDA/PyTorch fine-tuning path for the organization-provided
tomato dataset: class mapping, dataset preparation, QLoRA fine-tuning of
Qwen2.5-VL-7B-Instruct, held-out evaluation, and single-image prediction.

This package is deliberately separate from the research pipeline in the rest of
``plantdx``. It reuses the frozen knowledge base only as secondary reference
metadata and never feeds the research artifacts.
"""

from __future__ import annotations

from plantdx.company.classes import ClassEntry, CompanyDataset, load_company_dataset
from plantdx.company.config import CompanyTrainConfig, load_company_train_config
from plantdx.company.registry import ActiveAdapter, read_active_adapter, write_active_adapter

__all__ = [
    "ActiveAdapter",
    "ClassEntry",
    "CompanyDataset",
    "CompanyTrainConfig",
    "load_company_dataset",
    "load_company_train_config",
    "read_active_adapter",
    "write_active_adapter",
]
