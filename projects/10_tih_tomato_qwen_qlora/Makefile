# PlantDx developer Makefile. Run `make help` for the task list.
.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help install install-dev install-train install-eval hooks fmt lint type test test-unit test-integration \
        cov clean docs check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime only)
	pip install -e .

install-dev: ## Install package + dev tooling
	pip install -e ".[dev]"

install-train: ## Install training extras (Apple Silicon / MLX)
	pip install -e ".[train]"

install-eval: ## Install + cache the evaluation metrics stack (--stage analyze env)
	./scripts/setup_eval_env.sh

hooks: ## Install pre-commit hooks
	pre-commit install

fmt: ## Auto-format (ruff format + import sort)
	ruff format src tests
	ruff check --fix src tests

lint: ## Lint without modifying
	ruff check src tests

type: ## Static type check
	mypy

test: ## Run the full test suite
	pytest

test-unit: ## Run unit tests only
	pytest -m unit

test-integration: ## Run integration tests only
	pytest -m integration

cov: ## Run tests with coverage report
	pytest --cov=plantdx --cov-report=term-missing --cov-report=html

check: fmt lint type test ## Full local CI (format, lint, type, test)

docs: ## Build documentation site
	mkdocs build

clean: ## Remove caches and build artifacts (NOT pipeline artifacts/)
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
