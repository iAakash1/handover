"""Training service: command construction, run state, log parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from app import training_service as svc


@pytest.mark.unit
def test_train_argv_is_an_argument_list_not_a_shell_string() -> None:
    argv = svc.build_train_argv({})
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)
    assert argv[0] == sys.executable
    assert argv[1:5] == ["-m", "plantdx", "company", "train"]
    assert "--config" in argv


@pytest.mark.unit
def test_train_argv_carries_every_advanced_setting() -> None:
    argv = svc.build_train_argv(
        {
            "run_name": "tih_experiment_b",
            "epochs": 5,
            "learning_rate": 2e-5,
            "lora_rank": 32,
            "lora_alpha": 64,
            "gradient_accumulation_steps": 4,
            "image_max_side": 336,
            "seed": 11,
        }
    )
    for flag, value in [
        ("--run-name", "tih_experiment_b"),
        ("--epochs", "5"),
        ("--lora-rank", "32"),
        ("--lora-alpha", "64"),
        ("--gradient-accumulation-steps", "4"),
        ("--image-max-side", "336"),
        ("--seed", "11"),
    ]:
        assert flag in argv
        assert argv[argv.index(flag) + 1] == value
    assert argv[argv.index("--learning-rate") + 1].startswith("2e-05")


@pytest.mark.unit
def test_train_argv_omits_unset_overrides() -> None:
    argv = svc.build_train_argv({"epochs": None, "learning_rate": None})
    assert "--epochs" not in argv
    assert "--learning-rate" not in argv


@pytest.mark.unit
def test_resume_flag() -> None:
    argv = svc.build_train_argv({}, resume=True)
    assert "--resume" in argv
    assert "--resume" not in svc.build_train_argv({})


@pytest.mark.unit
def test_train_argv_is_parseable_by_the_cli() -> None:
    """The exact command the button launches must parse without error."""
    from plantdx.cli import build_parser

    argv = svc.build_train_argv({"epochs": 3, "lora_rank": 16})
    args = build_parser().parse_args(argv[3:])  # drop "<python> -m plantdx"
    assert args.command == "company"
    assert args.subcommand == "train"
    assert args.epochs == 3.0
    assert args.lora_rank == 16


@pytest.mark.unit
def test_evaluation_argv_is_parseable_by_the_cli() -> None:
    from plantdx.cli import build_parser

    argv = svc.run_evaluation_subprocess("tih_tomato_qwen_qlora", "test", base=True)
    args = build_parser().parse_args(argv[3:])
    assert args.command == "company"
    assert args.subcommand == "evaluate"
    assert args.split == "test"
    assert args.base is True


@pytest.mark.unit
def test_run_state_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "STATE_FILE", tmp_path / "state.json")
    assert svc.read_state() is None

    state = svc.RunState(
        run_name="r", pid=4242, log_path=str(tmp_path / "t.log"), argv=["a"], started_at=1.0
    )
    svc.write_state(state)
    loaded = svc.read_state()
    assert loaded is not None
    assert loaded.pid == 4242
    assert loaded.run_name == "r"

    svc.clear_state()
    assert svc.read_state() is None


@pytest.mark.unit
def test_corrupt_state_file_reads_as_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(svc, "STATE_FILE", path)
    assert svc.read_state() is None


@pytest.mark.unit
def test_status_is_idle_without_a_state() -> None:
    assert svc.status(None, None) == svc.IDLE


@pytest.mark.unit
def test_status_reads_a_recorded_return_code(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text("nothing useful", encoding="utf-8")
    state = svc.RunState(
        run_name="r", pid=-1, log_path=str(log), argv=[], started_at=0.0, returncode=0
    )
    assert svc.status(state, None) == svc.SUCCEEDED
    state.returncode = 1
    assert svc.status(state, None) == svc.FAILED


@pytest.mark.unit
def test_a_dead_run_whose_log_shows_completion_counts_as_succeeded(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text("Training tih...\nAdapter written : checkpoints/tih\nCOMPANY RUN COMPLETE\n", encoding="utf-8")
    state = svc.RunState(run_name="r", pid=-1, log_path=str(log), argv=[], started_at=0.0)
    assert svc.status(state, None) == svc.SUCCEEDED


@pytest.mark.unit
def test_a_dead_run_with_no_completion_marker_counts_as_failed(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text("Preparing dataset...\nboom\n", encoding="utf-8")
    state = svc.RunState(run_name="r", pid=-1, log_path=str(log), argv=[], started_at=0.0)
    assert svc.status(state, None) == svc.FAILED


@pytest.mark.unit
def test_failure_reason_surfaces_error_lines(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text(
        "Preparing dataset...\nTraceback (most recent call last):\n"
        "RuntimeError: CUDA out of memory\n",
        encoding="utf-8",
    )
    reason = svc.failure_reason(log)
    assert "CUDA out of memory" in reason


@pytest.mark.unit
def test_failure_reason_is_never_empty(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text("all quiet\n", encoding="utf-8")
    assert svc.failure_reason(log)


@pytest.mark.unit
def test_last_stage_tracks_progress(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text("Preparing dataset...\n  loading base model\n", encoding="utf-8")
    assert svc.last_stage(log) == "loading base model"


@pytest.mark.unit
def test_read_log_tolerates_a_missing_file(tmp_path: Path) -> None:
    assert svc.read_log(tmp_path / "absent.log") == ""


@pytest.mark.unit
def test_progress_is_parsed_from_the_loss_history(tmp_path: Path) -> None:
    class Cfg:
        log_dir = tmp_path

        class optim:  # noqa: N801 - mirrors the config object's shape
            epochs = 3.0

    path = tmp_path / "loss_history.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "total_steps": 100,
                    "epoch": step / 40,
                    "total_epochs": 3.0,
                    "loss": 5.0 - step * 0.1,
                    "learning_rate": 5e-5,
                    "elapsed_seconds": step * 4.5,
                    "memory": {"max_reserved_gb": 9.8},
                }
            )
            for step in (10, 20, 30)
        ),
        encoding="utf-8",
    )
    progress = svc.read_progress(Cfg())
    assert progress.step == 30
    assert progress.total_steps == 100
    assert progress.loss == pytest.approx(2.0)
    assert progress.fraction == pytest.approx(0.3)
    assert len(progress.history) == 3
    assert progress.memory["max_reserved_gb"] == 9.8


@pytest.mark.unit
def test_progress_is_empty_before_training_writes_anything(tmp_path: Path) -> None:
    class Cfg:
        log_dir = tmp_path

        class optim:  # noqa: N801 - mirrors the config object's shape
            epochs = 3.0

    progress = svc.read_progress(Cfg())
    assert progress.step == 0
    assert progress.fraction == 0.0


@pytest.mark.unit
def test_is_alive_is_false_for_an_impossible_pid() -> None:
    assert svc.is_alive(-1) is False
    assert svc.is_alive(0) is False
