"""The launch plumbing behind RUN TRAINING, exercised without training anything.

``start_training`` is driven for real -- a child process is spawned, its output
is streamed to the run log, state and lock files are written, status transitions
are observed, and a stop is delivered -- but ``build_train_argv`` is redirected
to a harmless command. That proves the wiring the button depends on while never
starting a company training run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from app import training_service as svc

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the service's state, lock, and log at a temporary directory."""
    monkeypatch.setattr(svc, "STATE_FILE", tmp_path / "training_state.json")
    monkeypatch.setattr(svc, "LOCK_FILE", tmp_path / "training.lock")

    class Cfg:
        run_name = "plumbing_probe"
        log_dir = tmp_path / "logs"
        checkpoint_dir = tmp_path / "checkpoints" / "plumbing_probe"

        class optim:  # noqa: N801 - mirrors the real config object's shape
            epochs = 3.0

    monkeypatch.setattr(svc, "load_config", lambda overrides=None: Cfg())
    return tmp_path


def _use_command(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    """Replace the training command with a harmless inline Python program."""
    monkeypatch.setattr(
        svc,
        "build_train_argv",
        lambda overrides, resume=False: [sys.executable, "-c", code],
    )


def _wait(predicate, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


@pytest.mark.integration
def test_launch_streams_output_and_reports_success(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_command(
        monkeypatch,
        "import sys; print('Preparing dataset...'); "
        "print('Adapter written : somewhere'); sys.stdout.flush()",
    )
    state, process = svc.start_training({})

    assert state.run_id.startswith("plumbing_probe-")
    assert svc.read_state() is not None
    assert svc.LOCK_FILE.is_file()

    assert _wait(lambda: process.poll() is not None), "child did not exit"
    assert svc.status(state, process) == svc.SUCCEEDED

    log = svc.read_log(state.log_path)
    assert "Preparing dataset" in log
    assert "Adapter written" in log
    assert state.run_id in log, "the run id must be recorded in the log header"

    svc.release_lock_if_finished(state)
    assert not svc.LOCK_FILE.is_file()


@pytest.mark.integration
def test_a_failing_child_is_reported_with_its_error(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_command(
        monkeypatch,
        "import sys; print('Preparing dataset...'); "
        "print('RuntimeError: CUDA out of memory', file=sys.stderr); sys.exit(1)",
    )
    state, process = svc.start_training({})
    assert _wait(lambda: process.poll() is not None)

    assert svc.status(state, process) == svc.FAILED
    assert "CUDA out of memory" in svc.failure_reason(state.log_path)
    assert svc.last_stage(state.log_path) == "Preparing dataset"


@pytest.mark.integration
def test_a_second_launch_is_refused_while_the_first_runs(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_command(monkeypatch, "import time; time.sleep(30)")
    state, process = svc.start_training({})
    try:
        assert svc.status(state, process) == svc.RUNNING
        with pytest.raises(RuntimeError, match="already active"):
            svc.start_training({})
    finally:
        svc.stop_training(state)
        process.wait(timeout=30)


@pytest.mark.integration
def test_stop_terminates_the_child_and_keeps_its_checkpoint(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = isolated / "checkpoints" / "plumbing_probe"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"already saved")

    _use_command(monkeypatch, "import time; time.sleep(60)")
    state, process = svc.start_training({})
    assert svc.status(state, process) == svc.RUNNING

    assert svc.stop_training(state) is True
    assert _wait(lambda: process.poll() is not None), "child survived the stop"

    assert (checkpoint / "adapter_model.safetensors").read_bytes() == b"already saved"
    assert svc.status(state, None) == svc.STOPPED
    assert not svc.LOCK_FILE.is_file()


@pytest.mark.integration
def test_state_survives_a_browser_reload(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rerun with no Popen handle must still find the run through the state file."""
    _use_command(monkeypatch, "import time; time.sleep(20)")
    state, process = svc.start_training({})
    try:
        reloaded = svc.read_state()
        assert reloaded is not None
        assert reloaded.pid == state.pid
        assert reloaded.run_id == state.run_id
        assert svc.status(reloaded, None) == svc.RUNNING
    finally:
        svc.stop_training(state)
        process.wait(timeout=30)


@pytest.mark.integration
def test_the_real_training_command_resolves_and_is_runnable() -> None:
    """The argv the button builds must name a real, importable CLI entry point.

    Runs ``--help`` on the exact command path (never the command itself), so the
    interpreter, module, subcommand, and flags are all proved to resolve.
    """
    import subprocess

    argv = svc.build_train_argv({"epochs": 3, "lora_rank": 16})
    result = subprocess.run(
        [*argv, "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "--epochs" in result.stdout
    assert "--resume" in result.stdout
