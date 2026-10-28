"""Headless orchestration behind the Training page.

No Streamlit here: the page is only a view over these functions. Training runs
as a child process so a CUDA error, an OOM, or a hard crash kills the child and
leaves the app running with a readable log.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.utils import REPO_ROOT

STATE_FILE = REPO_ROOT / "artifacts" / "runtime" / "training_state.json"
LOCK_FILE = REPO_ROOT / "artifacts" / "runtime" / "training.lock"
DEFAULT_CONFIG = "configs/train/qwen25vl_tih_tomato_strong.yaml"

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
STOPPED = "stopped"
IDLE = "idle"


@dataclass
class RunState:
    """A launched (or finished) training run."""

    run_name: str
    pid: int
    log_path: str
    argv: list[str]
    started_at: float
    run_id: str = ""
    checkpoint_dir: str = ""
    finished_at: float | None = None
    returncode: int | None = None
    stopped_by_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "pid": self.pid,
            "log_path": self.log_path,
            "checkpoint_dir": self.checkpoint_dir,
            "argv": self.argv,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "stopped_by_user": self.stopped_by_user,
        }

    @property
    def elapsed(self) -> float:
        """Seconds since the run started (or its total runtime once finished)."""
        return (self.finished_at or time.time()) - self.started_at


@dataclass
class Progress:
    """The latest parsed training progress."""

    step: int = 0
    total_steps: int = 0
    epoch: float = 0.0
    total_epochs: float = 0.0
    loss: float | None = None
    validation_loss: float | None = None
    learning_rate: float | None = None
    elapsed_seconds: float = 0.0
    memory: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        """Overall completion in [0, 1]; 0 until the total is known."""
        if self.total_steps <= 0:
            return 0.0
        return max(0.0, min(1.0, self.step / self.total_steps))

    @property
    def epoch_fraction(self) -> float:
        """Completion of the current epoch in [0, 1]."""
        if self.epoch <= 0:
            return 0.0
        return max(0.0, min(1.0, self.epoch - int(self.epoch) or 1.0))

    @property
    def current_epoch(self) -> int:
        """1-based index of the epoch being trained."""
        if self.total_epochs <= 0:
            return 1
        return min(int(self.total_epochs), int(self.epoch) + 1)

    @property
    def eta_seconds(self) -> float | None:
        """Seconds remaining, extrapolated from the observed step rate."""
        if self.step <= 0 or self.total_steps <= 0 or self.elapsed_seconds <= 0:
            return None
        return (self.elapsed_seconds / self.step) * (self.total_steps - self.step)


def load_config(overrides: dict[str, Any] | None = None) -> Any:
    """Load the company training config with the advanced-settings overrides."""
    from plantdx.company.config import load_company_train_config

    return load_company_train_config(DEFAULT_CONFIG, overrides=overrides or {})


def load_dataset(cfg: Any) -> Any:
    """Load the company dataset profile referenced by ``cfg``."""
    from plantdx.company.classes import load_company_dataset

    return load_company_dataset(cfg.data.profile)


def run_preflight(overrides: dict[str, Any] | None = None, *, deep: bool = False) -> Any:
    """Run the preflight checks, converting any unexpected error into a failed check."""
    from plantdx.company.preflight import FAIL, Check, PreflightReport
    from plantdx.company.preflight import run_preflight as _run

    try:
        return _run(config_path=DEFAULT_CONFIG, overrides=overrides, deep=deep)
    except Exception as exc:
        return PreflightReport(checks=[Check("Preflight", FAIL, f"{type(exc).__name__}: {exc}")])


def smoke_status(cfg: Any) -> Any:
    """The last recorded smoke-test result for this run, or ``None``."""
    from plantdx.company.smoke import last_smoke

    try:
        return last_smoke(cfg)
    except Exception:
        return None


def build_train_argv(
    overrides: dict[str, Any],
    *,
    resume: bool = False,
) -> list[str]:
    """The exact argument list the Train button launches. Never a shell string."""
    argv = [
        sys.executable,
        "-m",
        "plantdx",
        "company",
        "train",
        "--config",
        DEFAULT_CONFIG,
    ]
    flags = {
        "run_name": "--run-name",
        "epochs": "--epochs",
        "learning_rate": "--learning-rate",
        "lora_rank": "--lora-rank",
        "lora_alpha": "--lora-alpha",
        "gradient_accumulation_steps": "--gradient-accumulation-steps",
        "image_max_side": "--image-max-side",
        "seed": "--seed",
    }
    for key, flag in flags.items():
        if overrides.get(key) is not None:
            argv += [flag, str(overrides[key])]
    if resume:
        argv.append("--resume")
    return argv


def start_training(
    overrides: dict[str, Any],
    *,
    resume: bool = False,
) -> tuple[RunState, subprocess.Popen[bytes]]:
    """Launch the official training run as a child process and record its state.

    This is the only function in the app that starts a full training run, and it
    is called from exactly one place: the RUN TRAINING button's click branch. A
    lock file taken before the spawn makes a duplicate launch impossible even if
    Streamlit reruns the script while the process is still starting.

    Raises:
        RuntimeError: A training run is already in progress.
    """
    existing = read_state()
    if existing is not None and is_alive(existing.pid):
        raise RuntimeError(
            f"training run {existing.run_name!r} is already active (pid {existing.pid}); "
            "stop it before starting another"
        )
    lock = _acquire_lock()

    try:
        cfg = load_config(overrides)
        argv = build_train_argv(overrides, resume=resume)
        run_id = f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        log_path = cfg.log_dir / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handle = log_path.open("ab" if resume else "wb")
        handle.write(f"# run_id: {run_id}{os.linesep}$ {' '.join(argv)}{os.linesep}".encode())
        handle.flush()
        process = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except BaseException:
        _release_lock(lock)
        raise

    state = RunState(
        run_name=cfg.run_name,
        run_id=run_id,
        pid=process.pid,
        log_path=str(log_path),
        checkpoint_dir=str(cfg.checkpoint_dir),
        argv=argv,
        started_at=time.time(),
    )
    write_state(state)
    lock.write_text(json.dumps({"pid": process.pid, "run_id": run_id}), encoding="utf-8")
    return state, process


def _acquire_lock() -> Path:
    """Create the start lock, or fail if another launch already holds it.

    Exclusive create, so two Streamlit reruns racing to start cannot both win.
    A lock whose recorded process is gone is stale and gets reclaimed.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_FILE.touch(exist_ok=False)
        return LOCK_FILE
    except FileExistsError:
        pass

    holder = 0
    try:
        holder = int(json.loads(LOCK_FILE.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, ValueError, AttributeError):
        holder = 0
    if holder and is_alive(holder):
        raise RuntimeError(f"another training launch is in progress (pid {holder})")
    LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.touch(exist_ok=False)
    return LOCK_FILE


def _release_lock(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def release_lock_if_finished(state: RunState | None) -> None:
    """Drop the start lock once the run it guards is no longer alive."""
    if state is None or not is_alive(state.pid):
        LOCK_FILE.unlink(missing_ok=True)


def stop_training(state: RunState) -> bool:
    """Terminate a running child, keeping its checkpoints and logs intact.

    Signals the whole process tree. Checkpoints already written to disk are
    untouched, so a stopped run stays resumable.
    """
    if not is_alive(state.pid):
        LOCK_FILE.unlink(missing_ok=True)
        return False
    delivered = False
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(state.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=20,
            )
            delivered = True
        except (OSError, subprocess.SubprocessError):
            delivered = False
    else:
        try:
            os.kill(state.pid, signal.SIGTERM)
            delivered = True
        except OSError:
            delivered = False
    state.stopped_by_user = True
    state.finished_at = time.time()
    write_state(state)
    LOCK_FILE.unlink(missing_ok=True)
    return delivered


def is_alive(pid: int) -> bool:
    """Whether a process with ``pid`` is still running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            output = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return str(pid) in output
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_state(state: RunState) -> None:
    """Persist the run state so a browser reload can re-attach to it."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def read_state() -> RunState | None:
    """The last launched run, or ``None``."""
    if not STATE_FILE.is_file():
        return None
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return RunState(
            run_name=str(raw["run_name"]),
            run_id=str(raw.get("run_id", "")),
            pid=int(raw["pid"]),
            log_path=str(raw["log_path"]),
            checkpoint_dir=str(raw.get("checkpoint_dir", "")),
            argv=[str(a) for a in raw.get("argv", [])],
            started_at=float(raw.get("started_at", 0.0)),
            finished_at=raw.get("finished_at"),
            returncode=raw.get("returncode"),
            stopped_by_user=bool(raw.get("stopped_by_user", False)),
        )
    except (OSError, ValueError, KeyError):
        return None


def clear_state() -> None:
    """Forget the last run (does not touch checkpoints or logs)."""
    if STATE_FILE.is_file():
        STATE_FILE.unlink()


def status(state: RunState | None, process: subprocess.Popen[bytes] | None) -> str:
    """One of ``running``, ``succeeded``, ``failed``, ``stopped``, ``idle``."""
    if state is None:
        return IDLE
    if process is not None:
        code = process.poll()
        if code is None:
            return RUNNING
        if state.stopped_by_user:
            return STOPPED
        return SUCCEEDED if code == 0 else FAILED
    if is_alive(state.pid):
        return RUNNING
    if state.stopped_by_user:
        return STOPPED
    if state.returncode is not None:
        return SUCCEEDED if state.returncode == 0 else FAILED
    return SUCCEEDED if _log_looks_complete(Path(state.log_path)) else FAILED


def _log_looks_complete(log_path: Path) -> bool:
    tail = read_log(log_path, lines=40)
    return "COMPANY RUN COMPLETE" in tail


def read_log(log_path: str | Path, *, lines: int = 400) -> str:
    """The last ``lines`` lines of a run log, tolerating a partially written file."""
    path = Path(log_path)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(could not read log: {exc})"
    return "\n".join(text.splitlines()[-lines:])


def read_progress(cfg: Any) -> Progress:
    """Parse the structured loss history this run has written so far."""
    from plantdx.company.train import LOSS_HISTORY

    path = Path(cfg.log_dir) / LOSS_HISTORY
    progress = Progress(total_epochs=float(cfg.optim.epochs))
    if not path.is_file():
        return progress
    history: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return progress
    if not history:
        return progress
    last = next((row for row in reversed(history) if "total_steps" in row), history[-1])
    progress.history = history
    progress.step = int(last.get("step", 0))
    progress.total_steps = int(last.get("total_steps", 0))
    progress.epoch = float(last.get("epoch", 0.0))
    progress.total_epochs = float(last.get("total_epochs", progress.total_epochs))
    progress.loss = last.get("loss")
    progress.learning_rate = next((r["learning_rate"] for r in reversed(history) if r.get("learning_rate") is not None), None)
    progress.validation_loss = next((r["validation_loss"] for r in reversed(history) if r.get("validation_loss") is not None), None)
    progress.elapsed_seconds = float(last.get("elapsed_seconds", 0.0))
    progress.memory = last.get("memory") or {}
    return progress


def failure_reason(log_path: str | Path) -> str:
    """A short, human-readable reason pulled from the tail of a failed log."""
    tail = read_log(log_path, lines=200).splitlines()
    markers = (
        "Error",
        "error:",
        "Traceback",
        "failed",
        "FAILED",
        "not ready",
        "CUDA out of memory",
    )
    hits = [line.strip() for line in tail if any(m in line for m in markers)]
    if not hits:
        return "The training process exited without a recognizable error message."
    return "\n".join(hits[-6:])


def last_stage(log_path: str | Path) -> str:
    """The last stage headline the run printed."""
    stages = (
        "Preparing dataset",
        "Running smoke test",
        "loading base model",
        "adapter attached",
        "Training",
        "Evaluating",
        "Adapter written",
    )
    found = "starting up"
    for line in read_log(log_path, lines=800).splitlines():
        for stage in stages:
            if stage in line:
                found = stage
    return found


def latest_checkpoint(checkpoint_dir: str | Path) -> str:
    """Name of the newest ``checkpoint-N`` directory, or a placeholder."""
    from plantdx.company.train import latest_checkpoint as _latest

    try:
        found = _latest(checkpoint_dir)
    except Exception:
        return "-"
    return found.name if found is not None else "none yet"


def gpu_sample() -> Any:
    """A live nvidia-smi reading for the progress panel."""
    from plantdx.company.environment import gpu_telemetry

    return gpu_telemetry()


def list_runs() -> list[dict[str, Any]]:
    """Every checkpoint directory that has produced something.

    Directories that are still empty are skipped: preflight creates the run
    directory to prove it is writable, long before any run exists, and an empty
    placeholder is not a result worth showing.
    """
    from plantdx.company.modeling import adapter_is_valid

    root = REPO_ROOT / "checkpoints"
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        ok, reason = adapter_is_valid(path)
        has_checkpoints = any(path.glob("checkpoint-*"))
        if not ok and not has_checkpoints and not any(path.iterdir()):
            continue
        runs.append(
            {
                "run_name": path.name,
                "path": str(path),
                "valid": ok,
                "reason": reason,
                "has_checkpoints": has_checkpoints,
                "modified": path.stat().st_mtime,
            }
        )
    return runs


def read_metrics(run_name: str, split: str, *, tuned: bool = True) -> Any:
    """Load one evaluation result for a run, or ``None`` if it has not been run."""
    from plantdx.company.evaluate import load_result

    tag = "finetuned" if tuned else "base"
    path = REPO_ROOT / "reports" / run_name / f"{split}_{tag}"
    try:
        return load_result(path)
    except Exception:
        return None


def activate(run_name: str, adapter_dir: str | Path, *, crop: str = "tomato") -> Path:
    """Record ``adapter_dir`` as the adapter the Diagnose page should serve."""
    import datetime as _dt

    from plantdx.company.registry import ActiveAdapter, write_active_adapter

    cfg = load_config()
    dataset = load_dataset(cfg)
    result = read_metrics(run_name, "validation")
    metrics = (
        {
            "accuracy": result.accuracy,
            "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
        }
        if result is not None
        else {}
    )
    entry = ActiveAdapter(
        crop=crop,
        run_name=run_name,
        adapter_path=str(adapter_dir),
        base_model=cfg.model.model_id,
        dataset=dataset.name,
        backend="transformers_cuda",
        classes=dataset.labels,
        metrics=metrics,
        activated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    return write_active_adapter(entry, cfg.base_dir)


def run_evaluation_subprocess(run_name: str, split: str, *, base: bool = False) -> list[str]:
    """The argument list for an evaluation run (launched the same way as training)."""
    argv = [
        sys.executable,
        "-m",
        "plantdx",
        "company",
        "evaluate",
        "--config",
        DEFAULT_CONFIG,
        "--split",
        split,
    ]
    if base:
        argv.append("--base")
    else:
        argv += ["--adapter", str(REPO_ROOT / "checkpoints" / run_name)]
    return argv
