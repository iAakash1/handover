"""The boundary that keeps a full training run from starting on its own.

These tests are the enforcement of one rule: only an explicit click on RUN
TRAINING (or an explicit ``plantdx company train`` invocation) may begin a full
run. Rendering the page, preflight, dataset preparation, and the smoke test must
all stop by themselves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VIEW = REPO_ROOT / "app" / "training_view.py"
SERVICE = REPO_ROOT / "app" / "training_service.py"
COMPANY_CLI = REPO_ROOT / "src" / "plantdx" / "company" / "cli.py"
SMOKE = REPO_ROOT / "src" / "plantdx" / "company" / "smoke.py"
PREFLIGHT = REPO_ROOT / "src" / "plantdx" / "company" / "preflight.py"


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    """Every call to ``name`` in ``tree``, matched on the attribute or bare name."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        matched = (isinstance(func, ast.Attribute) and func.attr == name) or (
            isinstance(func, ast.Name) and func.id == name
        )
        if matched:
            found.append(node)
    return found


def _enclosing_function(tree: ast.AST, target: ast.Call) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(child is target for child in ast.walk(node)):
            return node.name
    return "<module>"


@pytest.mark.unit
def test_only_one_place_in_the_app_starts_training() -> None:
    """A second call site would be a second way to start a multi-hour run."""
    tree = ast.parse(VIEW.read_text(encoding="utf-8"))
    calls = _calls(tree, "start_training")
    assert len(calls) == 1, f"expected exactly one start_training call, found {len(calls)}"
    assert _enclosing_function(tree, calls[0]) == "_launch"


@pytest.mark.unit
def test_the_launch_helper_is_only_reached_from_button_branches() -> None:
    """``_launch`` must be called only from inside ``if st...button(...)`` blocks."""
    source = VIEW.read_text(encoding="utf-8")
    tree = ast.parse(source)
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launch"
    ]
    assert launch_calls, "the page must have a launch path"

    guarded = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.get_source_segment(source, node.test) or ""
        if "button(" not in test_src:
            continue
        for child in ast.walk(node):
            if child in launch_calls:
                guarded += 1
    assert guarded == len(launch_calls), "every _launch call must sit under a button test"


@pytest.mark.unit
def test_rendering_the_page_never_calls_start_training() -> None:
    """The render entry point must not reach the launcher without a click."""
    tree = ast.parse(VIEW.read_text(encoding="utf-8"))
    render = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_training"
    )
    called = {
        node.func.id
        for node in ast.walk(render)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_launch" not in called
    assert "start_training" not in called


@pytest.mark.unit
def test_preflight_never_starts_training_or_a_smoke_run() -> None:
    """Preflight reads the recorded smoke result; it must not run one."""
    source = PREFLIGHT.read_text(encoding="utf-8")
    assert "run_smoke_test" not in source
    assert "run_training" not in source
    assert "last_smoke" in source


@pytest.mark.unit
def test_the_smoke_test_never_calls_the_full_training_command() -> None:
    """The smoke test bounds itself with max_steps and a throwaway directory."""
    source = SMOKE.read_text(encoding="utf-8")
    assert "stop_after_steps=1" in source
    assert "max_steps=2" in source
    assert "resume=True" in source
    assert "output_dir=adapter_dir" in source
    assert "report_dir=scratch" in source
    assert "shutil.rmtree" in source


@pytest.mark.unit
def test_the_train_command_does_not_chain_from_a_smoke_test() -> None:
    """``company train`` must not run a smoke test and roll into a full run."""
    source = COMPANY_CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_train"
    )
    train_source = ast.get_source_segment(source, train) or ""
    assert "run_smoke_test" not in train_source
    assert "--skip-smoke" not in source


@pytest.mark.unit
def test_the_launch_command_carries_no_smoke_flag() -> None:
    """The button's command must be a plain training invocation."""
    from app import training_service as svc

    argv = svc.build_train_argv({})
    assert "--skip-smoke" not in argv
    assert "smoke" not in argv


@pytest.mark.unit
def test_a_second_launch_is_refused_while_one_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import training_service as svc

    monkeypatch.setattr(svc, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(svc, "LOCK_FILE", tmp_path / "train.lock")
    monkeypatch.setattr(svc, "is_alive", lambda pid: pid == 999)

    svc.write_state(
        svc.RunState(
            run_name="tih", pid=999, log_path=str(tmp_path / "t.log"), argv=[], started_at=0.0
        )
    )
    with pytest.raises(RuntimeError, match="already active"):
        svc.start_training({})


@pytest.mark.unit
def test_the_lock_refuses_a_concurrent_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two Streamlit reruns racing to start must not both win."""
    import json

    from app import training_service as svc

    lock = tmp_path / "train.lock"
    monkeypatch.setattr(svc, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(svc, "LOCK_FILE", lock)
    monkeypatch.setattr(svc, "is_alive", lambda pid: pid == 4242)

    lock.write_text(json.dumps({"pid": 4242, "run_id": "x"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="another training launch is in progress"):
        svc._acquire_lock()


@pytest.mark.unit
def test_a_stale_lock_is_reclaimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from app import training_service as svc

    lock = tmp_path / "train.lock"
    monkeypatch.setattr(svc, "LOCK_FILE", lock)
    monkeypatch.setattr(svc, "is_alive", lambda pid: False)

    lock.write_text(json.dumps({"pid": 123456, "run_id": "gone"}), encoding="utf-8")
    assert svc._acquire_lock() == lock
    assert lock.is_file()


@pytest.mark.unit
def test_stopping_marks_the_run_and_keeps_the_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import training_service as svc

    monkeypatch.setattr(svc, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(svc, "LOCK_FILE", tmp_path / "train.lock")
    monkeypatch.setattr(svc, "is_alive", lambda pid: False)

    checkpoint = tmp_path / "checkpoints" / "tih"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"kept")

    state = svc.RunState(
        run_name="tih",
        pid=-1,
        log_path=str(tmp_path / "t.log"),
        argv=[],
        started_at=0.0,
        checkpoint_dir=str(checkpoint),
    )
    svc.stop_training(state)
    assert (checkpoint / "adapter_model.safetensors").read_bytes() == b"kept"


@pytest.mark.unit
def test_a_user_stopped_run_reports_stopped_not_failed(tmp_path: Path) -> None:
    from app import training_service as svc

    log = tmp_path / "t.log"
    log.write_text("Preparing dataset...\n", encoding="utf-8")
    state = svc.RunState(
        run_name="tih",
        pid=-1,
        log_path=str(log),
        argv=[],
        started_at=0.0,
        stopped_by_user=True,
    )
    assert svc.status(state, None) == svc.STOPPED


@pytest.mark.unit
def test_the_service_never_imports_streamlit() -> None:
    """Training logic must stay usable outside the app."""
    source = SERVICE.read_text(encoding="utf-8")
    assert "import streamlit" not in source
