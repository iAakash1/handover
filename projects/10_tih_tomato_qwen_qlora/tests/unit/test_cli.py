"""CLI surface tests (Milestone 1): parsing, version, help, stubbed dispatch."""

from __future__ import annotations

import pytest

from plantdx.cli import build_parser, main


@pytest.mark.unit
def test_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "plantdx"


@pytest.mark.unit
def test_version_action(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert "plantdx" in capsys.readouterr().out


@pytest.mark.unit
def test_no_command_prints_help_and_succeeds() -> None:
    assert main([]) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        ["dataset", "build"],
        ["dataset", "convert", "--model", "qwen3_vl"],
        ["qa", "sample"],
    ],
)
def test_stubbed_commands_exit_two(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    # Stubbed stages return exit code 2 with a milestone message (not a traceback).
    assert main(argv) == 2
    assert "Milestone" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        ["prepare-training", "--config", "configs/train/qwen25vl_tomato.yaml"],
        ["train", "--config", "configs/train/qwen25vl_tomato.yaml", "--dry-run"],
        ["infer", "--image", "leaf.JPG"],
        ["evaluate", "--stage", "analyze"],
    ],
)
def test_training_commands_parse(argv: list[str]) -> None:
    # The M6/M7 training + evaluation commands are implemented (not stubs) and parse cleanly.
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        ["concepts", "--stats-only"],
        ["templates", "--validate-only"],
        ["generate", "--stats-only"],
        ["generate", "--condition", "tomato_early_blight", "--validate-only"],
        ["validate", "--crop", "mango"],
    ],
)
def test_new_commands_parse(argv: list[str]) -> None:
    # The M3 language-layer commands parse into the expected top-level command.
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]
