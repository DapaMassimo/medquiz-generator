from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from medquiz_generator.domain.models import StructuredDocument
from medquiz_generator.entrypoints import cli
from tests.fakes.fake_container import FakeContainer, make_settings

runner = CliRunner()


@pytest.fixture
def fake_cli(tmp_path: Path, fixture_document: StructuredDocument) -> FakeContainer:
    container = FakeContainer(make_settings(tmp_path, plan_min_unit_chars=400), fixture_document)
    cli._state["container"] = container
    yield container  # type: ignore[misc]
    cli._state.clear()


def test_cli_stage_by_stage(tmp_path: Path, fake_cli: FakeContainer) -> None:
    pdf = tmp_path / "manuale.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    result = runner.invoke(cli.app, ["plan", "--total-questions", "5"])
    assert result.exit_code != 0  # no run yet

    result = runner.invoke(cli.app, ["parse", str(pdf)])
    assert result.exit_code == 0, result.output
    assert "parsed 6 pages" in result.output

    result = runner.invoke(cli.app, ["plan"])
    assert result.exit_code != 0  # --total-questions is mandatory

    result = runner.invoke(cli.app, ["plan", "--total-questions", "10"])
    assert result.exit_code == 0, result.output
    assert "10/10 questions" in result.output

    result = runner.invoke(cli.app, ["generate"])  # runs classify implicitly
    assert result.exit_code == 0, result.output
    assert "running classify first" in result.output
    assert "10 questions generated" in result.output

    result = runner.invoke(cli.app, ["verify"])
    assert result.exit_code == 0, result.output
    assert "Judge verdicts: pass=10" in result.output

    result = runner.invoke(cli.app, ["export", "--include-flagged"])
    assert result.exit_code == 0, result.output
    assert "xlsx:" in result.output

    result = runner.invoke(cli.app, ["eval"])
    assert result.exit_code == 0, result.output
    assert "Eval for run" in result.output

    result = runner.invoke(cli.app, ["runs"])
    assert result.exit_code == 0 and result.output.startswith("*")
    result = runner.invoke(cli.app, ["show"])
    assert result.exit_code == 0 and "Taxonomy" in result.output
    result = runner.invoke(cli.app, ["--json-logs", "report"])
    assert result.exit_code == 0 and "Cost per delivered question" in result.output


def test_cli_full_run_and_eval_compare(tmp_path: Path, fake_cli: FakeContainer) -> None:
    pdf = tmp_path / "manuale.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = runner.invoke(cli.app, ["run", str(pdf), "-n", "8"])
    assert result.exit_code == 0, result.output
    assert "Artefacts:" in result.output
    run_a = fake_cli.store.latest_run_id()
    assert run_a is not None
    runner.invoke(cli.app, ["eval"])
    result = runner.invoke(cli.app, ["run", str(pdf), "-n", "4", "--include-flagged"])
    assert result.exit_code == 0, result.output
    run_b = fake_cli.store.latest_run_id()
    result = runner.invoke(cli.app, ["eval", "compare", run_a, run_b])  # type: ignore[list-item]
    assert result.exit_code != 0  # run_b not evaluated yet
    runner.invoke(cli.app, ["eval", "--limit", "2"])
    result = runner.invoke(cli.app, ["eval", "compare", run_a, run_b])  # type: ignore[list-item]
    assert result.exit_code == 0, result.output
    assert "delta" in result.output


def test_cli_run_requires_total(tmp_path: Path, fake_cli: FakeContainer) -> None:
    pdf = tmp_path / "manuale.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = runner.invoke(cli.app, ["run", str(pdf)])
    assert result.exit_code != 0
    result = runner.invoke(cli.app, ["run", str(pdf), "-n", "0"])
    assert result.exit_code != 0
