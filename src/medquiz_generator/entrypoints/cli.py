"""Typer CLI — twin entry point of the Streamlit UI. Every command re-runs only its stage."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from medquiz_generator.application import artifacts
from medquiz_generator.application.eval import compare as compare_reports
from medquiz_generator.composition.container import Container
from medquiz_generator.domain.errors import MedQuizError
from medquiz_generator.domain.models import EvalReport, GenerationPlan, Taxonomy
from medquiz_generator.entrypoints.logsetup import configure_logging
from medquiz_generator.entrypoints.presentation import (
    render_comparison,
    render_eval,
    render_plan,
    render_report,
    render_taxonomy,
)

app = typer.Typer(
    help="medquiz-generator: systematic, verifiable SSM-style questions from medical PDFs.",
    no_args_is_help=True,
)
eval_app = typer.Typer(help="Evaluation harness (golden sets, run comparison).")
app.add_typer(eval_app, name="eval")

RunIdOption = Annotated[
    str | None, typer.Option("--run-id", help="Run to operate on (default: latest run).")
]
TotalOption = Annotated[
    int,
    typer.Option(
        "--total-questions",
        "-n",
        min=1,
        help="Total number of questions to generate (mandatory, decided per run).",
    ),
]

_state: dict[str, Container] = {}


def container() -> Container:
    if "container" not in _state:
        _state["container"] = Container()
    return _state["container"]


def resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = container().store.latest_run_id()
    if latest is None:
        raise typer.BadParameter("No run found: start with `medquiz parse <pdf>` or `medquiz run`.")
    return latest


def _guard[T](func: Callable[[], T]) -> T:
    try:
        return func()
    except MedQuizError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.callback()
def main(
    log_level: Annotated[str | None, typer.Option(help="Override MQ_LOG_LEVEL")] = None,
    json_logs: Annotated[bool, typer.Option("--json-logs", help="Structured JSON logs")] = False,
) -> None:
    settings = container().settings
    configure_logging(log_level or settings.log_level, json_logs=json_logs or settings.log_json)


@app.command()
def parse(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Native PDF")],
    run_id: RunIdOption = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the Docling cache")] = False,
) -> None:
    """Stage 1: extract the section tree from the PDF into runs/<run_id>/document.json."""
    c = container()
    rid = run_id or c.new_run_id()
    ctx = c.context(rid)

    def go() -> None:
        document = c.parse_stage().run(ctx, pdf, use_cache=not no_cache)
        typer.echo(
            f"run {rid}: parsed {document.page_count} pages, {len(document.sections)} sections "
            f"→ {c.store.run_dir(rid) / 'document.json'}"
        )

    _guard(go)


@app.command()
def plan(total_questions: TotalOption, run_id: RunIdOption = None) -> None:
    """Stage 2: allocate the requested total across sections (plan.json, hand-editable)."""
    c = container()
    ctx = c.context(resolve_run_id(run_id))

    def go() -> None:
        result = c.plan_stage().run(ctx, total_questions)
        typer.echo(render_plan(result))

    _guard(go)


@app.command()
def classify(run_id: RunIdOption = None) -> None:
    """Stage 3: extract the taxonomy (subjects → topics → sections) from the document."""
    c = container()
    ctx = c.context(resolve_run_id(run_id))

    def go() -> None:
        taxonomy = c.classify_stage(ctx).run(ctx)
        typer.echo(render_taxonomy(taxonomy))

    _guard(go)


@app.command()
def generate(
    run_id: RunIdOption = None,
    unit: Annotated[
        list[str] | None, typer.Option("--unit", help="Only these plan unit ids")
    ] = None,
) -> None:
    """Stage 4: generate questions per plan unit (runs classify first if needed)."""
    c = container()
    ctx = c.context(resolve_run_id(run_id))

    def go() -> None:
        if not c.store.exists(ctx.run_id, artifacts.TAXONOMY):
            typer.echo("No taxonomy yet: running classify first.")
            c.classify_stage(ctx).run(ctx)
        items = c.generate_stage(ctx).run(ctx, unit_ids=unit)
        report = c.report_stage().run(ctx)
        typer.echo(f"run {ctx.run_id}: {len(items)} questions generated")
        typer.echo(render_report(report))

    _guard(go)


@app.command()
def verify(run_id: RunIdOption = None) -> None:
    """Stage 5: deterministic checks + independent LLM judge."""
    c = container()
    ctx = c.context(resolve_run_id(run_id))

    def go() -> None:
        c.verify_stage(ctx).run(ctx)
        typer.echo(render_report(c.report_stage().run(ctx)))

    _guard(go)


@app.command()
def dedup(run_id: RunIdOption = None) -> None:
    """Stage 6: flag near-duplicate questions via embedding similarity."""
    c = container()
    ctx = c.context(resolve_run_id(run_id))

    def go() -> None:
        c.dedup_stage(ctx).run(ctx)
        typer.echo(render_report(c.report_stage().run(ctx)))

    _guard(go)


@app.command()
def export(
    run_id: RunIdOption = None,
    include_flagged: Annotated[
        bool | None, typer.Option("--include-flagged/--no-flagged", help="Add flagged items")
    ] = None,
    approved_only: Annotated[
        bool, typer.Option("--approved-only", help="Only items approved in the review UI")
    ] = False,
) -> None:
    """Stage 7: write JSON + XLSX under runs/<run_id>/export/."""
    c = container()
    ctx = c.context(resolve_run_id(run_id))

    def go() -> None:
        if not c.store.exists(ctx.run_id, artifacts.QUESTIONS_DEDUPED) and c.store.exists(
            ctx.run_id, artifacts.QUESTIONS_VERIFIED
        ):
            c.dedup_stage(ctx).run(ctx)
        written = c.export_stage().run(
            ctx, include_flagged=include_flagged, approved_only=approved_only
        )
        c.report_stage().run(ctx)
        for fmt, path in written.items():
            typer.echo(f"{fmt}: {path}")

    _guard(go)


@app.command()
def report(run_id: RunIdOption = None) -> None:
    """Rebuild and print the run report (coverage, verdicts, costs, timings)."""
    c = container()
    ctx = c.context(resolve_run_id(run_id))
    _guard(lambda: typer.echo(render_report(c.report_stage().run(ctx))))


@app.command()
def runs() -> None:
    """List runs (latest marked with *)."""
    c = container()
    latest = c.store.latest_run_id()
    for rid in c.store.list_runs():
        typer.echo(f"{'*' if rid == latest else ' '} {rid}")


@app.command()
def show(run_id: RunIdOption = None) -> None:
    """Print plan and taxonomy of a run."""
    c = container()
    rid = resolve_run_id(run_id)
    if c.store.exists(rid, artifacts.PLAN):
        typer.echo(render_plan(c.store.load(rid, artifacts.PLAN, GenerationPlan)))
    if c.store.exists(rid, artifacts.TAXONOMY):
        typer.echo(render_taxonomy(c.store.load(rid, artifacts.TAXONOMY, Taxonomy)))


@app.command()
def run(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Native PDF")],
    total_questions: TotalOption,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    include_flagged: Annotated[bool | None, typer.Option("--include-flagged/--no-flagged")] = None,
) -> None:
    """Full pipeline: parse → plan → classify → generate → verify → dedup → export → report."""
    c = container()
    rid = c.new_run_id()
    ctx = c.context(rid)

    def go() -> None:
        result = c.pipeline(ctx).run(
            ctx, pdf, total_questions, use_cache=not no_cache, include_flagged=include_flagged
        )
        typer.echo(render_report(result))
        typer.echo(f"Artefacts: {c.store.run_dir(rid)}")

    _guard(go)


@eval_app.callback(invoke_without_command=True)
def eval_run(
    ctx_typer: typer.Context,
    run_id: RunIdOption = None,
    limit: Annotated[int | None, typer.Option(help="Score at most N items")] = None,
    include_flagged: Annotated[bool, typer.Option("--include-flagged")] = False,
) -> None:
    """Score a run's questions against the golden sets (style + regression)."""
    if ctx_typer.invoked_subcommand is not None:
        return
    c = container()
    ctx = c.context(resolve_run_id(run_id))

    def go() -> None:
        result = c.eval_stage(ctx, include_flagged=include_flagged).run(ctx, limit=limit)
        typer.echo(render_eval(result))
        c.report_stage().run(ctx)

    _guard(go)


@eval_app.command("compare")
def eval_compare(run_a: str, run_b: str) -> None:
    """Delta per rubric dimension between two evaluated runs."""
    c = container()

    def go() -> None:
        reports = []
        for rid in (run_a, run_b):
            if not c.store.exists(rid, artifacts.EVAL_REPORT):
                raise MedQuizError(f"Run {rid} has no eval_report.json: run `medquiz eval` first.")
            reports.append(c.store.load(rid, artifacts.EVAL_REPORT, EvalReport))
        typer.echo(render_comparison(compare_reports(reports[0], reports[1])))

    _guard(go)


@app.command()
def serve(port: Annotated[int, typer.Option(help="Streamlit port")] = 8501) -> None:
    """Launch the Streamlit review app."""
    script = Path(__file__).with_name("streamlit_app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)]
    raise typer.Exit(code=subprocess.call(command))


if __name__ == "__main__":  # pragma: no cover
    app()
