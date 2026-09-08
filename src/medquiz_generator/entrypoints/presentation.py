"""Plain-text rendering of artefacts for the CLI (and reused by the UI where handy)."""

from __future__ import annotations

from medquiz_generator.domain.models import (
    EVAL_DIMENSIONS,
    EvalComparison,
    EvalReport,
    GenerationPlan,
    RunReport,
    Taxonomy,
)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def render_plan(plan: GenerationPlan) -> str:
    rows = [
        [
            u.id,
            u.title[:48],
            f"{u.page_start}-{u.page_end}",
            str(u.char_count),
            f"{u.weight:.1%}",
            str(u.questions),
            " ".join(f"{k[:3]}={v}" for k, v in u.styles.items()),
            u.note,
        ]
        for u in plan.units
    ]
    mode = "auto" if plan.min_unit_chars_auto else "fixed"
    header = (
        f"Plan for run {plan.run_id}: {plan.planned_questions}/{plan.total_questions} "
        f"questions over {len(plan.units)} units, merge threshold {plan.min_unit_chars} chars "
        f"({mode}); {len(plan.skipped_section_ids)} sections skipped as noise\n"
    )
    return header + _table(
        ["unit", "title", "pages", "chars", "weight", "q", "styles", "note"], rows
    )


def render_taxonomy(taxonomy: Taxonomy) -> str:
    lines = [f"Taxonomy ({taxonomy.source}) — {len(taxonomy.subjects)} subjects"]
    for subject in taxonomy.subjects:
        lines.append(f"- {subject.name}")
        for topic in subject.topics:
            lines.append(f"    - {topic.name}  [{', '.join(topic.section_ids)}]")
    return "\n".join(lines)


def render_report(report: RunReport) -> str:
    total = report.usage_total
    lines = [
        f"Run {report.run_id}",
        f"Source: {report.source_path}  (hash {report.document_hash[:12]})",
        f"Requested: {report.total_questions_requested}  | units planned: "
        f"{report.units_planned}, covered: {report.units_covered}  | sections planned: "
        f"{report.sections_total}, with delivered questions: {report.sections_covered}",
        f"Generated: {report.generated}  verified: {report.verified}  flagged: "
        f"{report.flagged}  discarded: {report.discarded}  exported: {report.exported}  "
        f"duplicates: {report.duplicates}",
    ]
    if report.styles_requested or report.styles_delivered:
        lines.append(
            "Styles requested: "
            + ", ".join(f"{k}={v}" for k, v in report.styles_requested.items())
            + " | delivered: "
            + ", ".join(f"{k}={v}" for k, v in report.styles_delivered.items())
        )
    if report.deterministic_failures:
        lines.append(
            "Deterministic failures: "
            + ", ".join(f"{k}={v}" for k, v in sorted(report.deterministic_failures.items()))
        )
    if report.judge_verdicts:
        lines.append(
            "Judge verdicts: "
            + ", ".join(f"{k}={v}" for k, v in sorted(report.judge_verdicts.items()))
        )
    if report.discard_reasons:
        lines.append("Discard reasons:")
        lines += [f"  - {v} × {k}" for k, v in report.discard_reasons.items()]
    if report.flag_reasons:
        lines.append("Flag reasons:")
        lines += [f"  - {v} × {k}" for k, v in report.flag_reasons.items()]
    if report.stage_timings:
        lines.append("Stage timings:")
        lines.append(
            _table(
                ["stage", "status", "seconds", "error"],
                [
                    [t.stage, t.status, f"{t.duration_s:.2f}", (t.error or "")[:60]]
                    for t in report.stage_timings
                ],
            )
        )
    if report.usage_by_stage:
        lines.append("Usage by stage:")
        lines.append(
            _table(
                ["stage", "model", "calls", "tokens in", "tokens out", "cost USD"],
                [
                    [
                        stage,
                        report.models.get(stage, ""),
                        str(u.calls),
                        str(u.input_tokens),
                        str(u.output_tokens),
                        f"{u.cost_usd:.4f}",
                    ]
                    for stage, u in report.usage_by_stage.items()
                ],
            )
        )
    lines.append(
        f"Total: {total.calls} calls, {total.input_tokens} in / {total.output_tokens} out "
        f"tokens, {total.cost_usd:.4f} USD"
    )
    if report.cost_per_delivered_question_usd is not None:
        lines.append(
            f"Cost per delivered question: {report.cost_per_delivered_question_usd:.4f} USD"
        )
    lines.append(f"Total duration: {report.total_duration_s:.1f} s")
    return "\n".join(lines)


def render_eval(report: EvalReport) -> str:
    lines = [
        f"Eval for run {report.run_id} — judge {report.judge_model}, "
        f"{len(report.items)} items, golden style={report.golden_style_count}, "
        f"regression={report.golden_regression_count}",
        _table(
            ["dimension", "avg (1-5)"],
            [[dim, f"{report.averages.get(dim, 0.0):.2f}"] for dim in (*EVAL_DIMENSIONS, "mean")],
        ),
    ]
    return "\n".join(lines)


def render_comparison(comparison: EvalComparison) -> str:
    rows = [
        [
            dim,
            f"{comparison.averages_a.get(dim, 0.0):.2f}",
            f"{comparison.averages_b.get(dim, 0.0):.2f}",
            f"{comparison.delta.get(dim, 0.0):+.2f}",
        ]
        for dim in (*EVAL_DIMENSIONS, "mean")
    ]
    return f"{comparison.run_a} → {comparison.run_b}\n" + _table(
        ["dimension", "run A", "run B", "delta"], rows
    )
