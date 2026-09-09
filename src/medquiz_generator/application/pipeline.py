"""End-to-end orchestration: parse → plan → classify → generate → verify → dedup → export.

Each stage persists its artefact and can be re-run alone; ``Pipeline.run`` simply chains
them and refreshes the run report after every stage so a partial run still has a report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medquiz_generator.application.classify import ClassifyStage
from medquiz_generator.application.context import RunContext
from medquiz_generator.application.dedup import DedupStage
from medquiz_generator.application.export import ExportStage
from medquiz_generator.application.figures import FiguresStage
from medquiz_generator.application.generate import GenerateStage
from medquiz_generator.application.parse import ParseStage
from medquiz_generator.application.plan import PlanStage
from medquiz_generator.application.report import ReportStage
from medquiz_generator.application.verify import VerifyStage
from medquiz_generator.domain.models import RunReport


@dataclass
class Stages:
    parse: ParseStage
    plan: PlanStage
    classify: ClassifyStage
    generate: GenerateStage
    verify: VerifyStage
    dedup: DedupStage
    export: ExportStage
    report: ReportStage
    figures: FiguresStage | None = None


class Pipeline:
    def __init__(self, stages: Stages) -> None:
        self._stages = stages

    def run(
        self,
        ctx: RunContext,
        pdf_path: Path,
        total_questions: int,
        *,
        use_cache: bool = True,
        include_flagged: bool | None = None,
        min_unit_chars: int | None = None,
        style_mix: dict[str, int] | None = None,
    ) -> RunReport:
        stages = self._stages
        stages.parse.run(ctx, pdf_path, use_cache=use_cache)
        if stages.figures is not None:
            stages.figures.run(ctx)
        stages.plan.run(ctx, total_questions, min_unit_chars=min_unit_chars, style_mix=style_mix)
        stages.classify.run(ctx)
        stages.report.run(ctx)
        stages.generate.run(ctx)
        stages.report.run(ctx)
        stages.verify.run(ctx)
        stages.report.run(ctx)
        stages.dedup.run(ctx)
        stages.export.run(ctx, include_flagged=include_flagged)
        return stages.report.run(ctx)
