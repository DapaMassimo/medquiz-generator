"""Stage — figures: triage of the figures extracted by parse (opt-in, ``MQ_IMAGES_ENABLED``).

Parse saves every figure of the PDF as a PNG under ``runs/<id>/assets/figures/`` and records
it in the document (page, section, caption). This stage asks a vision model to classify each
figure — clinical image (ECG, radiology, histology, photo…) versus diagram, chart, table image
or icon — and to say whether it could support an exam question. Only ``question_worthy``
figures are ever used to generate image questions; schematics never are.
"""

from __future__ import annotations

from pathlib import Path

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.models import FigureCatalog, FigureRef, Stage, StructuredDocument
from medquiz_generator.domain.ports import FigureTriage


class FiguresStage:
    def __init__(self, triage: FigureTriage | None) -> None:
        self._triage = triage

    def run(self, ctx: RunContext) -> FigureCatalog:
        with ctx.stage(Stage.FIGURES):
            document = ctx.require(artifacts.DOCUMENT, StructuredDocument)
            run_dir = ctx.store.run_dir(ctx.run_id)
            previous: dict[str, FigureRef] = {}
            if ctx.store.exists(ctx.run_id, artifacts.FIGURES):
                previous = {
                    f.id: f
                    for f in ctx.store.load(ctx.run_id, artifacts.FIGURES, FigureCatalog).figures
                }
            figures: list[FigureRef] = []
            for figure in document.figures:
                if figure.id in previous and previous[figure.id].kind != "unclassified":
                    # Reuse the assessment, but take placement (section, caption) from the
                    # current document: a re-parse may have moved the figure.
                    done = previous[figure.id]
                    figures.append(
                        figure.model_copy(
                            update={
                                "kind": done.kind,
                                "description": done.description,
                                "question_worthy": done.question_worthy,
                                "triage_reason": done.triage_reason,
                            }
                        )
                    )
                    continue
                image_path = run_dir / figure.path
                if self._triage is None or not image_path.exists():
                    figures.append(figure)
                    continue
                ctx.tracker.set_context(unit_id=figure.section_id, item_id=figure.id)
                assessment = self._triage.classify(figure, image_path.read_bytes())
                figures.append(
                    figure.model_copy(
                        update={
                            "kind": assessment.kind,
                            "description": assessment.description,
                            "question_worthy": assessment.question_worthy,
                            "triage_reason": assessment.reason,
                        }
                    )
                )
                catalog = FigureCatalog(run_id=ctx.run_id, figures=list(figures))
                ctx.store.save(ctx.run_id, artifacts.FIGURES, catalog)
            ctx.tracker.set_context()
            catalog = FigureCatalog(run_id=ctx.run_id, figures=figures)
            ctx.store.save(ctx.run_id, artifacts.FIGURES, catalog)
            ctx.log(
                "figures triaged",
                figures=len(figures),
                question_worthy=sum(1 for f in figures if f.question_worthy),
                kinds={
                    k: sum(1 for f in figures if f.kind == k) for k in {f.kind for f in figures}
                },
                model=self._triage.model_name if self._triage else None,
            )
        return catalog


def figure_image_path(run_dir: Path, figure: FigureRef) -> Path:
    return run_dir / figure.path
