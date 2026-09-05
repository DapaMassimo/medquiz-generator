"""Stage 2 — plan: StructuredDocument + total questions → GenerationPlan.

Systematic coverage: every substantial section becomes (part of) a unit and the requested
total is split proportionally to text length with a minimum of one question per unit. The
total is a mandatory input — there is deliberately no default anywhere in the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.domain.errors import PlanningError
from medquiz_generator.domain.models import (
    GenerationPlan,
    PlanUnit,
    Section,
    Stage,
    StructuredDocument,
)

# Sections shorter than this are pure headings / noise and are never planned on their own.
NOISE_CHARS = 120
# Sections that are not teaching content (bibliographies, indexes…) are never planned.
NON_CONTENT_TITLE = re.compile(
    r"^\s*(bibliografia|riferimenti bibliografici|references|letture consigliate|indice"
    r"|sommario|ringraziamenti|glossario|abbreviazioni|acknowledg)",
    re.IGNORECASE,
)


def is_non_content(section: Section) -> bool:
    return bool(NON_CONTENT_TITLE.match(section.title))


@dataclass
class _Group:
    sections: list[Section] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(s.char_count for s in self.sections)

    @property
    def parent_id(self) -> str | None:
        return self.sections[0].parent_id if self.sections else None


def _group_sections(sections: list[Section], min_unit_chars: int) -> list[_Group]:
    """Big sections stand alone; consecutive small siblings are merged until they reach
    ``min_unit_chars``. Returns groups in document order."""
    groups: list[_Group] = []
    pending = _Group()

    def flush() -> None:
        nonlocal pending
        if pending.sections:
            groups.append(pending)
        pending = _Group()

    for section in sections:
        if section.char_count == 0 or is_non_content(section):
            continue
        if section.char_count >= min_unit_chars:
            flush()
            groups.append(_Group([section]))
            continue
        if pending.sections and pending.parent_id != section.parent_id:
            flush()
        pending.sections.append(section)
        if pending.chars >= min_unit_chars:
            flush()
    flush()
    return groups


def allocate(total: int, weights: list[float], minimum: int = 1) -> list[int]:
    """Largest-remainder apportionment with a per-unit minimum.

    Guarantees ``sum(result) == total``. When ``total`` is smaller than the number of units,
    the heaviest units get one question each and the others get zero.
    """
    n = len(weights)
    if n == 0:
        return []
    if total < n * minimum:
        order = sorted(range(n), key=lambda i: (-weights[i], i))
        result = [0] * n
        for i in order[:total]:
            result[i] = minimum
        return result
    remaining = total - n * minimum
    weight_sum = sum(weights) or 1.0
    raw = [remaining * w / weight_sum for w in weights]
    result = [minimum + int(r) for r in raw]
    leftover = total - sum(result)
    by_remainder = sorted(range(n), key=lambda i: (-(raw[i] - int(raw[i])), -weights[i], i))
    for i in by_remainder[:leftover]:
        result[i] += 1
    return result


def build_plan(
    document: StructuredDocument,
    run_id: str,
    total_questions: int,
    *,
    min_unit_chars: int = 600,
) -> GenerationPlan:
    if total_questions < 1:
        raise PlanningError("total_questions must be >= 1 (it is a mandatory run parameter).")
    groups = _group_sections(document.sections, min_unit_chars)
    substantial = [g for g in groups if g.chars >= NOISE_CHARS]
    skipped = [s.id for g in groups if g.chars < NOISE_CHARS for s in g.sections]
    skipped += [s.id for s in document.sections if s.char_count == 0 or is_non_content(s)]
    if not substantial:
        raise PlanningError("The document has no section with enough text to plan questions.")

    total_chars = sum(g.chars for g in substantial)
    weights = [g.chars / total_chars for g in substantial]
    budget = allocate(total_questions, weights)

    units: list[PlanUnit] = []
    for index, (group, weight, questions) in enumerate(
        zip(substantial, weights, budget, strict=True), start=1
    ):
        first = group.sections[0]
        title = (
            first.title
            if len(group.sections) == 1
            else f"{first.title} (+{len(group.sections) - 1})"
        )
        note = ""
        if len(group.sections) > 1:
            note = "merged small sibling sections"
        if questions == 0:
            note = (note + "; " if note else "") + "no budget left (total < units)"
        units.append(
            PlanUnit(
                id=f"u{index:03d}",
                title=title,
                section_ids=[s.id for s in group.sections],
                page_start=min(s.page_start for s in group.sections),
                page_end=max(s.page_end for s in group.sections),
                char_count=group.chars,
                weight=weight,
                questions=questions,
                note=note,
            )
        )
    return GenerationPlan(
        run_id=run_id,
        document_hash=document.content_hash,
        total_questions=total_questions,
        min_unit_chars=min_unit_chars,
        units=units,
        skipped_section_ids=skipped,
    )


class PlanStage:
    def __init__(self, min_unit_chars: int = 600) -> None:
        self._min_unit_chars = min_unit_chars

    def run(self, ctx: RunContext, total_questions: int) -> GenerationPlan:
        with ctx.stage(Stage.PLAN):
            document = ctx.require(artifacts.DOCUMENT, StructuredDocument)
            plan = build_plan(
                document, ctx.run_id, total_questions, min_unit_chars=self._min_unit_chars
            )
            ctx.store.save(ctx.run_id, artifacts.PLAN, plan)
            ctx.update_meta(total_questions=total_questions)
            ctx.log(
                "plan built",
                units=len(plan.units),
                planned=plan.planned_questions,
                skipped_sections=len(plan.skipped_section_ids),
            )
        return plan
