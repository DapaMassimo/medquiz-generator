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
    QuestionStyle,
    Section,
    Stage,
    StructuredDocument,
)

DEFAULT_STYLE_MIX: dict[str, int] = {
    QuestionStyle.VIGNETTA.value: 70,
    QuestionStyle.NOZIONISTICA.value: 30,
}

# Sections shorter than this are pure headings / noise and are never planned on their own.
NOISE_CHARS = 120
# Bounds for the automatic merge threshold: never merge below the floor (units would be too
# thin to write a question on) and never above the cap (~2k tokens of source per call).
AUTO_MIN_UNIT_CHARS_FLOOR = 600
AUTO_MIN_UNIT_CHARS_CAP = 8000
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


def _substantial_groups(sections: list[Section], min_unit_chars: int) -> list[_Group]:
    return [g for g in _group_sections(sections, min_unit_chars) if g.chars >= NOISE_CHARS]


def choose_min_unit_chars(
    document: StructuredDocument,
    total_questions: int,
    *,
    floor: int = AUTO_MIN_UNIT_CHARS_FLOOR,
    cap: int = AUTO_MIN_UNIT_CHARS_CAP,
) -> int:
    """Pick the merge threshold so that every unit can receive at least one question.

    Walks a geometric grid of thresholds from ``floor`` to ``cap`` and returns the smallest one
    whose unit count does not exceed ``total_questions`` (finer units are better as long as
    none is left without budget). If even ``cap`` yields more units than questions, ``cap`` is
    returned and some units will get zero: the document is simply larger than the budget.
    """
    candidates: list[int] = []
    value = float(floor)
    while value < cap:
        candidates.append(round(value))
        value *= 1.25
    candidates.append(cap)
    for threshold in candidates:
        if len(_substantial_groups(document.sections, threshold)) <= total_questions:
            return threshold
    return cap


def parse_style_mix(spec: str) -> dict[str, int]:
    """Parse "vignetta=70,nozionistica=30" into percentages; unknown styles are rejected."""
    if not spec.strip():
        return dict(DEFAULT_STYLE_MIX)
    mix: dict[str, int] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        try:
            name, value = part.split("=")
            mix[QuestionStyle(name.strip().lower()).value] = int(value)
        except ValueError as exc:
            raise PlanningError(
                f"Invalid style mix {spec!r}: use e.g. 'vignetta=70,nozionistica=30' with "
                f"styles among {[s.value for s in QuestionStyle]}."
            ) from exc
    if sum(mix.values()) <= 0:
        raise PlanningError(f"Invalid style mix {spec!r}: shares must sum to a positive number.")
    return mix


def allocate_styles(questions_per_unit: list[int], mix: dict[str, int]) -> list[dict[str, int]]:
    """Split every unit's budget among styles so that the run-level shares match ``mix``.

    Style totals are apportioned with the largest-remainder rule; units are then filled in
    order, so the mix holds globally while a unit with one question gets one style only.
    """
    total = sum(questions_per_unit)
    names = list(mix)
    totals = allocate(total, [float(mix[n]) for n in names], minimum=0)
    remaining = dict(zip(names, totals, strict=True))
    result: list[dict[str, int]] = []
    for budget in questions_per_unit:
        counts = dict.fromkeys(names, 0)
        for _ in range(budget):
            # Take the style that is furthest below its target share among those left.
            name = max(names, key=lambda n: remaining[n])
            if remaining[name] <= 0:
                name = names[0]
            counts[name] += 1
            remaining[name] -= 1
        result.append({n: c for n, c in counts.items() if c})
    return result


def build_plan(
    document: StructuredDocument,
    run_id: str,
    total_questions: int,
    *,
    min_unit_chars: int | None = None,
    style_mix: dict[str, int] | None = None,
) -> GenerationPlan:
    """``min_unit_chars=None`` selects the threshold automatically (see choose_min_unit_chars)."""
    if total_questions < 1:
        raise PlanningError("total_questions must be >= 1 (it is a mandatory run parameter).")
    mix = dict(style_mix) if style_mix else dict(DEFAULT_STYLE_MIX)
    auto = min_unit_chars is None
    if min_unit_chars is None:
        min_unit_chars = choose_min_unit_chars(document, total_questions)
    groups = _group_sections(document.sections, min_unit_chars)
    substantial = [g for g in groups if g.chars >= NOISE_CHARS]
    skipped = [s.id for g in groups if g.chars < NOISE_CHARS for s in g.sections]
    skipped += [s.id for s in document.sections if s.char_count == 0 or is_non_content(s)]
    if not substantial:
        raise PlanningError("The document has no section with enough text to plan questions.")

    total_chars = sum(g.chars for g in substantial)
    weights = [g.chars / total_chars for g in substantial]
    budget = allocate(total_questions, weights)

    styles_per_unit = allocate_styles(budget, mix)
    units: list[PlanUnit] = []
    for index, (group, weight, questions, styles) in enumerate(
        zip(substantial, weights, budget, styles_per_unit, strict=True), start=1
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
                styles=styles,
                note=note,
            )
        )
    return GenerationPlan(
        run_id=run_id,
        document_hash=document.content_hash,
        total_questions=total_questions,
        min_unit_chars=min_unit_chars,
        min_unit_chars_auto=auto,
        style_mix=mix,
        units=units,
        skipped_section_ids=skipped,
    )


class PlanStage:
    def __init__(
        self, min_unit_chars: int | None = None, style_mix: dict[str, int] | None = None
    ) -> None:
        """``None`` = choose the merge threshold automatically from the question budget."""
        self._min_unit_chars = min_unit_chars
        self._style_mix = style_mix

    def run(
        self,
        ctx: RunContext,
        total_questions: int,
        *,
        min_unit_chars: int | None = None,
        style_mix: dict[str, int] | None = None,
    ) -> GenerationPlan:
        with ctx.stage(Stage.PLAN):
            document = ctx.require(artifacts.DOCUMENT, StructuredDocument)
            threshold = min_unit_chars if min_unit_chars is not None else self._min_unit_chars
            plan = build_plan(
                document,
                ctx.run_id,
                total_questions,
                min_unit_chars=threshold,
                style_mix=style_mix if style_mix is not None else self._style_mix,
            )
            ctx.store.save(ctx.run_id, artifacts.PLAN, plan)
            ctx.update_meta(total_questions=total_questions)
            ctx.log(
                "plan built",
                units=len(plan.units),
                planned=plan.planned_questions,
                zero_budget_units=sum(1 for u in plan.units if u.questions == 0),
                min_unit_chars=plan.min_unit_chars,
                auto=plan.min_unit_chars_auto,
                skipped_sections=len(plan.skipped_section_ids),
            )
        return plan
