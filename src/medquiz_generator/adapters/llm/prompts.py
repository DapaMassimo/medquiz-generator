"""Prompts and wire schemas for the LLM adapters.

Instructions are in English; the generated content must be in Italian. No medical content is
embedded here: everything a question contains must come from the source passage passed in.
Wire schemas have no defaults so they translate to strict JSON schemas for structured output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from medquiz_generator.application.examples import format_examples
from medquiz_generator.domain.models import (
    PlanUnit,
    QuizItem,
    Section,
    SectionLabel,
    SsmExample,
    StructuredDocument,
)

OptionIdWire = Literal["A", "B", "C", "D", "E"]

# ------------------------------------------------------------------ wire schemas


class WireOption(BaseModel):
    id: OptionIdWire
    text: str


class WireQuestion(BaseModel):
    question: str = Field(description="Testo del quesito in italiano (vignetta clinica o nozione)")
    options: list[WireOption] = Field(description="Esattamente 5 opzioni con id A, B, C, D, E")
    correct_option_id: OptionIdWire
    explanation: str = Field(description="2-4 frasi in italiano che motivano la risposta corretta")
    difficulty: Literal["facile", "media", "difficile"]
    quote: str = Field(
        description="Citazione testuale, contigua e letterale, presa dal testo della sezione, "
        "che sostiene la risposta corretta (max 60 parole)"
    )
    page: int = Field(description="Pagina del documento da cui proviene la citazione")


class WireBatch(BaseModel):
    questions: list[WireQuestion]


OptionStatusWire = Literal["supported", "contradicted", "not_in_passage"]


class WireOptionAssessment(BaseModel):
    id: OptionIdWire
    status: OptionStatusWire = Field(
        description="supported = the passage states this option answers the question; "
        "contradicted = the passage states otherwise; not_in_passage = the passage says nothing"
    )
    evidence: str = Field(
        description="The words of the passage (or the numbers compared) that justify the status, "
        "in Italian; empty when not_in_passage"
    )


class WireVerdict(BaseModel):
    passage_answer: str = Field(
        description="What the passage itself says about the question, in one sentence in Italian, "
        "before looking at the options"
    )
    options: list[WireOptionAssessment] = Field(
        description="One assessment for every option, A to E, in order"
    )
    verdict: Literal["pass", "discard", "flag"]
    reason: str = Field(description="Short justification in Italian")


class WireAssignment(BaseModel):
    section_id: str = Field(description="Section id exactly as given in the outline")
    subject: str = Field(description="Materia, in italiano, breve e normalizzata")
    topic: str = Field(description="Argomento, in italiano, specifico e riusato per sezioni affini")


class WireTaxonomy(BaseModel):
    assignments: list[WireAssignment] = Field(
        description="Exactly one entry per section id of the outline, in outline order"
    )


class WireStyleScores(BaseModel):
    ssm_style: int = Field(ge=1, le=5)
    distractor_quality: int = Field(ge=1, le=5)
    explanation_clarity: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    rationale: str


# -------------------------------------------------------------------- generation

GENERATION_SYSTEM = """You write multiple-choice questions for the Italian SSM exam (concorso
nazionale per le Scuole di Specializzazione in Medicina). You receive ONE section of a medical
textbook and must produce exactly the requested number of questions, all in Italian.

Hard rules — violating any of them makes the question unusable:
1. Source-only: every fact in the question, in the correct answer and in the explanation must
   be stated in the provided section text. Do not use outside knowledge, do not invent numbers,
   drug names, thresholds or mechanisms that are not in the text.
2. Exactly 5 options with ids A, B, C, D, E; exactly ONE is correct. Options must be distinct,
   non-empty, of similar length and register; no "tutte le precedenti" / "nessuna delle
   precedenti". Vary the position of the correct option across the questions you write: no
   letter must be systematically the right one.
3. Distractors must be plausible but unequivocally wrong according to the section text: build
   them from typical errors and from neighbouring concepts that appear in the same material
   (other values, other classes, other conditions mentioned nearby).
4. Style: SSM register. When the content allows it, open with a short clinical vignette
   followed by a precise question; otherwise ask a direct, factual question in the same formal
   register. One question = one concept. Vary the vignette (age, sex, setting, presenting
   complaint): do not reuse the ages or the openings of the example questions, and never copy
   their answer markers into the option texts.
5. `quote`: a literal, contiguous excerpt copied verbatim from the section text (max 60 words)
   that supports the correct answer. Do not paraphrase, do not merge distant sentences.
6. `page`: the page number of the quote, within the page range given for the section.
7. `explanation`: 2-4 sentences in Italian explaining why the correct option is right and,
   briefly, why the main distractor is wrong — again only with facts from the text. Refer to
   options by their content, never by their letter (the options will be reordered).
8. Cover different parts of the section: do not ask twice about the same sentence.
9. Estimate `difficulty` as facile / media / difficile.

Follow the format and register of the example questions when they are provided.
Output only the structured object."""


def generation_user_prompt(
    unit: PlanUnit,
    sections: list[Section],
    n: int,
    examples: list[SsmExample],
    label: SectionLabel | None,
) -> str:
    parts: list[str] = []
    if examples:
        parts.append(
            "### Example questions from real SSM exams (style and format reference only — "
            "their medical content is NOT a source):\n\n" + format_examples(examples)
        )
    meta = [f"Unit: {unit.title}", f"Pages: {unit.page_start}-{unit.page_end}"]
    if label is not None:
        meta.append(f"Subject: {label.subject} — Topic: {label.topic}")
    parts.append("### Section metadata\n" + "\n".join(meta))
    body: list[str] = []
    for section in sections:
        body.append(
            f"[Section: {section.path_str} | pages {section.page_start}-{section.page_end}]\n"
            f"{section.text}"
        )
    parts.append("### Section text (the ONLY allowed source)\n\n" + "\n\n".join(body))
    parts.append(
        f"### Task\nWrite exactly {n} question(s) in Italian following the rules. "
        "Return them in the structured format."
    )
    return "\n\n".join(parts)


# ------------------------------------------------------------------------- judge

JUDGE_SYSTEM = """You are an independent examiner verifying a multiple-choice question against
the ONLY source you are given: a passage from a medical textbook. You did not write the
question and you must not rely on outside medical knowledge — if a claim is not in the passage
it is unsupported.

Work in this order:
1. passage_answer: state what the passage says about the question, ignoring the options. When
   the passage contains numbers, ranges, thresholds or a table, compare the values in the
   question against them explicitly (e.g. "150/95 falls in the 140-159/90-99 row").
2. options: assess EVERY option A-E against the passage, one by one, quoting the evidence:
   "supported" only if the passage states that this option answers the question;
   "contradicted" if the passage states otherwise; "not_in_passage" if it says nothing.
   Judge each option by its TEXT, not by its letter; the marked answer is given as letter and
   text so you can check they correspond.
3. verdict:
   - "discard" when the marked option is not "supported", when a second option is also
     "supported", or when the question contradicts the passage;
   - "flag" when support is only partial, the wording is ambiguous, a distractor is borderline,
     or the passage is too short to be sure — a human must review it;
   - "pass" only when the marked option is supported beyond doubt and no other option is.
Write the reason in Italian, one or two sentences, consistent with the option assessments."""


def judge_user_prompt(item: QuizItem, passage: str) -> str:
    options = "\n".join(f"{o.id}: {o.text}" for o in item.options)
    correct = item.correct_option
    marked = correct.text if correct is not None else "(opzione non trovata)"
    return (
        f"### Passage (source of truth)\n{passage}\n\n"
        f"### Cited quote\n{item.source.quote}\n\n"
        f"### Question\n{item.question}\n\n"
        f"### Options\n{options}\n\n"
        f"### Marked correct option\n{item.correct_option_id}: {marked}\n\n"
        "Evaluate and return the structured verdict."
    )


# ---------------------------------------------------------------------- taxonomy

TAXONOMY_SYSTEM = """You organise the outline of a medical textbook into a two-level taxonomy:
subjects (materie, e.g. the medical discipline or organ system a chapter belongs to) and topics
(argomenti, the specific clinical subject of a section). Names must be in Italian, short and
normalised: merge sections that clearly belong to the same subject even if their headings differ,
and reuse the same topic name for sections about the same argument.

Rules:
- Return ONE assignment per section id, in outline order: no id omitted, no id repeated. The
  outline states how many sections there are; return exactly that many assignments.
- Do not invent subjects that the outline does not support; derive them from the headings and
  the text snippets. Do not use the chapter title itself as a subject when the medical
  discipline is evident from the content.
- Keep the number of subjects small (typically 1-8 for a single volume) and topics specific:
  a topic usually groups several consecutive sections about the same argument, and should not
  exceed roughly 15 sections; a chapter with more than 100 sections needs at least 8-10 topics.
  Do not collapse the whole document into one topic.
- Generic headings such as "Diagnosi", "Terapia", "Storia naturale", "Esame obiettivo" must be
  resolved into the argument they belong to, using the neighbouring sections and the snippets
  (e.g. the "Diagnosi" section that follows "Danno renale" belongs to the topic "Danno renale").
Return only the structured list of assignments."""


def taxonomy_user_prompt(document: StructuredDocument, snippet_chars: int = 1000) -> str:
    with_text = [s for s in document.sections if s.text.strip()]
    lines = [
        f"Document: {document.title} ({document.page_count} pages)",
        f"Sections to assign: {len(with_text)} (return exactly {len(with_text)} assignments)",
        "",
        "### Outline",
    ]
    for section in with_text:
        indent = "  " * max(0, section.level - 1)
        snippet = " ".join(section.text.split())[:snippet_chars]
        lines.append(
            f"{indent}- id={section.id} | level={section.level} | pages "
            f"{section.page_start}-{section.page_end} | {section.title}\n"
            f"{indent}  snippet: {snippet}"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------- style judge

STYLE_JUDGE_SYSTEM = """You are a senior author of Italian SSM exam questions evaluating a
candidate question against a golden set of real exam questions. Score each dimension from 1
(poor) to 5 (indistinguishable from the golden set), and justify in Italian.

Dimensions:
- ssm_style: register, structure and length match the golden examples; a clinical vignette is
  used when the content allows it, otherwise a crisp factual stem.
- distractor_quality: distractors are plausible, homogeneous, built on neighbouring concepts,
  and unambiguously wrong; no giveaway patterns.
- explanation_clarity: the explanation is 2-4 sentences, precise, and teaches why the correct
  option is right and the key distractor wrong.
- groundedness: question, correct answer and explanation are fully supported by the cited
  passage; no information beyond it.
Do not reward medical content that is absent from the passage."""


def style_judge_user_prompt(
    item: QuizItem,
    golden_style: list[SsmExample],
    golden_regression: list[QuizItem],
    passage: str = "",
) -> str:
    parts = ["### Golden set — real SSM questions\n" + format_examples(golden_style)]
    if golden_regression:
        rendered = []
        for golden in golden_regression:
            options = "\n".join(f"{o.id}: {o.text}" for o in golden.options)
            rendered.append(
                f"{golden.question}\n{options}\nCorretta: {golden.correct_option_id}\n"
                f"Commento: {golden.explanation}"
            )
        parts.append(
            "### Golden set — previously approved generated questions\n\n"
            + "\n\n---\n\n".join(rendered)
        )
    options = "\n".join(f"{o.id}: {o.text}" for o in item.options)
    parts.append(
        "### Candidate question\n"
        f"{item.question}\n{options}\nCorretta: {item.correct_option_id}\n"
        f"Spiegazione: {item.explanation}\n\n"
        f"### Cited quote\n{item.source.quote}\n\n"
        f"### Source passage (context around the quote)\n{passage or item.source.quote}"
    )
    parts.append("Score the candidate on the four dimensions.")
    return "\n\n".join(parts)
