# medquiz-generator

Systematic, verifiable generation of Italian **SSM-style multiple-choice questions** (concorso
nazionale per le Scuole di Specializzazione in Medicina) from native medical PDFs — with a
two-level anti-hallucination gate, cost/observability built in, and a Streamlit app for exam
simulation and human review.

Code, comments and this README are in English; the generated content (questions,
explanations) is in Italian.

## Features

- **Systematic coverage, not retrieval** — the whole document is covered section by section;
  the requested total is apportioned proportionally to section length (min. one question per
  substantial section). The total is a mandatory per-run parameter: there is no default.
- **Document-derived taxonomy** — no predefined subject list: *materie → argomenti → sezioni*
  are extracted from the document structure and normalised by an LLM; every question inherits
  its labels from its source section. The taxonomy is a run deliverable (`taxonomy.json`).
- **SSM style** — clinical vignette where the content allows it, 5 options (A–E), one correct,
  2–4 sentence explanation, few-shot on real exam questions (`examples/ssm/ssm_examples.txt`).
- **Anti-hallucination, two levels** — deterministic checks (5 distinct non-empty options,
  one correct, quote *really* present in the section text with whitespace/punctuation-normalised
  matching, page in range) and an **independent LLM judge** that sees only the cited passage.
  Failures are discarded or flagged, never delivered silently.
- **Dedup** — embedding similarity flags near-duplicates across neighbouring sections.
- **Export** — JSON (full schema) + XLSX (one row per question); flagged items only on request.
- **Observability first** — tokens and USD cost per call, per stage, per section and per
  question (price table in `pricing.json`, editable without code); persisted run report;
  structured logs with run id; optional LangSmith tracing (off by default).
- **Eval harness with growing golden sets** — a dedicated style judge scores generated
  questions against the real SSM questions and against items approved in the UI and promoted
  to `golden/`; `medquiz eval compare` gives per-dimension deltas between two runs.
- **Resumable stages** — every stage persists its artefact; each CLI command re-runs only its
  own stage; parsing is cached by PDF content hash.
- **Review UI (Streamlit)** — exam simulator (the client-facing preview) and a review table
  with side-by-side cited passage, approve/reject with notes, export of approved items, and
  promotion to the golden set. Decisions persist on file; no database.

## Architecture

Hexagonal (ports & adapters); the inward dependency rule is **enforced by import-linter**
(`.importlinter`, run by `make check`).

```
entrypoints (cli.py · streamlit_app.py · presentation · logsetup)
        │
composition (container.py — the only module importing concrete adapters, DI from Settings)
        │
adapters ─────────────────────► domain ◄───────────────── application
(Docling parser + cache,        (Pydantic entities,        (use cases, one per stage:
 LangChain generator / judge /   Protocol ports,            parse · plan · classify ·
 taxonomy / style judge,         text normalisation,        generate · verify · dedup ·
 OpenAI embeddings,              usage & pricing)           export · eval · report ·
 JSON/XLSX exporters,                                       review · pipeline)
 file run store, LangSmith)
```

Pipeline (each arrow is a persisted artefact under `runs/<run_id>/`):

```
PDF ─parse─► document.json ─plan(N)─► plan.json ─classify─► taxonomy.json
    ─generate─► questions.generated.json ─verify─► questions.verified.json
    ─dedup─► questions.deduped.json ─export─► export/quiz.{json,xlsx}
    ─report─► run_report.json          ─eval─► eval_report.json
```

### Project layout

```
src/medquiz_generator/
├── domain/        models.py (Section, StructuredDocument, GenerationPlan, Taxonomy, QuizItem,
│                  JudgeVerdict, RunReport, EvalReport…) · ports.py · text.py · usage.py · errors.py
├── application/   parse · plan · classify · generate · verify · dedup · export · report ·
│                  review · eval · pipeline · examples (ssm_examples.txt parser) · context
├── adapters/      parsing/ (docling_parser, docling_cache, outline) · llm/ (langchain_llm,
│                  prompts, adapters) · embeddings/ · export/ (json, xlsx) · storage/ · tracing/
├── config/        settings.py (pydantic-settings, prefix MQ_) · pricing.py
├── composition/   container.py
└── entrypoints/   cli.py (typer) · streamlit_app.py · presentation.py · logsetup.py
tests/             fixtures/structured_document.json · fakes/ · unit/ · integration/ · eval/
examples/ssm/      ssm_examples.txt (real exam questions: few-shot + style golden set)
golden/regression/ items promoted from the review UI
runs/<run_id>/     every artefact of a run (git-ignored)
pricing.json       USD per 1M tokens per model
```

## Quick start

```bash
make install                 # uv sync --all-groups
cp .env.example .env         # set OPENAI_API_KEY (and tweak MQ_* if needed)

# Full pipeline: the total number of questions is mandatory
uv run medquiz run data/input/capitolo1.pdf --total-questions 40

# Review / simulate
make run                     # Streamlit → http://localhost:8501   (or: uv run medquiz serve)
```

Stage by stage (each command reads the previous stage's artefact of the latest run, or of
`--run-id <id>`):

```bash
uv run medquiz parse data/input/capitolo1.pdf   # creates a run id, writes document.json
uv run medquiz plan --total-questions 40        # plan.json — inspect/edit by hand if you like
uv run medquiz classify                          # taxonomy.json (subjects → topics → sections)
uv run medquiz generate                          # questions.generated.json (runs classify if missing)
uv run medquiz verify                            # deterministic checks + LLM judge
uv run medquiz dedup                             # near-duplicate flagging
uv run medquiz export --include-flagged          # export/quiz.json + quiz.xlsx
uv run medquiz report                            # coverage, verdicts, timings, costs
uv run medquiz eval --limit 20                   # rubric scores vs golden sets → eval_report.json
uv run medquiz eval compare <run_a> <run_b>      # per-dimension deltas
uv run medquiz runs | show                       # list runs / print plan + taxonomy
```

The plan is a plain JSON file: change `questions` per unit (or drop units) before `generate`.

## The question schema (`QuizItem`)

| Field | Content |
|-------|---------|
| `id` | `<unit>-q<nn>` (unit = plan unit id) |
| `materia`, `argomento` | inherited from the run taxonomy via the source section |
| `question`, `options[5]` (`A`–`E`), `correct_option_id`, `explanation`, `difficulty` | the question itself (Italian) |
| `source` | `section_id`, `section_title`, `section_path`, `page_start`, `page_end`, `page`, `quote` (verbatim passage) |
| `status` | `generated` / `verified` / `flagged` / `discarded` + `reason`, failed `checks`, judge verdict/reason, `duplicate_of` |
| `model_metadata` | model, run id, unit id, timestamp |

## Verification policy

| Check | Outcome |
|-------|---------|
| ≠ 5 options, empty/duplicate options, wrong ids, invalid correct id | **discard** |
| quote empty or not found in the cited section (normalised match) | **discard** |
| page outside the section range, explanation too short, quote too long | **flag** (judge still runs) |
| judge: answer unsupported / two correct / contradiction | **discard** |
| judge: partial support, ambiguity, borderline distractor | **flag** |
| judge pass + no soft failures | **verified** |
| near-duplicate (cosine ≥ `MQ_DEDUP_THRESHOLD`) | **flag** with `duplicate_of` |

Discarded questions never reach an export; flagged ones only with `--include-flagged`.

## Configuration (`.env`, prefix `MQ_`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | – | Generator, judges, taxonomy, embeddings |
| `ANTHROPIC_API_KEY` | – | Only when `MQ_LLM_PROVIDER=anthropic` |
| `MQ_LLM_PROVIDER` | `openai` | `openai` \| `anthropic` (LangChain partner packages) |
| `MQ_GENERATOR_MODEL` / `MQ_JUDGE_MODEL` / `MQ_TAXONOMY_MODEL` / `MQ_EVAL_JUDGE_MODEL` | `gpt-4o-mini` | Model per role |
| `MQ_EMBEDDING_MODEL` | `text-embedding-3-small` | Dedup embeddings |
| `MQ_LLM_TEMPERATURE` | `0.1` | Low temperature for all structured calls |
| `MQ_RUNS_DIR` / `MQ_GOLDEN_DIR` | `runs` / `golden` | Artefacts and promoted golden items |
| `MQ_EXAMPLES_PATH` | `examples/ssm/ssm_examples.txt` | Few-shot + style golden set |
| `MQ_PRICING_PATH` | `pricing.json` | Price table (USD per 1M tokens) |
| `MQ_DOCLING_CACHE_DIR` | `.cache/docling` | Parsed documents cached by content hash |
| `MQ_DOCLING_ARTIFACTS_PATH` | – | Local Docling models for offline parsing |
| `MQ_DOCLING_DO_OCR` | `false` | OCR (only for scanned PDFs) |
| `MQ_PLAN_MIN_UNIT_CHARS` | `600` | Small sibling sections are merged into one unit |
| `MQ_MAX_EXAMPLES_IN_PROMPT` | `5` | Few-shot examples per generation call |
| `MQ_VERIFY_JUDGE_ENABLED` | `true` | `false` = deterministic checks only |
| `MQ_DEDUP_ENABLED` / `MQ_DEDUP_THRESHOLD` | `true` / `0.92` | Near-duplicate flagging |
| `MQ_EXPORT_INCLUDE_FLAGGED` | `false` | Default for exports |
| `MQ_LANGSMITH_TRACING` / `MQ_LANGSMITH_PROJECT` / `LANGSMITH_API_KEY` | `false` | Optional tracing of every LLM call |
| `MQ_LOG_LEVEL` / `MQ_LOG_JSON` | `INFO` / `false` | Logging (JSON lines available) |

Model choices live in one place (`Settings` → `Container`); switching provider or model never
touches application code.

## Observability and costs

Every LLM/embedding call is recorded in `runs/<run_id>/usage.jsonl` (model, tokens in/out,
USD from `pricing.json`, duration, stage, unit, question). `run_report.json` aggregates:

- sections/units planned vs covered; generated / verified / flagged / discarded with reasons;
- deterministic-check failures vs judge verdicts; duplicates;
- duration per stage; tokens and cost per stage, per unit, per question;
- **cost per delivered question** — the number to extrapolate to a whole library.

`medquiz report` prints it; the UI shows it in the *Report* tab. Indicative figures with
`gpt-4o-mini`: one generation call per section (a few thousand input tokens) plus one short
judge call per question — parsing, planning, dedup maths and export are local and free apart
from the embedding call. Update `pricing.json` when prices change.

## Eval harness and golden sets

`make test-eval` (opt-in, needs `OPENAI_API_KEY` and an existing run) and `medquiz eval` score
each delivered question on a 1–5 rubric — *SSM style, distractor quality, explanation clarity,
groundedness* — against two golden sets:

1. **Style golden set**: the real exam questions in `examples/ssm/ssm_examples.txt`.
2. **Regression golden set**: questions that passed verification **and** were approved by a
   human, promoted from the review UI into `golden/regression/`. It grows with the project,
   so every prompt/model change can be measured before/after on the same references.

Results persist as `eval_report.json`; `medquiz eval compare <run_a> <run_b>` prints the
delta per dimension.

## Web app

`make run` opens two main tabs:

- **Simulatore** — the client-facing preview: questions one at a time or as a list, filter by
  materia/argomento, pick an answer, reveal the correct option, explanation and the cited
  source passage; running score.
- **Revisione** — table with status/filters; detail with the highlighted passage side by side;
  approve/reject with a note (persisted in `review_decisions.json`); export of approved items
  only; *Promuovi al golden set* for approved items.

Plus *Report* (run report, eval, downloads) and *Tassonomia* (the extracted taxonomy). A new
run can be launched from the sidebar (PDF + mandatory total).

## Development

```bash
make check      # ruff + ruff format --check + import-linter + mypy --strict + pytest (cov ≥ 80%)
make test       # tests only (eval tests excluded)
make test-eval  # opt-in live evaluation
make format     # ruff format + autofix
make hooks      # pre-commit install
```

Tests never touch the network: LLM, embeddings and parser are replaced by fakes honouring the
ports (`tests/fakes/`), and a synthetic structured document (`tests/fixtures/`) stands in for
Docling output. Covered contracts include the planner budget, every deterministic check
(missing quote, duplicate options, page out of range, wrong option count…), dedup, export,
report aggregation, review/promotion, eval comparison, the CLI and the full pipeline.

### Notes

- **OCR** is off by default because the inputs are native PDFs; enable
  `MQ_DOCLING_DO_OCR=true` only for scanned documents (much slower).
- **Offline parsing**: download Docling models once and point `MQ_DOCLING_ARTIFACTS_PATH` at
  them (see `.env.example`).
- **Windows**: `uv run …` commands work natively; `make` is not shipped with Windows, so
  either install GNU make or run the Makefile targets from WSL on the same folder (keep the
  virtualenv on the Linux filesystem with `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/medquiz`).
- **Provided taxonomy**: `TaxonomyBuilder` is a port; a builder fed by a client-supplied
  taxonomy can replace the extracted one without touching the pipeline (not implemented).

## Out of scope (possible evolutions)

Image-based questions, a user platform with accounts, payments, and e-learning/LMS integration
are intentionally not implemented in this MVP.

## License

MIT
