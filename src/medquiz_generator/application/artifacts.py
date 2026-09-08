"""Names of the artefacts each stage persists under ``runs/<run_id>/``.

Every stage reads the artefact of the previous one by these names, which is what makes each
CLI command independently re-runnable.
"""

from __future__ import annotations

DOCUMENT = "document"
FIGURES = "figures"
FIGURES_DIR = "assets/figures"
PLAN = "plan"
TAXONOMY = "taxonomy"
QUESTIONS_GENERATED = "questions.generated"
QUESTIONS_VERIFIED = "questions.verified"
QUESTIONS_DEDUPED = "questions.deduped"
VERIFICATION_REPORT = "verification_report"
DEDUP_REPORT = "dedup_report"
RUN_REPORT = "run_report"
EVAL_REPORT = "eval_report"
REVIEW_STATE = "review_decisions"
RUN_META = "run_meta"

# JSON-lines artefacts (append-only)
USAGE_LOG = "usage"
TIMINGS_LOG = "timings"

EXPORT_DIR = "export"
