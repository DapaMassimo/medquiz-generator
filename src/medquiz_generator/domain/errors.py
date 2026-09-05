"""Domain-level exceptions. Adapters translate framework errors into these."""

from __future__ import annotations


class MedQuizError(Exception):
    """Base class for all application errors."""


class ParsingError(MedQuizError):
    """The source document could not be parsed into a structured document."""


class PlanningError(MedQuizError):
    """A generation plan could not be produced (e.g. empty document, invalid total)."""


class StageInputMissingError(MedQuizError):
    """A stage was launched but the artefact produced by the previous stage is missing."""


class LLMError(MedQuizError):
    """An LLM or embedding call failed after the adapter's own retries."""


class ExportError(MedQuizError):
    """An export target could not be written."""
