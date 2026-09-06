"""Optional LangSmith tracing. Off by default; when enabled every LangChain call is traced.

LangChain reads the standard ``LANGSMITH_*`` environment variables, so enabling tracing is
purely an environment concern — no code path differs when the flag is off.

That reading happens against ``os.environ``, which a ``.env`` file never reaches on its own:
pydantic-settings parses it into ``Settings``, not into the process environment. Every
variable LangChain needs therefore has to be re-exported here, or a line in ``.env`` is
silently dropped.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def configure_tracing(*, enabled: bool, api_key: str, project: str, endpoint: str = "") -> bool:
    """Translate the app's settings into the environment LangChain reads.

    ``endpoint`` selects the LangSmith region, e.g. ``https://eu.api.smith.langchain.com``
    for an EU workspace. Empty leaves whatever the real environment already sets, falling
    back to LangSmith's own default (US) — an EU key against that default logs nothing and
    reports no error, so the region has to be set explicitly.
    """
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    if not api_key and not os.environ.get("LANGSMITH_API_KEY"):
        log.warning("MQ_LANGSMITH_TRACING=true but no LANGSMITH_API_KEY: tracing disabled.")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_PROJECT"] = project
    log.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s)",
        project,
        os.environ.get("LANGSMITH_ENDPOINT", "default"),
    )
    return True
