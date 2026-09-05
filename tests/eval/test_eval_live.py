"""Opt-in live evaluation (``make test-eval``): needs OPENAI_API_KEY and an existing run.

Scores the latest run's verified questions against the golden sets with the real style judge
and prints the rubric averages. Skipped automatically without a key or without a run.
"""

from __future__ import annotations

import os

import pytest

from medquiz_generator.composition.container import Container
from medquiz_generator.entrypoints.presentation import render_eval

pytestmark = pytest.mark.eval


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_eval_on_latest_run() -> None:
    container = Container()
    run_id = container.store.latest_run_id()
    if run_id is None:
        pytest.skip("No run to evaluate: run `medquiz run <pdf> -n N` first")
    ctx = container.context(run_id)
    report = container.eval_stage(ctx).run(ctx, limit=int(os.environ.get("MQ_EVAL_LIMIT", "5")))
    print(render_eval(report))
    assert report.items, "no verified items to score"
    assert report.averages["groundedness"] >= 3.0
