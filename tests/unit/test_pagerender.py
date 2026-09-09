from __future__ import annotations

from pathlib import Path

import pytest

from medquiz_generator.entrypoints.pagerender import (
    rect_to_pixels,
    render_page,
    search_variants,
)


def test_rect_to_pixels_flips_the_pdf_y_axis() -> None:
    # A 10pt-high box near the top of a 792pt page, rendered at scale 2.
    assert rect_to_pixels((91.5, 648.0, 275.9, 658.0), 792.0, 2.0) == (183, 268, 551, 288)


def test_search_variants_shorten_progressively() -> None:
    quote = "Coinvolgenti la giunzione gastroesofagea adenocarcinomi stomaco prostata polmone extra"
    variants = search_variants(quote)
    assert variants[0] == quote
    assert (
        variants[1]
        == "Coinvolgenti la giunzione gastroesofagea adenocarcinomi stomaco prostata polmone"
    )
    assert variants[2] == "Coinvolgenti la giunzione gastroesofagea"
    assert search_variants("una frase") == ["una frase"]
    assert search_variants("   ") == []


def test_render_page_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert render_page(tmp_path / "missing.pdf", [1], "x") is None


SAMPLE = Path("data/input/capitolo25.pdf")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_render_page_finds_the_quote_on_the_right_page() -> None:
    result = render_page(SAMPLE, [14, 15, 16], "Coinvolgenti la giunzione gastroesofagea")
    assert result is not None
    png, page = result
    assert page == 15 and png[:4] == b"\x89PNG"
