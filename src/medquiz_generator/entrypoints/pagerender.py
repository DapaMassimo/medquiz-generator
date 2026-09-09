"""Render a page of the source PDF as an image, with the cited quote highlighted.

Presentation-only helper for the review UI: when the cited passage comes from a table or a
boxed list, the extracted text loses its layout, and the reader is better served by the page
itself. Uses pypdfium2 (already a Docling dependency); every failure degrades to ``None``.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

Rect = tuple[float, float, float, float]  # PDF coordinates: left, bottom, right, top


def rect_to_pixels(rect: Rect, page_height: float, scale: float) -> tuple[int, int, int, int]:
    """PDF user-space rectangle (origin bottom-left) → pixel box (origin top-left)."""
    left, bottom, right, top = rect
    return (
        int(left * scale),
        int((page_height - top) * scale),
        int(right * scale),
        int((page_height - bottom) * scale),
    )


def search_variants(quote: str, max_words: int = 8) -> list[str]:
    """Progressively shorter prefixes of the quote to search for on the page: the full quote
    rarely matches verbatim (line breaks, hyphenation), its first words usually do."""
    words = quote.split()
    if not words:
        return []
    variants = [quote.strip()]
    for n in (max_words, 4):
        if len(words) > n:
            variants.append(" ".join(words[:n]))
    return variants


def _find_rects(textpage: object, quote: str) -> list[Rect]:
    for needle in search_variants(quote):
        searcher = textpage.search(needle)  # type: ignore[attr-defined]
        hit = searcher.get_next()
        if hit is None:
            continue
        index, count = hit
        n = textpage.count_rects(index, count)  # type: ignore[attr-defined]
        return [textpage.get_rect(i) for i in range(n)]  # type: ignore[attr-defined]
    return []


def render_page(
    pdf_path: Path, pages: Sequence[int], quote: str = "", scale: float = 2.0
) -> tuple[bytes, int] | None:
    """PNG of the first page in ``pages`` (1-based) where ``quote`` is found, with the match
    highlighted; if it is found nowhere, the first page unhighlighted. ``None`` on any error."""
    try:
        import pypdfium2 as pdfium
        from PIL import Image, ImageDraw

        pdf = pdfium.PdfDocument(str(pdf_path))
        candidates = [p for p in pages if 1 <= p <= len(pdf)]
        if not candidates:
            return None
        chosen, rects = candidates[0], []
        if quote:
            for page_no in candidates:
                found = _find_rects(pdf[page_no - 1].get_textpage(), quote)
                if found:
                    chosen, rects = page_no, found
                    break
        page = pdf[chosen - 1]
        image = page.render(scale=scale).to_pil().convert("RGBA")
        if rects:
            _, height = page.get_size()
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for rect in rects:
                x0, y0, x1, y1 = rect_to_pixels(rect, height, scale)
                draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(255, 230, 0, 110))
            image = Image.alpha_composite(image, overlay)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), chosen
    except Exception:  # presentation helper: never fail the page
        return None
