"""Docling adapter for the DocumentParser port.

* OCR off by default (native PDFs); switch on via ``MQ_DOCLING_DO_OCR=true``.
* ``artifacts_path`` points Docling at pre-downloaded models for offline parsing.
* Parsed DoclingDocuments are cached by PDF content hash (see ``docling_cache``).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from medquiz_generator.adapters.parsing.docling_cache import DoclingDocumentCache
from medquiz_generator.adapters.parsing.outline import (
    OutlineItem,
    build_sections,
    infer_levels_from_case,
)
from medquiz_generator.domain.errors import ParsingError
from medquiz_generator.domain.models import StructuredDocument

log = logging.getLogger(__name__)

_SKIPPED_LABELS = {"page_header", "page_footer", "picture", "caption", "footnote"}


def content_hash_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_converter(artifacts_path: Path | None, do_ocr: bool) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=do_ocr)
    if artifacts_path is not None:
        options.artifacts_path = str(artifacts_path)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _page_of(item: Any) -> int:
    prov = getattr(item, "prov", None) or []
    if prov:
        page = getattr(prov[0], "page_no", None)
        if isinstance(page, int):
            return page
    return 1


def iter_outline_items(document: Any) -> Iterator[OutlineItem]:
    """Translate Docling items into OutlineItems, tolerant to Docling version drift."""
    for item, _level in document.iterate_items():
        label = str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", "")))
        cls = type(item).__name__
        page = _page_of(item)
        if cls == "TitleItem" or label == "title":
            yield OutlineItem("heading", str(getattr(item, "text", "")), page, level=1)
        elif cls == "SectionHeaderItem" or label == "section_header":
            level = int(getattr(item, "level", 1)) + 1
            yield OutlineItem("heading", str(getattr(item, "text", "")), page, level=level)
        elif cls == "TableItem" or label == "table":
            try:
                markdown = item.export_to_markdown(doc=document)
            except Exception:  # table export is best-effort
                markdown = ""
            if markdown:
                yield OutlineItem("text", markdown, page)
        elif label in _SKIPPED_LABELS:
            continue
        else:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                yield OutlineItem("text", text, page)


def structured_document_from_docling(
    document: Any, *, source_path: Path, content_hash: str, parser_name: str
) -> StructuredDocument:
    items = infer_levels_from_case(list(iter_outline_items(document)))
    title = next((i.text for i in items if i.kind == "heading"), source_path.stem)
    sections = build_sections(items, root_title=title)
    if not sections:
        raise ParsingError(f"No text could be extracted from {source_path}.")
    pages = getattr(document, "pages", None)
    page_count = len(pages) if pages else max(s.page_end for s in sections)
    return StructuredDocument(
        source_path=str(source_path),
        content_hash=content_hash,
        title=title,
        page_count=max(1, page_count),
        sections=sections,
        parser=parser_name,
    )


class DoclingParser:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        artifacts_path: Path | None = None,
        do_ocr: bool = False,
    ) -> None:
        self._cache = DoclingDocumentCache(cache_dir)
        self._artifacts_path = artifacts_path
        self._do_ocr = do_ocr
        self._converter: Any | None = None

    @property
    def name(self) -> str:
        return f"docling(ocr={'on' if self._do_ocr else 'off'})"

    def _convert(self, pdf_path: Path) -> Any:
        if self._converter is None:
            self._converter = _build_converter(self._artifacts_path, self._do_ocr)
        try:
            return self._converter.convert(str(pdf_path)).document
        except Exception as exc:  # normalise any Docling failure
            raise ParsingError(f"Docling failed on {pdf_path}: {exc}") from exc

    def parse(self, pdf_path: Path, *, use_cache: bool = True) -> StructuredDocument:
        if not pdf_path.exists():
            raise ParsingError(f"PDF not found: {pdf_path}")
        content_hash = content_hash_of(pdf_path)
        suffix = ".ocr1" if self._do_ocr else ".ocr0"
        document = self._cache.get(content_hash, use_disk=use_cache, disk_suffix=suffix)
        if document is None:
            log.info("Parsing %s with Docling (%s)", pdf_path.name, self.name)
            document = self._convert(pdf_path)
            self._cache.put(content_hash, document, use_disk=use_cache, disk_suffix=suffix)
        else:
            log.info("Parse cache hit for %s", pdf_path.name)
        return structured_document_from_docling(
            document, source_path=pdf_path, content_hash=content_hash, parser_name=self.name
        )
