"""Docling adapter for the DocumentParser port.

* OCR off by default (native PDFs); switch on via ``MQ_DOCLING_DO_OCR=true``.
* ``artifacts_path`` points Docling at pre-downloaded models for offline parsing.
* Parsed DoclingDocuments are cached by PDF content hash (see ``docling_cache``).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from medquiz_generator.adapters.parsing.docling_cache import DoclingDocumentCache
from medquiz_generator.adapters.parsing.outline import (
    OutlineItem,
    build_outline,
    infer_levels_from_case,
)
from medquiz_generator.domain.errors import ParsingError
from medquiz_generator.domain.models import StructuredDocument

log = logging.getLogger(__name__)

# Captions are kept as text: a table's title is often the only trace of a rasterised table.
_SKIPPED_LABELS = {"page_header", "page_footer", "picture", "footnote"}
# Layout models label bold captions ("Tabella 3.5 …", "Figura 3.2") as section headers; they
# are not sections and would split the document into empty, meaningless nodes.
_CAPTION_HEADING = re.compile(r"^\s*(Tabella|Tab\.|Figura|Fig\.|Box|Scheda)\s*\d", re.IGNORECASE)

# ------------------------------------------------------------------ model licences
#
# Docling fetches model weights at runtime, and their licences are NOT covered by the
# Python package metadata — a clean `pip-licenses` report says nothing about them. The
# models this adapter may reach for are therefore pinned and allowlisted here.
#
# Cleared for commercial use and redistribution:
#   docling-project/docling-layout-heron   Apache-2.0             layout (Docling default)
#   docling-project/docling-models         CDLA-Permissive-2.0    TableFormer V1
#   RapidOCR / EasyOCR / Tesseract         Apache-2.0             OCR engines
#
# Deliberately excluded:
#   docling-project/TableFormerV2   model card declares no licence at all
#   datalab-to/chandra-ocr-*        revenue-capped commercial terms
#   nemotron-ocr, granite-vision    not reviewed
#
# A future Docling release may move its own defaults onto any of those, so the choices
# below are set explicitly rather than inherited, and _assert_allowlisted trips if the
# resolved configuration ever lands outside the cleared set.

ALLOWED_OCR_ENGINES = ("rapidocr", "easyocr", "tesseract")
_ALLOWED_TABLE_KIND = "docling_tableformer"  # TableFormer V1


def _ocr_options(engine: str) -> Any:
    """Build OCR options for an allowlisted engine, rejecting anything else."""
    if engine not in ALLOWED_OCR_ENGINES:
        raise ParsingError(
            f"OCR engine {engine!r} is not allowlisted. Use one of "
            f"{', '.join(ALLOWED_OCR_ENGINES)}: the other Docling backends ship models "
            "under revenue-capped or undeclared licences."
        )
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
    )

    builders: dict[str, type[Any]] = {
        "rapidocr": RapidOcrOptions,
        "easyocr": EasyOcrOptions,
        "tesseract": TesseractCliOcrOptions,
    }
    return builders[engine]()


def _assert_allowlisted(options: Any, *, do_ocr: bool) -> None:
    """Tripwire: fail loudly if Docling resolved a model outside the cleared set."""
    table_kind = getattr(options.table_structure_options, "kind", None)
    if table_kind != _ALLOWED_TABLE_KIND:
        raise ParsingError(
            f"Docling resolved table model {table_kind!r}, but only "
            f"{_ALLOWED_TABLE_KIND!r} (TableFormer V1) has a licence cleared for "
            "commercial use. Review the model's licence before allowing it."
        )
    if not do_ocr:
        return
    ocr_kind = getattr(options.ocr_options, "kind", None)
    if ocr_kind not in ALLOWED_OCR_ENGINES:
        raise ParsingError(
            f"Docling resolved OCR engine {ocr_kind!r}, which is not allowlisted. "
            f"Allowed: {', '.join(ALLOWED_OCR_ENGINES)}."
        )


def content_hash_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_converter(
    artifacts_path: Path | None, do_ocr: bool, ocr_engine: str, *, picture_images: bool = False
) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=do_ocr)
    if picture_images:
        # Keep rendered crops of the figures so they can be saved as run assets.
        options.generate_picture_images = True
        options.images_scale = 2.0
    # Pinned, not inherited: see the model-licence note above.
    options.table_structure_options = TableStructureOptions()
    if do_ocr:
        options.ocr_options = _ocr_options(ocr_engine)
    _assert_allowlisted(options, do_ocr=do_ocr)
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


FigureSaver = Callable[[Any, int], OutlineItem | None]


def _figure_size(item: Any) -> tuple[int, int]:
    prov = getattr(item, "prov", None) or []
    bbox = getattr(prov[0], "bbox", None) if prov else None
    if bbox is None:
        return 0, 0
    return int(abs(bbox.r - bbox.l)), int(abs(bbox.t - bbox.b))


def make_figure_saver(document: Any, assets_dir: Path, *, min_side: int) -> FigureSaver:
    """Save Docling picture items as PNG under ``assets_dir`` and describe them as outline
    items; tiny pictures (icons, bullets) are skipped."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    def save(item: Any, page: int) -> OutlineItem | None:
        width, height = _figure_size(item)
        if min(width, height) < min_side:
            return None
        try:
            image = item.get_image(document)
        except Exception:  # picture export is best-effort
            image = None
        if image is None:
            return None
        counter["n"] += 1
        filename = f"f{counter['n']:03d}.png"
        image.save(assets_dir / filename)
        try:
            caption = str(item.caption_text(document) or "")
        except Exception:
            caption = ""
        return OutlineItem(
            "figure",
            caption,
            page,
            ref=f"{assets_dir.parent.name}/{assets_dir.name}/{filename}",
            width=width,
            height=height,
        )

    return save


def iter_outline_items(
    document: Any, figure_saver: FigureSaver | None = None
) -> Iterator[OutlineItem]:
    """Translate Docling items into OutlineItems, tolerant to Docling version drift."""
    for item, _level in document.iterate_items():
        label = str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", "")))
        cls = type(item).__name__
        page = _page_of(item)
        if figure_saver is not None and (cls == "PictureItem" or label == "picture"):
            figure = figure_saver(item, page)
            if figure is not None:
                yield figure
            continue
        if cls == "TitleItem" or label == "title":
            yield OutlineItem("heading", str(getattr(item, "text", "")), page, level=1)
        elif cls == "SectionHeaderItem" or label == "section_header":
            heading = str(getattr(item, "text", ""))
            if _CAPTION_HEADING.match(heading):
                yield OutlineItem("text", heading, page)
                continue
            level = int(getattr(item, "level", 1)) + 1
            yield OutlineItem("heading", heading, page, level=level)
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
    document: Any,
    *,
    source_path: Path,
    content_hash: str,
    parser_name: str,
    assets_dir: Path | None = None,
    min_figure_side: int = 80,
) -> StructuredDocument:
    saver = (
        make_figure_saver(document, assets_dir, min_side=min_figure_side)
        if assets_dir is not None
        else None
    )
    items = infer_levels_from_case(list(iter_outline_items(document, saver)))
    title = next((i.text for i in items if i.kind == "heading"), source_path.stem)
    sections, figures = build_outline(items, root_title=title)
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
        figures=figures,
        parser=parser_name,
    )


class DoclingParser:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        artifacts_path: Path | None = None,
        do_ocr: bool = False,
        ocr_engine: str = "rapidocr",
        min_figure_side: int = 80,
    ) -> None:
        self._cache = DoclingDocumentCache(cache_dir)
        self._artifacts_path = artifacts_path
        self._do_ocr = do_ocr
        self._ocr_engine = ocr_engine
        self._converters: dict[str, Any] = {}
        self._min_figure_side = min_figure_side

    @property
    def name(self) -> str:
        return f"docling(ocr={'on' if self._do_ocr else 'off'})"

    def _convert(self, pdf_path: Path, *, picture_images: bool = False) -> Any:
        key = "img" if picture_images else "text"
        converter = self._converters.get(key)
        if converter is None:
            converter = _build_converter(
                self._artifacts_path, self._do_ocr, self._ocr_engine, picture_images=picture_images
            )
            self._converters[key] = converter
        try:
            return converter.convert(str(pdf_path)).document
        except Exception as exc:  # normalise any Docling failure
            raise ParsingError(f"Docling failed on {pdf_path}: {exc}") from exc

    def parse(
        self, pdf_path: Path, *, use_cache: bool = True, assets_dir: Path | None = None
    ) -> StructuredDocument:
        if not pdf_path.exists():
            raise ParsingError(f"PDF not found: {pdf_path}")
        content_hash = content_hash_of(pdf_path)
        with_images = assets_dir is not None
        suffix = (".ocr1" if self._do_ocr else ".ocr0") + (".img1" if with_images else "")
        document = self._cache.get(content_hash, use_disk=use_cache, disk_suffix=suffix)
        if document is None:
            log.info("Parsing %s with Docling (%s)", pdf_path.name, self.name)
            document = self._convert(pdf_path, picture_images=with_images)
            self._cache.put(content_hash, document, use_disk=use_cache, disk_suffix=suffix)
        else:
            log.info("Parse cache hit for %s", pdf_path.name)
        return structured_document_from_docling(
            document,
            source_path=pdf_path,
            content_hash=content_hash,
            parser_name=self.name,
            assets_dir=assets_dir,
            min_figure_side=self._min_figure_side,
        )
