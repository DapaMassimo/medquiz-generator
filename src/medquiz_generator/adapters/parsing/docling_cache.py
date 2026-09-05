"""Content-hash cache for parsed DoclingDocuments (in-memory + optional disk).

Re-parsing a 40-page PDF with Docling takes tens of seconds; keyed by the PDF's content hash
(and OCR flag), a second run on the same file is instant. ``use_disk=False`` bypasses the disk
layer without breaking the in-memory hand-off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DoclingDocumentCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self._memory: dict[str, Any] = {}
        self._cache_dir = cache_dir

    def _path(self, content_hash: str, disk_suffix: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{content_hash}{disk_suffix}.json"

    def get(self, content_hash: str, *, use_disk: bool = True, disk_suffix: str = "") -> Any | None:
        cached = self._memory.get(content_hash)
        if cached is not None:
            return cached
        path = self._path(content_hash, disk_suffix)
        if use_disk and path is not None and path.exists():
            from docling_core.types.doc.document import DoclingDocument

            document = DoclingDocument.model_validate_json(path.read_text(encoding="utf-8"))
            self._memory[content_hash] = document
            return document
        return None

    def put(
        self, content_hash: str, document: Any, *, use_disk: bool = True, disk_suffix: str = ""
    ) -> None:
        self._memory[content_hash] = document
        path = self._path(content_hash, disk_suffix)
        if use_disk and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document.model_dump_json(), encoding="utf-8")

    def delete(self, content_hash: str) -> None:
        self._memory.pop(content_hash, None)
        if self._cache_dir is not None:
            for path in self._cache_dir.glob(f"{content_hash}*.json"):
                path.unlink(missing_ok=True)
