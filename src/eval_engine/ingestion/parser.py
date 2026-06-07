"""Clause-aware parser: turns scoped PDF pages into contextual Chunk objects.

Design (see models.py for the schema rationale):
- Prose clauses are split on their numbering (definitions: '33.', chapter
  clauses: '1.'); each becomes one chunk with a prepended context header.
- Tables are extracted structurally via pdfplumber and each row becomes its
  own chunk, so an LLM reads clean ``Disease - ICD code`` rather than
  flattened soup.
- Any page region that does not match the expected structure falls back to
  paragraph chunking and is logged, never silently dropped.

The parser is configured by a per-document ``DocSpec`` describing which page
ranges to ingest and how to label them, keeping the scoping decision (which
~35 of 172 pages are adjudication-relevant) declarative and reviewable.
"""

import logging
import re
from dataclasses import dataclass, field

import pdfplumber

from eval_engine.ingestion.models import Chunk, ChunkMetadata, Topic
from eval_engine.ingestion.normalize import normalize_clause_text

logger = logging.getLogger(__name__)

# Matches a numbered clause start: "33." or "1)" at line start (2020 uses
# period-style, 2024 uses paren-style). Captures the number and separator.
_CLAUSE_START = re.compile(r"^\s*(\d{1,3})[.)]\s+(.*)", re.DOTALL)


@dataclass
class PageRange:
    """A contiguous span of pages to ingest with shared labelling."""

    start: int  # 1-indexed, inclusive
    end: int  # 1-indexed, inclusive
    section: str | None
    chapter: str | None
    chapter_title: str | None
    topic: Topic
    has_tables: bool = False


@dataclass
class DocSpec:
    """Declarative ingestion scope for one source document."""

    source_doc: str
    doc_year: int
    pdf_path: str
    id_prefix: str  # e.g. "2020"
    ranges: list[PageRange] = field(default_factory=list)


def _context_header(r: PageRange) -> str:
    bits = [b for b in (r.section, r.chapter, r.chapter_title) if b]
    return f"[{' \u00b7 '.join(bits)}]" if bits else ""


def _make_chunk_id(
    prefix: str, r: PageRange, clause: str, page_no: int, *, suffix: str = ""
) -> str:
    parts = [prefix]
    if r.section:
        parts.append("S" + re.sub(r"\D", "", r.section))
    if r.chapter:
        # roman or arabic chapter token, stripped to alphanumerics
        parts.append("CH" + r.chapter.replace("Chapter", "").strip().replace(" ", ""))
    parts.append(f"P{page_no}")
    parts.append("C" + clause)
    cid = "-".join(parts)
    return f"{cid}{suffix}"


def _parse_prose_page(text: str, r: PageRange, spec: DocSpec, page_no: int) -> list[Chunk]:
    """Split a prose page into clause chunks on numbered boundaries."""
    chunks: list[Chunk] = []
    header = _context_header(r)
    # Split into clause blocks: a line starting with "N." begins a new clause.
    blocks: list[tuple[str, list[str]]] = []
    current_num: str | None = None
    current_lines: list[str] = []
    for line in text.split("\n"):
        m = _CLAUSE_START.match(line)
        if m:
            if current_num is not None:
                blocks.append((current_num, current_lines))
            current_num = m.group(1)
            current_lines = [line.strip()]
        elif current_num is not None:
            current_lines.append(line)
    if current_num is not None:
        blocks.append((current_num, current_lines))

    if not blocks:
        # Fallback: no numbered clauses found on a page we expected to parse.
        logger.warning(
            "No numbered clauses on page %d of %s; falling back to paragraph chunk",
            page_no,
            spec.source_doc,
        )
        body = normalize_clause_text(text)
        if body:
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(spec.id_prefix, r, "fallback", page_no),
                    embed_text=f"{header}\n{body}" if header else body,
                    clause_text=body,
                    metadata=ChunkMetadata(
                        source_doc=spec.source_doc,
                        doc_year=spec.doc_year,
                        section=r.section,
                        chapter=r.chapter,
                        chapter_title=r.chapter_title,
                        clause=None,
                        page=page_no,
                        topic=r.topic,
                    ),
                )
            )
        return chunks

    for num, lines in blocks:
        body = normalize_clause_text("\n".join(lines))
        if not body:
            continue
        # Strip the leading "N." or "N)" from the body; the header names the clause.
        body = re.sub(rf"^{re.escape(num)}[.)]\s*", "", body)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(spec.id_prefix, r, num, page_no),
                embed_text=f"{header}\nClause {num}: {body}" if header else f"Clause {num}: {body}",
                clause_text=body,
                metadata=ChunkMetadata(
                    source_doc=spec.source_doc,
                    doc_year=spec.doc_year,
                    section=r.section,
                    chapter=r.chapter,
                    chapter_title=r.chapter_title,
                    clause=num,
                    page=page_no,
                    topic=r.topic,
                ),
            )
        )
    return chunks


def _parse_tables_page(
    page: pdfplumber.page.Page, r: PageRange, spec: DocSpec, page_no: int
) -> list[Chunk]:
    """Each meaningful table row becomes a structured chunk."""
    chunks: list[Chunk] = []
    header = _context_header(r)
    tables = page.extract_tables() or []
    for t_idx, table in enumerate(tables):
        for row in table:
            cells = [(" ".join(c.split()) if c else "") for c in row]
            # Skip header rows / empty rows; first column should be a row number.
            if not cells or not cells[0].isdigit():
                continue
            row_text = " \u2014 ".join(c for c in cells[1:] if c)
            if not row_text:
                continue
            body = f"{cells[1]} (ICD: {cells[2]})" if len(cells) >= 3 and cells[2] else row_text
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(
                        spec.id_prefix, r, "1", page_no, suffix=f"-T{t_idx}R{cells[0]}"
                    ),
                    embed_text=f"{header}\nPermanently excludable disease (item {cells[0]}): {body}"
                    if header
                    else body,
                    clause_text=body,
                    metadata=ChunkMetadata(
                        source_doc=spec.source_doc,
                        doc_year=spec.doc_year,
                        section=r.section,
                        chapter=r.chapter,
                        chapter_title=r.chapter_title,
                        clause="1",
                        page=page_no,
                        topic=r.topic,
                        is_table=True,
                    ),
                )
            )
    return chunks


def parse_document(spec: DocSpec) -> list[Chunk]:
    """Parse one document according to its DocSpec, returning all chunks."""
    all_chunks: list[Chunk] = []
    with pdfplumber.open(spec.pdf_path) as pdf:
        for r in spec.ranges:
            for page_no in range(r.start, r.end + 1):
                page = pdf.pages[page_no - 1]
                text = page.extract_text() or ""
                if r.has_tables:
                    all_chunks.extend(_parse_tables_page(page, r, spec, page_no))
                else:
                    all_chunks.extend(_parse_prose_page(text, r, spec, page_no))
    logger.info("Parsed %d chunks from %s", len(all_chunks), spec.source_doc)
    return all_chunks
