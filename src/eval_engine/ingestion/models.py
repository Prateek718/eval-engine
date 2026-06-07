"""Chunk schema — the contract the golden set, agent, and eval layers depend on.

A chunk is a self-locating clause: it carries its own legible address
(chunk_id) plus enough context to be read standalone. Two text fields exist
on purpose:

- ``embed_text``  : context header + clause body. This is what gets embedded,
  so semantic search benefits from the surrounding context.
- ``clause_text`` : the verbatim regulatory text only. This is what the golden
  set cites, so faithfulness scoring compares against IRDAI's words, never
  our prepended header.

Blurring those two pollutes citation fidelity, so they stay separate.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Topic(StrEnum):
    """Coarse topic tag for Qdrant payload filtering.

    Lets the agent pre-filter the corpus (e.g. only ``permanent_exclusions``)
    before semantic search, turning a blob into a navigable structure.
    """

    DEFINITION = "definition"
    PERMANENT_EXCLUSIONS = "permanent_exclusions"
    ALLOWED_EXCLUSIONS = "allowed_exclusions"
    EXCLUSION_WORDING = "exclusion_wording"
    CLAIMS_PROCESS = "claims_process"
    POLICYHOLDER_ENTITLEMENT = "policyholder_entitlement"
    WAITING_PERIOD = "waiting_period"
    OTHER = "other"


class ChunkMetadata(BaseModel):
    """Structured payload stored alongside the vector in Qdrant."""

    source_doc: str = Field(
        description="Stable source identifier, e.g. 'irdai_standardization_health_2020'"
    )
    doc_year: int = Field(description="Publication year of the source document")
    section: str | None = Field(default=None, description="e.g. 'Section 2'")
    chapter: str | None = Field(default=None, description="e.g. 'Chapter IV'")
    chapter_title: str | None = Field(default=None, description="Human-readable chapter title")
    clause: str | None = Field(default=None, description="Clause/item number, e.g. '1', '33'")
    page: int = Field(description="1-indexed source page number")
    topic: Topic = Field(default=Topic.OTHER, description="Coarse topic tag for payload filtering")
    is_table: bool = Field(
        default=False, description="True if this chunk is a structured table row"
    )


class Chunk(BaseModel):
    """A single retrievable unit of the corpus."""

    chunk_id: str = Field(
        description="Legible address incl. page, e.g. '2020-S1-CHI-P10-C33'. Human + LLM readable."
    )
    embed_text: str = Field(description="Context header + clause body. This is what gets embedded.")
    clause_text: str = Field(
        description="Verbatim regulatory text only. This is what the golden set cites."
    )
    metadata: ChunkMetadata
