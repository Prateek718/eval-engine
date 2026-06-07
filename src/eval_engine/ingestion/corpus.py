"""Corpus scope — the declarative ingestion plan for both source documents.

This is the single place that encodes the scoping decision: of ~172 total
pages across the two circulars, only the ~35 pages that actually govern claim
adjudication are ingested. Keeping it declarative means a reviewer can see and
challenge the scope without reading parser internals.

Scope rationale (kept here as code comments, not repo prose elsewhere):
- 2024 Master Circular, Chapter I (p4-10): policyholder entitlements + the
  claims-process rules an adjudicator relies on (moratorium, settlement,
  cashless, repudiation procedure).
- 2020 Standardization Circular, Section 1 Chapter I (p5-13): standard
  definitions, including the Pre-Existing Disease definition (clause 33).
- 2020 Section 2 Chapter IV (p100-104): the permanent-exclusions table (the
  16 disease groups insurers may permanently exclude). The operative
  exclusions regime.
Everything else (TPA empanelment, return formats, the model product, ~120
pages of annexures) is deliberately out of scope.
"""

from eval_engine.ingestion.models import Topic
from eval_engine.ingestion.parser import DocSpec, PageRange

MASTER_CIRCULAR_2024 = DocSpec(
    source_doc="irdai_master_circular_health_2024",
    doc_year=2024,
    pdf_path="data/raw/irdai_master_circular_health_2024.pdf",
    id_prefix="2024",
    ranges=[
        PageRange(
            start=4,
            end=10,
            section=None,
            chapter="Chapter I",
            chapter_title="General Information for the Policyholder",
            topic=Topic.POLICYHOLDER_ENTITLEMENT,
        ),
    ],
)

STANDARDIZATION_2020 = DocSpec(
    source_doc="irdai_standardization_health_2020",
    doc_year=2020,
    pdf_path="data/raw/irdai_standardization_health_2020.pdf",
    id_prefix="2020",
    ranges=[
        PageRange(
            start=5,
            end=13,
            section="Section 1",
            chapter="Chapter I",
            chapter_title="Standard Definitions",
            topic=Topic.DEFINITION,
        ),
        PageRange(
            start=100,
            end=104,
            section="Section 2",
            chapter="Chapter IV",
            chapter_title="Existing Diseases allowed to be permanently excluded",
            topic=Topic.PERMANENT_EXCLUSIONS,
            has_tables=True,
        ),
    ],
)

CORPUS: list[DocSpec] = [MASTER_CIRCULAR_2024, STANDARDIZATION_2020]
