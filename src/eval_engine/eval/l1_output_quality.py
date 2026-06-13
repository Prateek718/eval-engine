"""L1: output-quality scoring with Ragas.

Three metrics over each adjudication:
- faithfulness: is the agent's answer grounded in the clauses it retrieved?
- answer relevancy: does the answer address the claim?
- context precision: were the retrieved clauses relevant to the reference label?

Scoring runs through an OutputQualityScorer seam. The real implementation wraps
Ragas (LLM-based metrics, real API calls); tests substitute a stub, so the suite
needs no Gemini and no network.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ScoreInput:
    """One adjudication framed for output-quality scoring.

    Two context sets, measured by different metrics:
    - evidence_contexts: everything the decision drew on (policy terms the agent
      looked up plus retrieved clauses). Faithfulness checks the answer against
      this, since the reasoning is grounded in the policy schedule as much as in
      retrieved regulations.
    - retrieved_contexts: only the clauses retrieval returned. Context precision
      checks these against the reference, measuring retrieval quality.
    """

    claim_text: str
    answer_text: str  # agent decision + amount + reasoning, as text
    evidence_contexts: list[str]  # policy terms + retrieved clauses; for faithfulness
    retrieved_contexts: list[str]  # retrieved clauses only; for context precision
    reference_text: str  # the golden label, as text


@dataclass
class OutputQualityScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float


class OutputQualityScorer(Protocol):
    """The seam: scores one ScoreInput on the three L1 metrics."""

    async def score(self, item: ScoreInput) -> OutputQualityScores: ...


def answer_to_text(decision: str, amount: int, reasoning: str) -> str:
    """Flatten an adjudication into the response text Ragas scores."""
    return f"Decision: {decision}. Payable amount: {amount}. Reasoning: {reasoning}"


def label_to_reference(decision: str, amount: int, reasoning: str) -> str:
    """Flatten a golden label into the reference text."""
    return f"Decision: {decision}. Payable amount: {amount}. {reasoning}"


class RagasScorer:
    """Real scorer. Wraps Ragas metrics with a Gemini evaluator LLM.

    Imported lazily so the module loads (and stub-based tests run) without Ragas,
    its langchain deps, or credentials present.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            ResponseRelevancy,
        )

        llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
        )
        embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,  # type: ignore[call-arg]  # langchain pydantic alias
            )
        )
        self._faithfulness = Faithfulness(llm=llm)
        self._relevancy = ResponseRelevancy(llm=llm, embeddings=embeddings)
        self._context_precision = LLMContextPrecisionWithReference(llm=llm)

    async def score(self, item: ScoreInput) -> OutputQualityScores:
        from ragas.dataset_schema import SingleTurnSample

        # Faithfulness checks the answer against the full evidence base.
        evidence_sample = SingleTurnSample(
            user_input=item.claim_text,
            response=item.answer_text,
            retrieved_contexts=item.evidence_contexts or [""],
            reference=item.reference_text,
        )
        # Context precision checks retrieval quality against the reference.
        retrieval_sample = SingleTurnSample(
            user_input=item.claim_text,
            response=item.answer_text,
            retrieved_contexts=item.retrieved_contexts or [""],
            reference=item.reference_text,
        )
        return OutputQualityScores(
            faithfulness=await self._faithfulness.single_turn_ascore(evidence_sample),
            answer_relevancy=await self._relevancy.single_turn_ascore(evidence_sample),
            context_precision=await self._context_precision.single_turn_ascore(retrieval_sample),
        )
