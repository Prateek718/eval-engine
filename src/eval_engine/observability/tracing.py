"""Tracing seam: Langfuse observability, injected optionally.

Tracing is cross-cutting observability, not agent logic. The agent must behave
identically with or without it, and the test suite must run with no credentials
and no network. So the system depends on the ``Tracer`` Protocol, never on
Langfuse directly, and a no-op tracer is the default.

Two responsibilities live here:
- Open a span per agent run that the LangChain callback handler logs under, and
  surface that run's trace_id (so L1 scores can be bonded to the exact run, and
  L2/L3 can reuse the same id later).
- Write scores onto a trace by id.

The trace_id is pinned explicitly via ``trace_context`` rather than read back
after the fact: the LangChain callback handler does not register its spans as
the active OTel context, so the "current" trace_id is not guaranteed to match
the trace the run actually logged to. Minting the id and forcing both the span
and the handler to adopt it removes that ambiguity.

Built against langfuse v4 (observation-centric model): ``start_as_current_
observation``, ``propagate_attributes`` for trace metadata (``update_trace`` is
gone), and explicit ``Langfuse(...)`` construction (singleton keyed by key).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Score:
    """One metric value to attach to a trace."""

    name: str
    value: float
    comment: str = ""


class Tracer(Protocol):
    """The observability seam.

    ``run_span`` yields (callbacks, trace_id): callbacks go into the LangGraph
    invoke config; trace_id identifies the run for later scoring. Both are
    empty/None when tracing is off, and callers must tolerate that.
    """

    def run_span(self, name: str, claim_id: str) -> Any: ...

    def write_scores(self, trace_id: str, scores: list[Score]) -> None: ...

    def flush(self) -> None: ...


class NullTracer:
    """Default no-op tracer. The agent runs untraced; nothing is sent."""

    @contextmanager
    def run_span(self, name: str, claim_id: str) -> Any:
        yield [], None

    def write_scores(self, trace_id: str, scores: list[Score]) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracer:
    """Real tracer. Imported lazily so this module loads (and stub-based tests
    run) without Langfuse or credentials present.
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        from langfuse import Langfuse, get_client
        from langfuse.langchain import CallbackHandler

        # Construct once from typed settings (singleton keyed by public_key),
        # then take the configured client. Credentials come from the settings
        # object, not ambient os.environ, keeping one config surface.
        Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._client = get_client()
        self._handler = CallbackHandler()

    @contextmanager
    def run_span(self, name: str, claim_id: str) -> Any:
        from langfuse import propagate_attributes

        # Mint the trace id and pin it on the span; the handler created inside
        # this context inherits it, so run and scores share one trace. claim_id
        # is propagated as trace metadata (v4: dict[str, str], <=200 chars).
        trace_id = self._client.create_trace_id()
        with (
            self._client.start_as_current_observation(
                as_type="span",
                name=name,
                trace_context={"trace_id": trace_id},
            ),
            propagate_attributes(trace_name=name, metadata={"claim_id": claim_id}),
        ):
            yield [self._handler], trace_id

    def write_scores(self, trace_id: str, scores: list[Score]) -> None:
        for s in scores:
            self._client.create_score(
                name=s.name,
                value=s.value,
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=s.comment or None,
            )

    def flush(self) -> None:
        self._client.flush()


def build_tracer(public_key: str, secret_key: str, host: str) -> Tracer:
    """Return a real tracer when credentials are present, else the no-op.

    Absent keys is the normal case in tests and offline runs; it is not an
    error. Tracing is observability, so its absence must not change behaviour.
    """
    if public_key and secret_key and host:
        return LangfuseTracer(public_key, secret_key, host)
    return NullTracer()
