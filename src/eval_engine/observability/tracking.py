"""MLflow seam: session-level eval-run tracking.

MLflow records one run per eval *session* (one invocation of an eval script),
logging the aggregate metrics and a few params so scores can be compared across
runs over time. This is deliberately a different granularity from Langfuse,
which holds per-claim traces for drill-down: MLflow answers "did faithfulness
drift between Tuesday and Friday", Langfuse answers "which claim caused it".

The session is the unit because cross-run trend is the point; per-claim logging
would duplicate the traces and lose the session-level row L3 drift needs.

Like the tracer, this is injected behind an interface and is a no-op when no
tracking URI is set -- so tests and offline runs neither reach MLflow nor need
credentials. Auth (username/token) is read by the mlflow library directly from
the environment; this seam only carries the URI and experiment name.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Protocol


class RunTracker(Protocol):
    """Wraps one eval session as a tracked run.

    ``session`` is a context manager: params/metrics logged inside it attach to
    that run. Outside a configured tracker it is a no-op.
    """

    def session(self, run_name: str, params: dict[str, Any]) -> Any: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...


class NullRunTracker:
    """Default no-op. Eval runs untracked; nothing is sent or required."""

    @contextmanager
    def session(self, run_name: str, params: dict[str, Any]) -> Any:
        yield

    def log_metrics(self, metrics: dict[str, float]) -> None:
        return None


class MlflowRunTracker:
    """Real tracker. mlflow is imported lazily so this module (and offline
    tests) load without the library or credentials present.
    """

    def __init__(self, tracking_uri: str, experiment: str) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def session(self, run_name: str, params: dict[str, Any]) -> Any:
        with self._mlflow.start_run(run_name=run_name):
            if params:
                self._mlflow.log_params(params)
            yield

    def log_metrics(self, metrics: dict[str, float]) -> None:
        # mlflow requires numeric metric values; callers pass floats only.
        self._mlflow.log_metrics(metrics)


def build_run_tracker(tracking_uri: str, experiment: str) -> RunTracker:
    """Real tracker when a URI is set, else the no-op. Absent URI is the normal
    offline/test case, not an error -- tracking must not change eval behaviour.
    """
    if tracking_uri:
        return MlflowRunTracker(tracking_uri, experiment)
    return NullRunTracker()
