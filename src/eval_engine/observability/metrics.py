"""Metrics seam: Prometheus push-gateway publishing, injected optionally.

Pushing metrics is observability, not eval logic. The eval must behave
identically with or without it, and tests must run with no gateway and without
prometheus_client reaching the network. So the system depends on the
``MetricsPublisher`` Protocol, never on Prometheus directly, and a no-op
publisher is the default.

The eval is a batch job -- it runs, scores, and exits -- not a long-running
service Prometheus can scrape. So a run PUSHES its final metrics to a
Pushgateway, the long-running target Prometheus scrapes. This is the standard
Prometheus idiom for batch jobs. A fresh registry per push means each run
replaces the prior values under its job name rather than accumulating, so the
gateway always reflects the latest run.

Like the tracer and tracker, this is injected behind an interface and is a
no-op when no gateway URL is set -- so tests and offline runs neither reach a
gateway nor need the library.
"""

from __future__ import annotations

from typing import Protocol


class MetricsPublisher(Protocol):
    """Publishes one eval run's session-level metrics.

    ``publish`` takes a flat name->value mapping (the gauges) and a mapping of
    drift-flag name->bool (rendered as 0/1 gauges so per-signal alerts can fire
    in Grafana). A no-op outside a configured gateway.
    """

    def publish(self, metrics: dict[str, float], flags: dict[str, bool]) -> None: ...


class NullMetricsPublisher:
    """Default no-op. Eval runs unpublished; nothing is sent or required."""

    def publish(self, metrics: dict[str, float], flags: dict[str, bool]) -> None:
        return None


class PrometheusMetricsPublisher:
    """Real publisher. prometheus_client is imported lazily so this module (and
    offline tests) load without the library or a gateway present.
    """

    def __init__(self, gateway_url: str, job: str) -> None:
        self._gateway_url = gateway_url
        self._job = job

    def publish(self, metrics: dict[str, float], flags: dict[str, bool]) -> None:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        # Fresh registry per push: each run replaces the prior values under this
        # job, so the gateway reflects the latest run, not a running sum.
        registry = CollectorRegistry()
        for name, value in metrics.items():
            gauge = Gauge(f"eval_{name}", f"eval metric: {name}", registry=registry)
            gauge.set(value)
        # Drift flags as 0/1 gauges, so a Grafana panel/alert can fire per signal.
        for name, flagged in flags.items():
            gauge = Gauge(f"eval_drift_{name}", f"L3 drift flag: {name}", registry=registry)
            gauge.set(1.0 if flagged else 0.0)
        push_to_gateway(self._gateway_url, job=self._job, registry=registry)


def build_metrics_publisher(gateway_url: str, job: str = "eval-engine") -> MetricsPublisher:
    """Real publisher when a gateway URL is set, else the no-op. Absent URL is
    the normal offline/test case, not an error -- publishing must not change
    eval behaviour.
    """
    if gateway_url:
        return PrometheusMetricsPublisher(gateway_url, job)
    return NullMetricsPublisher()
