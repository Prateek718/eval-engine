"""Metrics-publisher seam tests. Offline: the null publisher does nothing, and
the real publisher's registry-building is exercised without a live gateway by
intercepting the push call.
"""

import prometheus_client

from eval_engine.observability.metrics import (
    NullMetricsPublisher,
    PrometheusMetricsPublisher,
    build_metrics_publisher,
)


def test_no_gateway_url_yields_null_publisher() -> None:
    publisher = build_metrics_publisher("")
    assert isinstance(publisher, NullMetricsPublisher)
    # publish is a no-op: no gateway, no error, nothing required.
    publisher.publish({"mean_faithfulness": 0.23}, {"faithfulness": False})


def test_gateway_url_yields_real_publisher() -> None:
    publisher = build_metrics_publisher("localhost:9091")
    assert isinstance(publisher, PrometheusMetricsPublisher)


def test_real_publisher_pushes_means_and_drift_flag_gauges(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_push(gateway: str, job: str, registry: object, **kwargs: object) -> None:
        captured["job"] = job
        captured["body"] = prometheus_client.generate_latest(registry).decode()

    monkeypatch.setattr(prometheus_client, "push_to_gateway", fake_push)

    publisher = build_metrics_publisher("localhost:9091")
    publisher.publish(
        {"mean_faithfulness": 0.23, "regression_n_drifted": 2.0},
        {"faithfulness": True, "tool_precision": False},
    )

    assert captured["job"] == "eval-engine"
    body = captured["body"]
    assert "mean_faithfulness 0.23" in body
    assert "eval_drift_faithfulness 1.0" in body  # flagged -> 1
    assert "eval_drift_tool_precision 0.0" in body  # not flagged -> 0
