from fastapi.testclient import TestClient

from app.main import app
from app.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
)

client = TestClient(app)


def _counter_value(
    metric,
    *,
    labels: dict[str, str],
) -> float:
    """Read the current value of a labeled Prometheus counter."""
    return float(metric.labels(**labels)._value.get())


def _histogram_count(
    metric,
    *,
    labels: dict[str, str],
) -> float:
    """Read the observation count of a labeled Prometheus histogram."""
    samples = metric.collect()[0].samples

    for sample in samples:
        if (
            sample.name.endswith("_count")
            and all(sample.labels.get(key) == value for key, value in labels.items())
        ):
            return float(sample.value)

    return 0.0


def test_metrics_endpoint_is_available() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "support_http_requests_total" in response.text
    assert "support_http_request_duration_seconds" in response.text


def test_http_metrics_use_route_template() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    request_count = _counter_value(
        HTTP_REQUESTS_TOTAL,
        labels={
            "method": "GET",
            "path": "/health",
            "status_code": "200",
        },
    )

    duration_count = _histogram_count(
        HTTP_REQUEST_DURATION,
        labels={
            "method": "GET",
            "path": "/health",
        },
    )

    assert request_count >= 1
    assert duration_count >= 1


def test_metrics_endpoint_is_not_recorded_as_application_traffic() -> None:
    before = _counter_value(
        HTTP_REQUESTS_TOTAL,
        labels={
            "method": "GET",
            "path": "/metrics",
            "status_code": "200",
        },
    )

    response = client.get("/metrics")

    assert response.status_code == 200

    after = _counter_value(
        HTTP_REQUESTS_TOTAL,
        labels={
            "method": "GET",
            "path": "/metrics",
            "status_code": "200",
        },
    )

    assert after == before


def test_dynamic_case_route_uses_route_template() -> None:
    # Create a real case so FastAPI resolves the dynamic route.
    response = client.post(
        "/api/v1/support/analyze",
        json={
            "customer_message": "Where is my order?",
        },
    )

    assert response.status_code == 200

    case_id = response.json()["case_id"]

    detail_response = client.get(
        f"/api/v1/support/cases/{case_id}"
    )

    assert detail_response.status_code == 200

    assert detail_response.json()["escalation"]["reason_code"] == "policy_pass"

    route_samples = [
        sample
        for sample in HTTP_REQUESTS_TOTAL.collect()[0].samples
        if sample.name == "support_http_requests_total"
    ]

    paths = {
        sample.labels.get("path")
        for sample in route_samples
    }

    assert "/api/v1/support/cases/{case_id}" in paths
    assert f"/api/v1/support/cases/{case_id}" not in paths