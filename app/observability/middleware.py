import logging
import re
import time

from fastapi import Request

from app.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
)


logger = logging.getLogger("app.request")


_CASE_ROUTE_PATTERN = re.compile(
    r"^/api/v1/support/cases/[^/]+(?:/(events))?$"
)


def _get_metric_path(request: Request) -> str:
    """
    Return a low-cardinality route label for Prometheus metrics.

    FastAPI route resolution is not available yet when HTTP middleware
    executes, so dynamic resource identifiers are normalized explicitly.
    """
    path = request.url.path

    match = _CASE_ROUTE_PATTERN.match(path)

    if match:
        if match.group(1) == "events":
            return "/api/v1/support/cases/{case_id}/events"

        return "/api/v1/support/cases/{case_id}"

    return path


async def request_logging_middleware(
    request: Request,
    call_next,
):
    """
    Record HTTP request metrics and structured request logs.

    Prometheus metrics use low-cardinality route labels. Concrete request
    paths remain available in structured logs for debugging.

    The /metrics endpoint is excluded from application HTTP metrics so
    Prometheus scraping does not distort API traffic measurements.
    """
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    metric_path = _get_metric_path(request)

    try:
        response = await call_next(request)

        duration_seconds = time.perf_counter() - start_time
        duration_ms = duration_seconds * 1000

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=metric_path,
            status_code=str(response.status_code),
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=metric_path,
        ).observe(duration_seconds)

        logger.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "route": metric_path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response

    except Exception:
        duration_seconds = time.perf_counter() - start_time
        duration_ms = duration_seconds * 1000

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=metric_path,
            status_code="500",
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=metric_path,
        ).observe(duration_seconds)

        logger.exception(
            "request_failed",
            extra={
                "event": "request_failed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "route": metric_path,
                "status_code": 500,
                "duration_ms": round(duration_ms, 2),
            },
        )

        raise