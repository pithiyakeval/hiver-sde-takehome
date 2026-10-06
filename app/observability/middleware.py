import logging
import time

from fastapi import Request

from app.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
)


logger = logging.getLogger("app.request")


async def request_logging_middleware(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    try:
        response = await call_next(request)

        duration_seconds = time.perf_counter() - start_time
        duration_ms = duration_seconds * 1000

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(response.status_code),
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=request.url.path,
        ).observe(duration_seconds)

        logger.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
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
            path=request.url.path,
            status_code="500",
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=request.url.path,
        ).observe(duration_seconds)

        logger.exception(
            "request_failed",
            extra={
                "event": "request_failed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration_ms, 2),
            },
        )

        raise