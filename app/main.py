from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,Response
from fastapi.responses import JSONResponse
from app.api.errors import AppError
from app.api.routes.support import router as support_router
from app.config import get_settings
from app.middleware import request_id_middleware
from app.observability.logging import configure_logging
from app.observability.middleware import request_logging_middleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

settings = get_settings()
configure_logging()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered support operations API.",
)
@app.exception_handler(AppError)
async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    """
    Convert expected application errors into a consistent API response.
    """

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    return await request_id_middleware(request, call_next)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    return await request_logging_middleware(request, call_next)

@app.exception_handler(AppError)
async def handle_app_error(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "The request contains invalid data.",
            "request_id": request_id,
            "details": {
                "errors": exc.errors(),
            },
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    # Detailed logging will be added in the observability layer.
    # Do not expose the exception details to API consumers.

    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred.",
            "request_id": request_id,
            "details": {},
        },
    )


app.include_router(
    support_router,
    prefix=settings.api_prefix,
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.environment,
    }

@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )