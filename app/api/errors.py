from typing import Any


class AppError(Exception):
    """Base exception for expected application errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class AIServiceError(AppError):
    """Raised when the AI service cannot complete a request."""

    def __init__(
        self,
        message: str = "The AI service is temporarily unavailable.",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="AI_SERVICE_ERROR",
            status_code=503,
            details=details,
        )


class SupportAnalysisError(AppError):
    """Raised when support analysis cannot be completed."""

    def __init__(
        self,
        message: str = "Support analysis could not be completed.",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="SUPPORT_ANALYSIS_ERROR",
            status_code=500,
            details=details,
        )


class CaseNotFoundError(AppError):
    """Raised when a requested support case does not exist."""

    def __init__(
        self,
        case_id: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Support case '{case_id}' was not found.",
            code="CASE_NOT_FOUND",
            status_code=404,
            details=details,
        )


class InvalidCaseStatusTransitionError(AppError):
    """
    Raised when a support case lifecycle transition is not allowed.
    """

    def __init__(
        self,
        *,
        case_id: str,
        current_status: str,
        requested_status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            (
                f"Cannot transition case '{case_id}' "
                f"from '{current_status}' "
                f"to '{requested_status}'."
            ),
            code="INVALID_CASE_STATUS_TRANSITION",
            status_code=409,
            details=details
            or {
                "case_id": case_id,
                "current_status": current_status,
                "requested_status": requested_status,
            },
        )