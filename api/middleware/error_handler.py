"""Global error handling middleware."""

import logging
import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom API error with status code."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: dict | None = None,
    ):
        """Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            detail: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail


async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle exceptions and return appropriate JSON response.

    Args:
        request: Request that caused the error
        exc: Exception that was raised

    Returns:
        JSON error response
    """
    if isinstance(exc, APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    # Log unexpected errors
    logger.error(
        f"Unhandled error in {request.method} {request.url.path}: {exc}\n"
        f"{traceback.format_exc()}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": None,
        },
    )


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for catching and handling all errors."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and handle any errors.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response or error response
        """
        try:
            return await call_next(request)
        except Exception as exc:
            return await error_handler(request, exc)
