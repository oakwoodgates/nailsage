"""Common schemas used across the API."""

from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class TimestampMixin(BaseModel):
    """Mixin for models with timestamps (Unix milliseconds)."""

    created_at: Optional[int] = Field(None, description="Creation timestamp (Unix ms)")
    updated_at: Optional[int] = Field(None, description="Last update timestamp (Unix ms)")


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    limit: int = Field(default=100, ge=1, le=1000, description="Number of items to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: List[T] = Field(description="List of items")
    total: int = Field(description="Total number of items available")
    limit: int = Field(description="Limit used for this request")
    offset: int = Field(description="Offset used for this request")
    has_more: bool = Field(description="Whether more items are available")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    detail: Optional[Any] = Field(None, description="Additional error details")


class SuccessResponse(BaseModel):
    """Standard success response."""

    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(description="Success message")
    data: Optional[Any] = Field(None, description="Optional response data")
