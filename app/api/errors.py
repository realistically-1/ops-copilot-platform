import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.schemas.errors import ErrorDetail, ErrorResponse, ValidationIssue
from app.core.exceptions import ApplicationError

logger = logging.getLogger(__name__)


def error_response(
    request: Request,
    code: str,
    message: str,
    status_code: int,
    details: list[ValidationIssue] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unavailable")
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            details=details or [],
            diagnosis_id=getattr(request.state, "diagnosis_id", None),
        )
    )
    return JSONResponse(
        body.model_dump(mode="json", exclude_none=True),
        status_code=status_code,
        headers={"X-Request-ID": request_id},
    )


async def application_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApplicationError)
    return error_response(request, exc.code, exc.public_message, exc.status_code)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        ValidationIssue(
            location=list(error["loc"]),
            message=error["msg"],
            type=error["type"],
        )
        for error in exc.errors()
    ]
    return error_response(request, "validation_error", "Invalid request.", 422, details)


async def http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    response = error_response(request, "http_error", str(exc.detail), exc.status_code)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
