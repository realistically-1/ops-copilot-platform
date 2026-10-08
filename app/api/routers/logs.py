import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.schemas.errors import ErrorResponse
from app.core.exceptions import LogBackendError
from app.models.logs import LogQuery, LogSearchResult
from app.services.logs import LogSearchService

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


def get_log_service(request: Request) -> LogSearchService:
    return request.app.state.log_service


@router.post("/search", response_model=LogSearchResult, responses={502: {"model": ErrorResponse}})
async def search_logs(
    query: LogQuery,
    service: Annotated[LogSearchService, Depends(get_log_service)],
) -> LogSearchResult:
    try:
        async with asyncio.timeout(5):
            return await service.search(query)
    except TimeoutError as exc:
        raise LogBackendError("Log search timed out.") from exc
