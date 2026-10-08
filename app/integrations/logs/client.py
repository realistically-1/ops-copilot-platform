from typing import Protocol

from app.models.logs import LogQuery, LogSearchResult


class LogClient(Protocol):
    async def search(self, query: LogQuery) -> LogSearchResult: ...
