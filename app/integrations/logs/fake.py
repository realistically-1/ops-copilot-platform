from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files

from pydantic import TypeAdapter

from app.models.logs import LogEntry, LogQuery, LogSearchResult


class FakeLogClient:
    def __init__(
        self, entries: list[LogEntry], before_search: Callable[[LogQuery], None] | None = None
    ) -> None:
        ids = [entry.id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("log fixture IDs must be unique")
        self._entries = tuple(item.model_copy(deep=True) for item in entries)
        self._before_search = before_search

    @classmethod
    def from_fixture(cls) -> FakeLogClient:
        raw = files("app.data").joinpath("logs.json").read_text(encoding="utf-8")
        entries = TypeAdapter(list[LogEntry]).validate_json(raw)
        return cls(entries)

    async def search(self, query: LogQuery) -> LogSearchResult:
        if self._before_search:
            self._before_search(query.model_copy(deep=True))
        matches = [entry for entry in self._entries if entry.matches(query)]
        matches.sort(key=lambda entry: (entry.timestamp, entry.id), reverse=query.order == "desc")
        return LogSearchResult(
            entries=[entry.model_copy(deep=True) for entry in matches[: query.limit]],
            total=len(matches),
            truncated=len(matches) > query.limit,
            backend="fixture-logs-v1",
            simulated=True,
            queried_range=query.time_range,
        )
