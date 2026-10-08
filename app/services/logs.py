import logging
from collections import Counter

from app.core.exceptions import LogBackendError
from app.integrations.logs.client import LogClient
from app.models.logs import (
    LogEntry,
    LogFacet,
    LogLevel,
    LogPattern,
    LogQuery,
    LogSearchResult,
    LogStatistics,
)

logger = logging.getLogger(__name__)


class LogSearchService:
    def __init__(self, client: LogClient, max_results: int = 100) -> None:
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        self._client = client
        self._max_results = max_results

    async def search(self, query: LogQuery) -> LogSearchResult:
        normalized = LogQuery.model_validate(query.model_dump())
        normalized.limit = min(normalized.limit, self._max_results)
        result = await self._client.search(normalized)
        self._validate_response(result, normalized)
        entries = sorted(
            result.entries,
            key=lambda entry: (entry.timestamp, entry.id),
            reverse=normalized.order == "desc",
        )
        warnings = list(result.warnings)
        if normalized.limit < query.limit:
            warnings.append(f"Requested limit was capped at {normalized.limit}.")
        if result.truncated:
            warnings.append("Statistics and patterns describe returned entries only.")
        if not entries:
            warnings.append("An empty result does not establish service health.")
        logger.info(
            "logs.query_completed",
            extra={
                "service": query.service,
                "environment": query.environment,
                "result_count": len(entries),
            },
        )
        return LogSearchResult(
            entries=entries,
            total=result.total,
            truncated=result.truncated,
            backend=result.backend,
            simulated=result.simulated,
            statistics=self._statistics(entries),
            patterns=self._patterns(entries),
            facets=self._facets(entries),
            warnings=warnings,
            queried_range=normalized.time_range,
        )

    @staticmethod
    def _validate_response(result: LogSearchResult, query: LogQuery) -> None:
        if len(result.entries) > query.limit:
            raise LogBackendError("Log backend exceeded the requested limit.")
        ids = [entry.id for entry in result.entries]
        if len(ids) != len(set(ids)):
            raise LogBackendError("Log backend returned duplicate IDs.")
        if any(not entry.matches(query) for entry in result.entries):
            raise LogBackendError("Log backend returned data outside the query scope.")
        if result.total < len(result.entries) or result.truncated != (
            result.total > len(result.entries)
        ):
            raise LogBackendError("Log backend returned inconsistent counts.")

    @staticmethod
    def _statistics(entries: list[LogEntry]) -> LogStatistics:
        counts = Counter(entry.level for entry in entries)
        timestamps = [entry.timestamp for entry in entries]
        instances = sorted(
            {
                instance
                for entry in entries
                if (instance := entry.attributes.get("pod") or entry.attributes.get("instance"))
            }
        )
        return LogStatistics(
            level_counts=dict(counts),
            first_seen=min(timestamps, default=None),
            last_seen=max(timestamps, default=None),
            error_count=counts[LogLevel.ERROR] + counts[LogLevel.CRITICAL],
            returned_count=len(entries),
            distinct_instances=instances,
        )

    @staticmethod
    def _patterns(entries: list[LogEntry]) -> list[LogPattern]:
        groups: dict[tuple[LogLevel, str], list[LogEntry]] = {}
        for entry in entries:
            groups.setdefault((entry.level, entry.message), []).append(entry)
        patterns = []
        for (level, message), group in groups.items():
            patterns.append(
                LogPattern(
                    message=message,
                    level=level,
                    count=len(group),
                    sample_ids=[entry.id for entry in group[:3]],
                    first_seen=min(entry.timestamp for entry in group),
                    last_seen=max(entry.timestamp for entry in group),
                )
            )
        patterns.sort(key=lambda pattern: (-pattern.count, -pattern.level.rank, pattern.message))
        return patterns[:10]

    @staticmethod
    def _facets(entries: list[LogEntry]) -> dict[str, list[LogFacet]]:
        facets: dict[str, list[LogFacet]] = {}
        for attribute in ("dependency", "pod", "instance", "trace_id"):
            groups: dict[str, list[str]] = {}
            for entry in entries:
                value = entry.attributes.get(attribute)
                if value:
                    groups.setdefault(value, []).append(entry.id)
            if groups:
                ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
                facets[attribute] = [
                    LogFacet(value=value, count=len(ids), sample_ids=ids[:3])
                    for value, ids in ordered[:10]
                ]
        return facets
