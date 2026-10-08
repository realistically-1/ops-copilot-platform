import asyncio
import logging

from app.core.exceptions import LogBackendError
from app.integrations.logs.client import LogClient
from app.models.diagnosis import Evidence
from app.models.logs import LogQuery
from app.models.tools import ToolObservation
from app.services.logs import LogSearchService

logger = logging.getLogger(__name__)


class SearchLogsTool:
    name = "search_logs"
    description = "Search service logs in a half-open time range; keywords are OR-matched."

    def __init__(self, client: LogClient) -> None:
        self._client = client if isinstance(client, LogSearchService) else LogSearchService(client)

    async def __call__(self, query: LogQuery) -> ToolObservation:
        try:
            async with asyncio.timeout(5):
                result = await self._client.search(query)
        except (LogBackendError, TimeoutError) as exc:
            logger.warning(
                "log_search.failed",
                extra={
                    "tool": self.name,
                    "service": query.service,
                    "error_code": "logs_unavailable",
                },
            )
            return ToolObservation(
                tool=self.name,
                status="error",
                summary="Log search failed.",
                data_gaps=["Log backend did not return data for the requested window."],
                error_code="logs_timeout" if isinstance(exc, TimeoutError) else "logs_unavailable",
            )
        gaps: list[str] = list(result.warnings)
        if result.simulated:
            gaps.append("Logs come from a static fixture; live production logs were not queried.")
        if result.truncated:
            gaps.append(f"Log search returned {len(result.entries)} of {result.total} matches.")
        if not result.entries:
            gaps.append("No logs matched the requested service, environment, time, and keywords.")
        evidence = [
            Evidence(
                id=f"log:{entry.id}",
                source=self.name,
                summary=f"{entry.timestamp.isoformat()} {entry.level}: {entry.message}",
                reference=f"{result.backend}/{entry.service}/{entry.environment}/{entry.id}",
                simulated=result.simulated,
                observed_at=entry.timestamp,
                service=entry.service,
                environment=entry.environment,
                attributes=entry.attributes,
            )
            for entry in result.entries
        ]
        logger.info(
            "log_search.completed",
            extra={"tool": self.name, "service": query.service, "result_count": len(evidence)},
        )
        return ToolObservation(
            tool=self.name,
            status="ok" if evidence else "empty",
            summary=f"Found {result.total} matching logs; returned {len(evidence)}.",
            evidence=evidence,
            details={
                "statistics": result.statistics.model_dump(mode="json")
                if result.statistics
                else None,
                "patterns": [item.model_dump(mode="json") for item in result.patterns],
                "facets": {
                    key: [item.model_dump(mode="json") for item in values]
                    for key, values in result.facets.items()
                },
                "total_matches": result.total,
                "returned_entries": len(result.entries),
            },
            data_gaps=gaps,
        )
