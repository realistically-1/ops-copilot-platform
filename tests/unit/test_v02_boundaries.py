from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import CatalogError, LLMError, LLMResponseError, LogBackendError
from app.integrations.catalog.fake import FakeCatalogClient
from app.integrations.llm.client import ChatMessage, OpenAICompatibleLLMClient
from app.integrations.logs.fake import FakeLogClient
from app.models.alert import Alert
from app.models.context import DiagnosisOptions
from app.models.diagnosis import AffectedComponent, DiagnosisDraft, Evidence
from app.models.logs import LogLevel, LogQuery, TimeRange
from app.models.tools import ToolObservation
from app.services.catalog import ServiceCatalog
from app.services.context import DiagnosisContextBuilder
from app.services.evidence import EvidenceConflict, merge_observation
from app.services.logs import LogSearchService
from app.services.reports import DiagnosisReportBuilder
from tests.helpers import draft


def test_context_options_and_catalog_failure(alert: Alert, settings: Settings) -> None:
    catalog = ServiceCatalog.from_fixture()
    builder = DiagnosisContextBuilder(catalog, settings)
    context = builder.build(
        uuid4(), alert, DiagnosisOptions(lookback_minutes=5, lookahead_minutes=0)
    )
    assert context.window.seconds == 300
    assert context.window.end == alert.timestamp
    assert "payment-db" in context.available_components
    with patch.object(catalog, "get", side_effect=CatalogError):
        failed = builder.build(uuid4(), alert, DiagnosisOptions())
    assert failed.service_info is None
    assert any("could not be loaded" in gap for gap in failed.data_gaps)
    with pytest.raises(ValidationError):
        type(context).model_validate({**context.model_dump(), "available_components": ["foreign"]})


async def test_log_contract_filters_and_rejects_scope_leaks(query: LogQuery) -> None:
    client = FakeLogClient.from_fixture()
    service = LogSearchService(client, max_results=1)
    query.levels = [LogLevel.ERROR]
    query.order = "desc"
    result = await service.search(query)
    assert result.entries[0].id == "p006"
    assert result.total == 3
    assert result.statistics.returned_count == 1
    assert any("returned entries only" in warning for warning in result.warnings)
    leaked = result.model_copy(deep=True)
    leaked.entries[0] = leaked.entries[0].model_copy(update={"environment": "staging"})
    with patch.object(client, "search", new=AsyncMock(return_value=leaked)):
        with pytest.raises(LogBackendError):
            await service.search(query)


async def test_fake_integrations_inject_failures_only_when_requested(query: LogQuery) -> None:
    def log_failure(query: LogQuery) -> None:
        raise LogBackendError()

    def catalog_failure(service: str | None) -> None:
        raise CatalogError()

    with pytest.raises(LogBackendError):
        await FakeLogClient([], before_search=log_failure).search(query)
    with pytest.raises(CatalogError):
        ServiceCatalog(FakeCatalogClient([], before_read=catalog_failure)).describe("one")


def test_report_rejects_unregistered_components_and_preserves_timeline(
    alert: Alert, settings: Settings
) -> None:
    context = DiagnosisContextBuilder(ServiceCatalog.from_fixture(), settings).build(
        uuid4(), alert, DiagnosisOptions()
    )
    evidence = Evidence(
        id="log:1",
        source="search_logs",
        summary="error",
        reference="fixture/1",
        simulated=True,
        observed_at=alert.timestamp,
    )
    model_draft = draft(["log:1"])
    model_draft.affected_components = [
        AffectedComponent(
            name="foreign-db", role="dependency", impact="timeout", evidence_ids=["log:1"]
        )
    ]
    builder = DiagnosisReportBuilder()
    with pytest.raises(LLMResponseError):
        builder.build(alert, model_draft, [evidence], [], [], context)
    model_draft.affected_components[0] = model_draft.affected_components[0].model_copy(
        update={"name": "payment-db"}
    )
    result = builder.build(
        alert, model_draft, [evidence], [], [f"gap {i}" for i in range(25)], context
    )
    assert result.affected_components[0].name == "payment-service"
    assert result.timeline[0].simulated
    assert result.assessment.evidence_status == "simulated_context"
    assert len(result.data_gaps) == 20
    assert "omitted" in result.data_gaps[-1]


def test_evidence_collision_does_not_replace_alert(alert: Alert, query: LogQuery) -> None:
    original = Evidence(id="alert:input", source="alert", summary=alert.alert, reference="input")
    forged = original.model_copy(update={"source": "search_logs"})
    observation = ToolObservation(
        tool="search_logs", status="ok", summary="found", evidence=[forged]
    )
    with pytest.raises(EvidenceConflict):
        merge_observation(
            [original], observation, alert, TimeRange(start=query.start, end=query.end)
        )


async def test_llm_size_limits_do_not_retry(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"llm_max_prompt_chars": 1000, "llm_max_response_bytes": 1000}
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"x" * 1001)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatibleLLMClient(settings, http)
        with pytest.raises(LLMError):
            await client.complete_structured(
                [ChatMessage(role="user", content="x" * 1001)], DiagnosisDraft
            )
        assert not requests
        settings.llm_max_prompt_chars = 10000
        with pytest.raises(LLMResponseError):
            await client.complete_structured(
                [ChatMessage(role="user", content="test")], DiagnosisDraft
            )
        assert len(requests) == 1


def test_query_fingerprint_distinguishes_keyword_boundaries(query: LogQuery) -> None:
    first = query.model_copy(update={"keywords": ["one,two"]})
    second = query.model_copy(update={"keywords": ["one", "two"]})
    assert first.fingerprint() != second.fingerprint()
    reordered = query.model_copy(update={"keywords": ["TIMEOUT", "ERROR"]})
    assert reordered.fingerprint() == query.fingerprint()


def test_large_catalog_entry_remains_usable() -> None:
    from app.agents.tools.service_info import GetServiceInfoTool
    from app.models.service import ServiceInfo
    from app.models.tools import ServiceInfoArguments

    info = ServiceInfo(
        service="one",
        environment="prod",
        owner="team",
        language="Python",
        dependencies=[f"dependency-{i}-" + "x" * 80 for i in range(40)],
        deploy_version="1",
    )
    observation = GetServiceInfoTool(ServiceCatalog([info]))(
        ServiceInfoArguments(service="one", environment="prod")
    )
    assert observation.status == "ok"
    assert all(len(item.summary) <= 2000 for item in observation.evidence)
    assert any("shortened" in gap for gap in observation.data_gaps)
    assert len(observation.details["service"]["dependencies"]) == 40
