import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.nodes.parse_alert import ParseAlertNode
from app.agents.nodes.tools import ToolsNode
from app.agents.state import DiagnosisState
from app.agents.tools.search_logs import SearchLogsTool
from app.agents.tools.service_info import GetServiceInfoTool
from app.core.config import Settings
from app.core.exceptions import CatalogError, LogBackendError
from app.integrations.logs.fake import FakeLogClient
from app.models.alert import Alert
from app.models.logs import LogQuery
from app.models.tools import ServiceInfoArguments
from app.services.catalog import ServiceCatalog
from tests.helpers import search, service_info


async def test_search_tool_produces_source_evidence(query: LogQuery) -> None:
    observation = await SearchLogsTool(FakeLogClient.from_fixture())(query)
    assert observation.status == "ok"
    assert len(observation.evidence) == 3
    assert all(item.source == "search_logs" for item in observation.evidence)
    assert all(item.simulated for item in observation.evidence)
    assert observation.evidence[0].id == "log:p003"
    assert "fixture-logs-v1" in observation.evidence[0].reference
    assert any("static fixture" in gap for gap in observation.data_gaps)


async def test_search_tool_empty_results(query: LogQuery) -> None:
    observation = await SearchLogsTool(FakeLogClient([]))(query)
    assert observation.status == "empty"
    assert observation.evidence == []
    assert observation.error_code is None
    assert any("No logs matched" in gap for gap in observation.data_gaps)


async def test_search_tool_truncation_is_visible(query: LogQuery) -> None:
    query.limit = 1
    observation = await SearchLogsTool(FakeLogClient.from_fixture())(query)
    assert len(observation.evidence) == 1
    assert any("1 of 3" in gap for gap in observation.data_gaps)


async def test_search_tool_backend_exception(query: LogQuery) -> None:
    client = FakeLogClient([])
    with patch.object(
        client, "search", new=AsyncMock(side_effect=LogBackendError("private detail"))
    ):
        observation = await SearchLogsTool(client)(query)
    assert observation.status == "error"
    assert observation.error_code == "logs_unavailable"
    assert "private detail" not in observation.model_dump_json()
    assert observation.evidence == []


async def test_search_tool_timeout(query: LogQuery) -> None:
    client = FakeLogClient([])
    with patch.object(client, "search", new=AsyncMock(side_effect=TimeoutError)):
        observation = await SearchLogsTool(client)(query)
    assert observation.status == "error"
    assert observation.error_code == "logs_timeout"


async def test_search_tool_does_not_swallow_cancellation(query: LogQuery) -> None:
    client = FakeLogClient([])
    with patch.object(client, "search", new=AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError):
            await SearchLogsTool(client)(query)


def test_service_tool_returns_all_metadata() -> None:
    tool = GetServiceInfoTool(ServiceCatalog.from_fixture())
    result = tool(ServiceInfoArguments(service="payment-service", environment="prod"))
    assert result.status == "ok"
    evidence = {item.id.rsplit(":", 1)[-1]: item.summary for item in result.evidence}
    assert evidence["owner"] == "owner: payments-platform"
    assert evidence["language"] == "language: Python 3.12"
    assert "payment-db" in evidence["dependencies"]
    assert "2026.10.08-rc2" in evidence["deploy_version"]
    assert evidence["metadata.replicas"] == "metadata.replicas: 4"
    assert all(item.simulated for item in result.evidence)


def test_service_tool_handles_unknown_service() -> None:
    tool = GetServiceInfoTool(ServiceCatalog.from_fixture())
    result = tool(ServiceInfoArguments(service="unknown", environment="prod"))
    assert result.status == "empty"
    assert result.evidence == []
    assert "unknown/prod" in result.data_gaps[0]


def test_service_tool_does_not_use_other_environment() -> None:
    tool = GetServiceInfoTool(ServiceCatalog.from_fixture())
    result = tool(ServiceInfoArguments(service="ledger-service", environment="staging"))
    assert result.status == "empty"


def test_service_tool_catalog_exception() -> None:
    catalog = ServiceCatalog([])
    with patch.object(catalog, "get", side_effect=CatalogError("internal address")):
        result = GetServiceInfoTool(catalog)(ServiceInfoArguments(service="a", environment="prod"))
    assert result.error_code == "catalog_unavailable"
    assert "internal address" not in result.model_dump_json()


@pytest.fixture
def state(alert: Alert, settings: Settings, query: LogQuery) -> DiagnosisState:
    initial: DiagnosisState = {
        "alert": alert,
        "observations": [],
        "evidence": [],
        "data_gaps": [],
        "tool_calls": 0,
        "executed_calls": [],
        "steps": 0,
    }
    initial.update(ParseAlertNode(settings)(initial))
    initial["decision"] = search(query)
    return initial


@pytest.fixture
def node(settings: Settings) -> ToolsNode:
    return ToolsNode(
        SearchLogsTool(FakeLogClient.from_fixture()),
        GetServiceInfoTool(ServiceCatalog.from_fixture()),
        settings,
    )


async def test_tools_node_merges_evidence(node: ToolsNode, state: DiagnosisState) -> None:
    result = await node(state)
    assert result["tool_calls"] == 1
    assert result["steps"] == 2
    assert result["evidence"][0].id == "alert:input"
    assert len(result["evidence"]) == 4
    assert len(result["executed_calls"]) == 1
    assert len(state["evidence"]) == 1


async def test_tools_node_rejects_repeated_call(node: ToolsNode, state: DiagnosisState) -> None:
    state.update(await node(state))
    update = await node(state)
    assert update["observations"][-1].error_code == "invalid_tool_request"
    assert "already attempted" in update["observations"][-1].summary
    assert len(update["evidence"]) == 4
    assert update["tool_calls"] == 2


@pytest.mark.parametrize("service,environment", [("other", "prod"), ("payment-service", "staging")])
async def test_tools_node_rejects_scope(
    node: ToolsNode, state: DiagnosisState, service: str, environment: str
) -> None:
    state["decision"] = service_info(service, environment)
    result = await node(state)
    assert result["observations"][-1].error_code == "invalid_tool_request"
    assert result["evidence"] == state["evidence"]


async def test_tools_node_catches_unexpected_adapter_error(
    state: DiagnosisState, settings: Settings
) -> None:
    backend = FakeLogClient([])
    node = ToolsNode(SearchLogsTool(backend), GetServiceInfoTool(ServiceCatalog([])), settings)
    with patch.object(backend, "search", new=AsyncMock(side_effect=RuntimeError("secret"))):
        result = await node(state)
    observation = result["observations"][-1]
    assert observation.error_code == "tool_failed"
    assert "secret" not in observation.model_dump_json()
    assert result["tool_calls"] == 1


async def test_tools_node_preserves_cancellation(state: DiagnosisState, settings: Settings) -> None:
    backend = FakeLogClient([])
    node = ToolsNode(SearchLogsTool(backend), GetServiceInfoTool(ServiceCatalog([])), settings)
    with patch.object(backend, "search", new=AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError):
            await node(state)


async def test_tools_node_caps_log_limit(state: DiagnosisState, settings: Settings) -> None:
    backend = FakeLogClient.from_fixture()
    limited = settings.model_copy(update={"log_result_limit": 1})
    node = ToolsNode(SearchLogsTool(backend), GetServiceInfoTool(ServiceCatalog([])), limited)
    result = await node(state)
    assert len(result["observations"][-1].evidence) == 1
    assert any("1 of 3" in gap for gap in result["data_gaps"])


async def test_tools_node_requires_call(node: ToolsNode, state: DiagnosisState) -> None:
    from tests.helpers import finish

    state["decision"] = finish()
    with pytest.raises(ValueError, match="requires a tool call"):
        await node(state)
