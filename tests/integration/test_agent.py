import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import AgentLimitError, LLMError, LLMTimeoutError, LogBackendError
from app.integrations.logs.fake import FakeLogClient
from app.models.alert import Alert
from app.models.logs import LogQuery
from tests.helpers import ScriptedLLM, draft, finish, make_agent, search, service_info

pytestmark = pytest.mark.integration


async def test_agent_full_investigation(alert: Alert, settings: Settings, query: LogQuery) -> None:
    llm = ScriptedLLM(
        [
            service_info(),
            search(query),
            finish(),
            draft(["alert:input", "log:p003", "service:payment-service:prod:deploy_version"]),
        ]
    )
    run = await make_agent(llm, settings).run(alert)
    assert run.steps == 8
    assert run.tool_calls == 2
    assert [item.id for item in run.result.evidence] == [
        "alert:input",
        "log:p003",
        "service:payment-service:prod:deploy_version",
    ]
    assert "connection acquisition timeout" in run.result.evidence[1].summary
    assert any("static fixture" in gap for gap in run.result.data_gaps)
    assert any("fixture snapshot" in gap for gap in run.result.data_gaps)
    assert len(llm.calls) == 4
    assert not llm.responses


async def test_agent_can_diagnose_without_tools(alert: Alert, settings: Settings) -> None:
    llm = ScriptedLLM([finish(), draft()])
    run = await make_agent(llm, settings).run(alert)
    assert run.steps == 4
    assert run.tool_calls == 0
    assert any("No independent" in gap for gap in run.result.data_gaps)


async def test_agent_stops_at_tool_budget(
    alert: Alert, settings: Settings, query: LogQuery
) -> None:
    settings = settings.model_copy(update={"agent_max_tool_calls": 1})
    llm = ScriptedLLM([search(query), draft(["log:p003"])])
    run = await make_agent(llm, settings).run(alert)
    assert run.tool_calls == 1
    assert run.steps == 6
    assert len(llm.calls) == 2
    assert any("tool budget" in gap for gap in run.result.data_gaps)


async def test_agent_zero_tool_budget(alert: Alert, settings: Settings) -> None:
    settings = settings.model_copy(update={"agent_max_tool_calls": 0})
    llm = ScriptedLLM([draft()])
    run = await make_agent(llm, settings).run(alert)
    assert run.tool_calls == 0
    assert run.steps == 4
    assert len(llm.calls) == 1


async def test_agent_step_limit_is_application_error(alert: Alert, settings: Settings) -> None:
    settings = settings.model_copy(update={"agent_max_steps": 1})
    llm = ScriptedLLM([])
    with pytest.raises(AgentLimitError):
        await make_agent(llm, settings).run(alert)
    assert not llm.calls


async def test_tool_failure_becomes_data_gap(
    alert: Alert, settings: Settings, query: LogQuery
) -> None:
    logs = FakeLogClient([])
    llm = ScriptedLLM([search(query), finish(), draft()])
    with patch.object(logs, "search", new=AsyncMock(side_effect=LogBackendError)):
        run = await make_agent(llm, settings, logs=logs).run(alert)
    assert run.tool_calls == 1
    assert any("backend did not return" in gap for gap in run.result.data_gaps)
    assert "logs_unavailable" in llm.calls[1][0][-1].content


async def test_unexpected_tool_failure_does_not_abort_graph(
    alert: Alert, settings: Settings, query: LogQuery
) -> None:
    logs = FakeLogClient([])
    llm = ScriptedLLM([search(query), finish(), draft()])
    with patch.object(logs, "search", new=AsyncMock(side_effect=RuntimeError("adapter bug"))):
        run = await make_agent(llm, settings, logs=logs).run(alert)
    assert run.tool_calls == 1
    assert any("did not return usable context" in gap for gap in run.result.data_gaps)


async def test_agent_recovers_context_after_failed_tool(
    alert: Alert, settings: Settings, query: LogQuery
) -> None:
    logs = FakeLogClient([])
    llm = ScriptedLLM(
        [
            search(query),
            service_info(),
            finish(),
            draft(["service:payment-service:prod:owner"]),
        ]
    )
    with patch.object(logs, "search", new=AsyncMock(side_effect=LogBackendError)):
        run = await make_agent(llm, settings, logs=logs).run(alert)
    assert run.result.evidence[0].source == "get_service_info"
    assert run.tool_calls == 2
    assert any("Log backend" in gap for gap in run.result.data_gaps)


@pytest.mark.parametrize("error", [LLMError(), LLMTimeoutError()])
async def test_agent_propagates_model_failure(
    alert: Alert, settings: Settings, error: Exception
) -> None:
    llm = ScriptedLLM([error])
    with pytest.raises(type(error)):
        await make_agent(llm, settings).run(alert)
    assert len(llm.calls) == 1


async def test_agent_repeated_query_is_not_executed_twice(
    alert: Alert, settings: Settings, query: LogQuery
) -> None:
    logs = FakeLogClient.from_fixture()
    llm = ScriptedLLM([search(query), search(query), finish(), draft(["log:p003"])])
    with patch.object(logs, "search", wraps=logs.search) as mocked:
        run = await make_agent(llm, settings, logs=logs).run(alert)
    assert mocked.call_count == 1
    assert run.tool_calls == 2
    assert any("already attempted" in gap for gap in run.result.data_gaps)


async def test_sequential_runs_have_independent_state(alert: Alert, settings: Settings) -> None:
    llm = ScriptedLLM([service_info(), finish(), draft(), finish(), draft()])
    agent = make_agent(llm, settings)
    first = await agent.run(alert)
    second = await agent.run(alert)
    assert first.tool_calls == 1
    assert second.tool_calls == 0
    assert not any("fixture snapshot" in gap for gap in second.result.data_gaps)
    assert second.steps == 4


async def test_agent_cancellation_is_not_a_report(
    alert: Alert, settings: Settings, query: LogQuery
) -> None:
    logs = FakeLogClient([])
    llm = ScriptedLLM([search(query)])
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_search(query: LogQuery) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with patch.object(logs, "search", new=AsyncMock(side_effect=blocking_search)):
        task = asyncio.create_task(make_agent(llm, settings, logs=logs).run(alert))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(cancelled.wait(), timeout=2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
