from unittest.mock import AsyncMock, patch

import pytest

from app.agents.graph import AgentRun
from app.api.schemas.diagnosis import DiagnosisRequest
from app.core.config import Settings
from app.core.exceptions import LLMError
from app.models.alert import Alert
from app.models.diagnosis import DiagnosisResult
from app.services.diagnosis import DiagnosisService
from tests.helpers import ScriptedLLM, draft, finish, make_agent


async def test_service_preserves_agent_result(alert: Alert, settings: Settings) -> None:
    agent = make_agent(ScriptedLLM([]), settings)
    result = DiagnosisResult(
        root_cause="Insufficient evidence",
        confidence=0.1,
        evidence=[],
        recommended_actions=["Review logs"],
        data_gaps=["No backend data"],
    )
    run = AgentRun(result=result, steps=4, tool_calls=0)
    service = DiagnosisService(agent)
    request = DiagnosisRequest.model_validate(alert.model_dump())
    with patch.object(agent, "run", new=AsyncMock(return_value=run)) as invoke:
        response = await service.diagnose(request)
    assert response.result == result
    assert response.service == request.service
    assert response.environment == request.environment
    assert response.execution.steps == 4
    assert response.execution.tool_calls == 0
    assert response.execution.duration_ms >= 0
    invoke.assert_awaited_once()
    passed_alert = invoke.call_args.args[0]
    assert type(passed_alert) is Alert
    assert passed_alert == alert


async def test_service_assigns_unique_ids(alert: Alert, settings: Settings) -> None:
    llm = ScriptedLLM([finish(), draft(), finish(), draft()])
    service = DiagnosisService(make_agent(llm, settings))
    request = DiagnosisRequest.model_validate(alert.model_dump())
    first = await service.diagnose(request)
    second = await service.diagnose(request)
    assert first.diagnosis_id != second.diagnosis_id
    assert first.created_at <= second.created_at


async def test_service_does_not_hide_provider_errors(alert: Alert, settings: Settings) -> None:
    failure = LLMError()
    service = DiagnosisService(make_agent(ScriptedLLM([failure]), settings))
    with pytest.raises(LLMError) as error:
        await service.diagnose(DiagnosisRequest.model_validate(alert.model_dump()))
    assert error.value is failure


async def test_service_logs_identifiable_lifecycle_events(
    alert: Alert, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    llm = ScriptedLLM([finish(), draft()])
    service = DiagnosisService(make_agent(llm, settings))
    with caplog.at_level("INFO", logger="app.services.diagnosis"):
        response = await service.diagnose(DiagnosisRequest.model_validate(alert.model_dump()))
    events = [record for record in caplog.records if record.name == "app.services.diagnosis"]
    assert [record.message for record in events] == ["diagnosis.started", "diagnosis.completed"]
    assert all(record.diagnosis_id == str(response.diagnosis_id) for record in events)
    assert events[-1].step_count == 4
    assert events[-1].duration_ms == response.execution.duration_ms
    assert alert.alert not in caplog.text


async def test_service_failure_logs_public_code(
    alert: Alert, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    service = DiagnosisService(make_agent(ScriptedLLM([LLMError("private")]), settings))
    with caplog.at_level("WARNING", logger="app.services.diagnosis"):
        with pytest.raises(LLMError):
            await service.diagnose(DiagnosisRequest.model_validate(alert.model_dump()))
    record = next(record for record in caplog.records if record.message == "diagnosis.failed")
    assert record.error_code == "llm_error"
    assert "private" not in caplog.text
