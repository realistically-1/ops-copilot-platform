from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.api.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse, ExecutionSummary
from app.models.alert import Alert
from app.models.diagnosis import DiagnosisDraft, DiagnosisResult, Evidence
from app.models.logs import LogQuery
from app.models.tools import ReasoningDecision
from tests.helpers import draft, finish, search, service_info


def test_alert_normalizes_timezone() -> None:
    alert = Alert(
        service=" payment-service ",
        environment="prod",
        alert=" test ",
        timestamp="2026-10-08T12:05:00+08:00",
    )
    assert alert.service == "payment-service"
    assert alert.alert == "test"
    assert alert.timestamp == datetime(2026, 10, 8, 4, 5, tzinfo=UTC)
    assert alert.timestamp.tzinfo == UTC


@pytest.mark.parametrize(
    "field,value",
    [
        ("service", ""),
        ("service", "../../secret"),
        ("service", "with space"),
        ("service", "a" * 101),
        ("environment", ""),
        ("environment", "prod/us"),
        ("environment", "a" * 41),
        ("alert", "   "),
        ("alert", "x" * 4001),
        ("timestamp", "2026-10-08T04:05:00"),
        ("timestamp", "not-a-date"),
    ],
)
def test_invalid_alert_fields(alert: Alert, field: str, value: object) -> None:
    payload = alert.model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        DiagnosisRequest.model_validate(payload)


def test_alert_rejects_extra_fields(alert: Alert) -> None:
    with pytest.raises(ValidationError):
        Alert.model_validate({**alert.model_dump(), "tenant": "unexpected"})


def test_alert_is_immutable(alert: Alert) -> None:
    with pytest.raises(ValidationError):
        alert.service = "other"


def test_keywords_are_trimmed_deduplicated(query: LogQuery) -> None:
    payload = {**query.model_dump(), "keywords": [" ERROR ", "error", " timeout "]}
    result = LogQuery.model_validate(payload)
    assert result.keywords == ["ERROR", "timeout"]


@pytest.mark.parametrize("keywords", [[""], [" "], ["a" * 101], [str(i) for i in range(9)]])
def test_invalid_keywords(query: LogQuery, keywords: list[str]) -> None:
    with pytest.raises(ValidationError):
        LogQuery.model_validate({**query.model_dump(), "keywords": keywords})


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_log_query_limit(query: LogQuery, limit: int) -> None:
    with pytest.raises(ValidationError):
        LogQuery.model_validate({**query.model_dump(), "limit": limit})


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=-1), timedelta(hours=4)])
def test_log_query_window(query: LogQuery, offset: timedelta) -> None:
    with pytest.raises(ValidationError):
        LogQuery.model_validate({**query.model_dump(), "end": query.start + offset})


def test_log_query_three_hour_boundary(query: LogQuery) -> None:
    result = LogQuery.model_validate(
        {**query.model_dump(), "end": query.start + timedelta(hours=3)}
    )
    assert result.end - result.start == timedelta(hours=3)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_diagnosis_confidence_bounds(confidence: float) -> None:
    payload = {**draft().model_dump(), "confidence": confidence}
    with pytest.raises(ValidationError):
        DiagnosisDraft.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_cause", " "),
        ("root_cause", "a" * 2001),
        ("recommended_actions", []),
        ("recommended_actions", [""]),
        ("recommended_actions", ["check"] * 13),
        ("data_gaps", ["missing"] * 21),
    ],
)
def test_diagnosis_text_constraints(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        DiagnosisDraft.model_validate({**draft().model_dump(), field: value})


def test_result_rejects_duplicate_evidence() -> None:
    evidence = Evidence(id="log:1", source="search_logs", summary="timeout", reference="logs/1")
    with pytest.raises(ValidationError, match="unique"):
        DiagnosisResult(
            root_cause="unknown",
            confidence=0.1,
            evidence=[evidence, evidence],
            recommended_actions=["check logs"],
            data_gaps=[],
        )


def test_result_json_round_trip() -> None:
    result = DiagnosisResult(
        root_cause="连接池耗尽",
        confidence=0.5,
        evidence=[Evidence(id="log:1", source="search_logs", summary="超时", reference="logs/1")],
        recommended_actions=["检查连接数"],
        data_gaps=["缺少实时指标"],
    )
    assert DiagnosisResult.model_validate_json(result.model_dump_json()) == result


def test_decision_requires_tool_for_tool_action() -> None:
    with pytest.raises(ValidationError):
        ReasoningDecision(action="tool", rationale="need logs", tool_call=None)


def test_decision_rejects_tool_for_diagnose_action() -> None:
    with pytest.raises(ValidationError):
        ReasoningDecision(action="diagnose", rationale="done", tool_call=service_info().tool_call)


def test_decision_discriminated_union(query: LogQuery) -> None:
    for decision in [finish(), service_info(), search(query)]:
        restored = ReasoningDecision.model_validate_json(decision.model_dump_json())
        assert restored == decision


def test_decision_rejects_unknown_tool() -> None:
    payload = service_info().model_dump()
    payload["tool_call"]["name"] = "execute_shell"
    with pytest.raises(ValidationError):
        ReasoningDecision.model_validate(payload)


def test_response_schema_exposes_structured_result() -> None:
    schema = DiagnosisResponse.model_json_schema()
    assert schema["properties"]["result"]["$ref"].endswith("DiagnosisResult")
    assert set(schema["$defs"]["DiagnosisResult"]["required"]) == {
        "root_cause",
        "confidence",
        "evidence",
        "recommended_actions",
        "data_gaps",
    }


def test_execution_summary_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        ExecutionSummary(steps=4, tool_calls=0, duration_ms=-1)
