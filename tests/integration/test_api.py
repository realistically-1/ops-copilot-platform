import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from app.agents.graph import AgentRun
from app.api.dependencies import get_diagnosis_service
from app.api.schemas.diagnosis import DiagnosisResponse
from app.core.config import Settings
from app.core.exceptions import (
    AgentLimitError,
    ConfigurationError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.main import create_app
from app.models.alert import Alert
from app.models.diagnosis import DiagnosisResult
from app.models.logs import LogQuery
from app.services.diagnosis import DiagnosisService
from tests.helpers import ScriptedLLM, draft, finish, make_agent, search, service_info

pytestmark = pytest.mark.integration


@pytest.fixture
def api(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
async def client(api: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    with patch("app.main.configure_logging"):
        async with api.router.lifespan_context(api):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api), base_url="http://test"
            ) as client:
                yield client


def use_service(api: FastAPI, service: DiagnosisService) -> None:
    api.dependency_overrides[get_diagnosis_service] = lambda: service


async def test_api_success(
    api: FastAPI, client: httpx.AsyncClient, settings: Settings, alert: Alert, query: LogQuery
) -> None:
    llm = ScriptedLLM([service_info(), search(query), finish(), draft(["log:p003"])])
    use_service(api, DiagnosisService(make_agent(llm, settings)))
    response = await client.post(
        "/api/v1/diagnosis",
        json=alert.model_dump(mode="json"),
        headers={"X-Request-ID": "incident-42"},
    )
    assert response.status_code == 200
    body = DiagnosisResponse.model_validate(response.json())
    assert body.service == alert.service
    assert body.environment == alert.environment
    assert body.diagnosis_id.version == 4
    assert body.created_at.tzinfo is not None
    assert body.execution.tool_calls == 2
    assert body.execution.steps == 8
    assert body.execution.duration_ms >= 0
    assert body.result.confidence == 0.7
    assert body.result.evidence[0].id == "log:p003"
    assert response.headers["x-request-id"] == "incident-42"


@pytest.mark.parametrize("missing", ["service", "environment", "alert", "timestamp"])
async def test_api_missing_fields(client: httpx.AsyncClient, alert: Alert, missing: str) -> None:
    payload = alert.model_dump(mode="json")
    del payload[missing]
    response = await client.post("/api/v1/diagnosis", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert any(item["location"] == ["body", missing] for item in error["details"])
    assert error["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp", "2026-10-08T04:05:00"),
        ("alert", " "),
        ("service", "../path"),
        ("environment", "prod us"),
        ("unexpected", "sensitive-value"),
    ],
)
async def test_api_invalid_values(
    client: httpx.AsyncClient, alert: Alert, field: str, value: str
) -> None:
    payload = {**alert.model_dump(mode="json"), field: value}
    response = await client.post("/api/v1/diagnosis", json=payload)
    assert response.status_code == 422
    assert "sensitive-value" not in response.text
    assert response.json()["error"]["details"]


async def test_api_invalid_json(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/diagnosis", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("body", [None, [], "text", 42])
async def test_api_requires_object(client: httpx.AsyncClient, body: object) -> None:
    response = await client.post("/api/v1/diagnosis", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "error,status,code",
    [
        (ConfigurationError("secret"), 503, "configuration_error"),
        (LLMTimeoutError("secret"), 504, "llm_timeout"),
        (LLMError("secret"), 502, "llm_error"),
        (LLMResponseError("secret"), 502, "invalid_model_output"),
        (AgentLimitError("secret"), 422, "agent_step_limit"),
    ],
)
async def test_api_application_errors(
    api: FastAPI,
    client: httpx.AsyncClient,
    settings: Settings,
    alert: Alert,
    error: Exception,
    status: int,
    code: str,
) -> None:
    llm = ScriptedLLM([error])
    use_service(api, DiagnosisService(make_agent(llm, settings)))
    response = await client.post("/api/v1/diagnosis", json=alert.model_dump(mode="json"))
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "secret" not in response.text
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


async def test_api_unexpected_failure_is_sanitized(
    api: FastAPI, client: httpx.AsyncClient, settings: Settings, alert: Alert
) -> None:
    llm = ScriptedLLM([RuntimeError("password=never-expose")])
    use_service(api, DiagnosisService(make_agent(llm, settings)))
    response = await client.post("/api/v1/diagnosis", json=alert.model_dump(mode="json"))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "never-expose" not in response.text


async def test_api_not_found_has_error_shape(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/unknown")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"
    assert response.headers["x-request-id"]


async def test_api_wrong_method_preserves_allow_header(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/diagnosis")
    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
    assert response.json()["error"]["code"] == "http_error"


async def test_health_and_readiness(client: httpx.AsyncClient) -> None:
    health = await client.get("/healthz")
    readiness = await client.get("/readyz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": "0.2.0"}
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"


async def test_missing_configuration_readiness() -> None:
    app = create_app(Settings(_env_file=None, llm_api_key=""))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        readiness = await client.get("/readyz")
        health = await client.get("/healthz")
    assert readiness.status_code == 503
    assert readiness.json()["status"] == "not_configured"
    assert health.status_code == 200


async def test_generated_request_id_is_uuid(client: httpx.AsyncClient) -> None:
    first = await client.get("/healthz")
    second = await client.get("/healthz")
    assert UUID(first.headers["x-request-id"]).version == 4
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


@pytest.mark.parametrize("candidate", ["a" * 65, "has spaces", "bad/value", ""])
async def test_invalid_request_id_is_replaced(client: httpx.AsyncClient, candidate: str) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": candidate})
    assert response.headers["x-request-id"] != candidate
    assert UUID(response.headers["x-request-id"]).version == 4


async def test_openapi_publishes_request_result_and_errors(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    endpoint = schema["paths"]["/api/v1/diagnosis"]["post"]
    assert endpoint["requestBody"]["required"] is True
    for status in ["200", "422", "500", "502", "503", "504"]:
        assert status in endpoint["responses"]
    assert "DiagnosisResult" in schema["components"]["schemas"]


async def test_concurrent_requests_do_not_share_context(
    api: FastAPI, client: httpx.AsyncClient, settings: Settings, alert: Alert
) -> None:
    agent = make_agent(ScriptedLLM([]), settings)
    observed: list[str | None] = []

    async def run(received: Alert) -> AgentRun:
        from app.core.logging import request_id_context

        observed.append(request_id_context.get())
        await asyncio.sleep(0)
        observed.append(request_id_context.get())
        return AgentRun(
            result=DiagnosisResult(
                root_cause=received.service,
                confidence=0.1,
                evidence=[],
                recommended_actions=["inspect"],
                data_gaps=["no data"],
            ),
            steps=4,
            tool_calls=0,
        )

    use_service(api, DiagnosisService(agent))
    with patch.object(agent, "run", new=AsyncMock(side_effect=run)):
        responses = await asyncio.gather(
            *[
                client.post(
                    "/api/v1/diagnosis",
                    json={**alert.model_dump(mode="json"), "service": f"service-{index}"},
                    headers={"X-Request-ID": f"request-{index}"},
                )
                for index in range(8)
            ]
        )
    assert all(response.status_code == 200 for response in responses)
    assert len({response.json()["diagnosis_id"] for response in responses}) == 8
    for index, response in enumerate(responses):
        assert response.json()["result"]["root_cause"] == f"service-{index}"
        assert response.headers["x-request-id"] == f"request-{index}"
        assert observed.count(f"request-{index}") == 2
