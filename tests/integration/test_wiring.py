import json
from collections import deque
from unittest.mock import patch

import httpx
import pytest

from app.api.schemas.diagnosis import DiagnosisResponse
from app.core.config import Settings
from app.main import create_app
from app.models.alert import Alert
from app.models.logs import LogQuery
from tests.helpers import draft, finish, search, service_info
from tests.unit.test_llm_client import completion

pytestmark = pytest.mark.integration


async def test_application_lifespan_wires_real_client_and_closes_it(
    settings: Settings, alert: Alert, query: LogQuery
) -> None:
    responses = deque(
        [
            service_info(),
            search(query),
            finish(),
            draft(["log:p003", "service:payment-service:prod:deploy_version"]),
        ]
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses.popleft()
        return httpx.Response(200, json=completion(response.model_dump_json()))

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    application = create_app(settings)
    api_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    )
    with (
        patch("app.main.httpx.AsyncClient", return_value=upstream),
        patch("app.main.configure_logging"),
    ):
        async with application.router.lifespan_context(application):
            assert not upstream.is_closed
            async with api_client:
                response = await api_client.post(
                    "/api/v1/diagnosis", json=alert.model_dump(mode="json")
                )
    assert upstream.is_closed
    assert response.status_code == 200
    result = DiagnosisResponse.model_validate(response.json())
    assert result.execution.steps == 8
    assert result.execution.tool_calls == 2
    assert result.result.evidence[0].source == "search_logs"
    assert result.result.evidence[1].source == "get_service_info"
    assert len(requests) == 4
    assert not responses
    last_request = json.loads(requests[-1].content)
    context = json.loads(last_request["messages"][-1]["content"])
    assert any(item["id"] == "log:p003" for item in context["evidence"])
    assert any("fixture" in item for item in context["known_data_gaps"])


async def test_unconfigured_application_starts_and_returns_503(alert: Alert) -> None:
    settings = Settings(_env_file=None, llm_api_key="")
    application = create_app(settings)
    with patch("app.main.configure_logging"):
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                health = await client.get("/healthz")
                response = await client.post(
                    "/api/v1/diagnosis", json=alert.model_dump(mode="json")
                )
    assert health.status_code == 200
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "configuration_error"


async def test_invalid_upstream_structure_reaches_api_error(
    settings: Settings, alert: Alert
) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=completion("{}")))
    upstream = httpx.AsyncClient(transport=transport)
    app = create_app(settings)
    api_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    with (
        patch("app.main.httpx.AsyncClient", return_value=upstream),
        patch("app.main.configure_logging"),
    ):
        async with app.router.lifespan_context(app):
            async with api_client:
                response = await api_client.post(
                    "/api/v1/diagnosis", json=alert.model_dump(mode="json")
                )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"
    assert upstream.is_closed


async def test_upstream_timeout_reaches_api_error(settings: Settings, alert: Alert) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private diagnostic", request=request)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(settings)
    api_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    with (
        patch("app.main.httpx.AsyncClient", return_value=upstream),
        patch("app.main.configure_logging"),
    ):
        async with app.router.lifespan_context(app):
            async with api_client:
                response = await api_client.post(
                    "/api/v1/diagnosis", json=alert.model_dump(mode="json")
                )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "llm_timeout"
    assert "private diagnostic" not in response.text


async def test_lifespan_releases_client_when_request_block_raises(settings: Settings) -> None:
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    app = create_app(settings)
    with (
        patch("app.main.httpx.AsyncClient", return_value=upstream),
        patch("app.main.configure_logging"),
    ):
        with pytest.raises(RuntimeError, match="shutdown"):
            async with app.router.lifespan_context(app):
                raise RuntimeError("shutdown")
    assert upstream.is_closed
