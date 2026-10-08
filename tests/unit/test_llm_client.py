import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, LLMError, LLMResponseError, LLMTimeoutError
from app.integrations.llm.client import ChatMessage, OpenAICompatibleLLMClient
from app.models.diagnosis import DiagnosisDraft
from app.models.tools import ReasoningDecision
from tests.helpers import draft, finish

MESSAGES = [ChatMessage(role="user", content="Investigate payment errors")]


def completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}


async def test_llm_sends_openai_compatible_request(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=completion(draft().model_dump_json()))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await OpenAICompatibleLLMClient(settings, http).complete_structured(
            MESSAGES, DiagnosisDraft
        )
        assert not http.is_closed
    request = captured[0]
    body = json.loads(request.content)
    assert request.url == "https://model.invalid/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert body["max_tokens"] == settings.llm_max_tokens
    assert "evidence_ids" in body["messages"][0]["content"]
    assert body["messages"][-1]["content"] == MESSAGES[0].content
    assert request.extensions["timeout"]["read"] == settings.llm_timeout_seconds
    assert result == draft()


async def test_llm_can_parse_reasoning_decision(settings: Settings) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json=completion(finish().model_dump_json()))
    )
    async with httpx.AsyncClient(transport=transport) as http:
        result = await OpenAICompatibleLLMClient(settings, http).complete_structured(
            MESSAGES, ReasoningDecision
        )
    assert result.action == "diagnose"
    assert result.tool_call is None


async def test_missing_key_prevents_network(settings: Settings) -> None:
    empty = settings.model_copy(update={"llm_api_key": Settings(_env_file=None).llm_api_key})
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ConfigurationError):
            await OpenAICompatibleLLMClient(empty, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )
    assert called is False


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
async def test_http_errors_are_mapped_without_retry(settings: Settings, status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": "private upstream detail"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMError) as error:
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )
    assert calls == 1
    assert "private upstream detail" not in str(error.value)
    assert error.value.status_code == 502


@pytest.mark.parametrize("factory", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout])
async def test_timeout_is_mapped(
    settings: Settings, factory: Callable[..., httpx.TimeoutException]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise factory("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMTimeoutError) as error:
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )
    assert error.value.status_code == 504


async def test_transport_error_is_mapped(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private-host.internal", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMError) as error:
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )
    assert "private-host" not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        {},
        [],
        None,
        {"choices": []},
        {"choices": {}},
        {"choices": [None]},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "content_filter", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"refusal": "denied", "content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": " "}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": ["text"]}}]},
        completion("not json"),
        completion("{}"),
        completion("```json\n{}\n```"),
    ],
)
async def test_invalid_envelope_and_content(settings: Settings, body: object) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LLMResponseError):
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )


async def test_non_json_http_body(settings: Settings) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, text="<html>upstream error</html>")
    )
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LLMResponseError):
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )


async def test_schema_violation_is_not_coerced_to_success(settings: Settings) -> None:
    body = {**draft().model_dump(), "confidence": 9}
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json=completion(json.dumps(body)))
    )
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LLMResponseError):
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )


async def test_client_cancellation_propagates(settings: Settings) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(asyncio.CancelledError):
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )


async def test_redirect_is_not_followed(settings: Settings) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://other.invalid"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMError):
            await OpenAICompatibleLLMClient(settings, http).complete_structured(
                MESSAGES, DiagnosisDraft
            )
    assert len(requests) == 1


async def test_trailing_slash_and_provider_path() -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key="key",
        llm_base_url="https://provider.invalid/compatible-mode/v1/",
    )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=completion(draft().model_dump_json()))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await OpenAICompatibleLLMClient(settings, http).complete_structured(
            MESSAGES, DiagnosisDraft
        )
    assert seen == ["https://provider.invalid/compatible-mode/v1/chat/completions"]
