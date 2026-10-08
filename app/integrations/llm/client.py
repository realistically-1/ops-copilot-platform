import asyncio
import json
import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, LLMError, LLMResponseError, LLMTimeoutError
from app.integrations.llm.recording import record_model_call
from app.integrations.llm.schemas import CompletionEnvelope, ModelCallRecord

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    async def complete_structured[T: BaseModel](
        self, messages: Sequence[ChatMessage], response_model: type[T]
    ) -> T: ...


class OpenAICompatibleLLMClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http_client

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[ChatMessage], response_model: type[T]
    ) -> T:
        if not self._settings.llm_configured:
            raise ConfigurationError()
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        schema_message = ChatMessage(
            role="system",
            content=(
                "Return exactly one JSON object matching this JSON Schema. "
                "No markdown fences or additional prose. All required fields must be present.\n"
                + schema
            ),
        )
        prompt_characters = sum(len(item.content) for item in [schema_message, *messages])
        if prompt_characters > self._settings.llm_max_prompt_chars:
            raise LLMError("Model input exceeds the configured character budget.")
        payload = {
            "model": self._settings.llm_model,
            "messages": [item.model_dump() for item in [schema_message, *messages]],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self._settings.llm_max_tokens,
            "stream": False,
        }
        started = perf_counter()
        status: Literal["succeeded", "failed", "cancelled"] = "failed"
        error_code: str | None = None
        envelope: CompletionEnvelope | None = None
        try:
            response = await self._request(payload)
            envelope = self._parse_envelope(response)
            try:
                result = response_model.model_validate_json(envelope.structured_content())
            except (ValueError, ValidationError) as exc:
                raise LLMResponseError() from exc
            status = "succeeded"
            return result
        except asyncio.CancelledError:
            status = "cancelled"
            error_code = "request_cancelled"
            raise
        except LLMError as exc:
            error_code = exc.code
            raise
        finally:
            duration = round((perf_counter() - started) * 1000, 2)
            record_model_call(
                ModelCallRecord(
                    model=envelope.model or self._settings.llm_model
                    if envelope
                    else self._settings.llm_model,
                    response_schema=response_model.__name__,
                    status=status,
                    duration_ms=duration,
                    prompt_characters=prompt_characters,
                    provider_request_id=envelope.id if envelope else None,
                    usage=envelope.usage if envelope else None,
                    error_code=error_code,
                )
            )
            logger.info(
                "llm.completed",
                extra={
                    "duration_ms": duration,
                    "model": self._settings.llm_model,
                    "error_code": error_code,
                },
            )

    async def _request(self, payload: dict[str, object]) -> httpx.Response:
        try:
            response = await self._http.post(
                f"{self._settings.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._settings.llm_api_key.get_secret_value()}"
                },
                json=payload,
                timeout=self._settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            return response
        except httpx.TimeoutException as exc:
            logger.warning("llm.timeout", extra={"error_code": "llm_timeout"})
            raise LLMTimeoutError() from exc
        except httpx.HTTPStatusError as exc:
            logger.warning("llm.http_error", extra={"status_code": exc.response.status_code})
            raise LLMError() from exc
        except httpx.RequestError as exc:
            logger.warning("llm.transport_error", extra={"error_code": "llm_error"})
            raise LLMError() from exc

    def _parse_envelope(self, response: httpx.Response) -> CompletionEnvelope:
        if len(response.content) > self._settings.llm_max_response_bytes:
            raise LLMResponseError("Provider response exceeds the configured size limit.")
        try:
            return CompletionEnvelope.model_validate_json(response.content)
        except ValidationError as exc:
            raise LLMResponseError() from exc
