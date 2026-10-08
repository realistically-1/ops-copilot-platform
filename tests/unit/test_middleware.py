import asyncio
from typing import Any

import pytest
from starlette.types import Message, Receive, Scope, Send

from app.api.middleware import RequestContextMiddleware
from app.core.logging import request_id_context


def http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": "GET",
        "path": "/healthz",
        "raw_path": b"/healthz",
        "query_string": b"",
        "headers": headers or [],
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1234),
    }


async def receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


async def test_middleware_passes_non_http_scope_through() -> None:
    received: list[Scope] = []
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        received.append(scope)
        await send({"type": "lifespan.startup.complete"})

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {"type": "lifespan"}
    await RequestContextMiddleware(app)(scope, receive, send)
    assert received == [scope]
    assert sent == [{"type": "lifespan.startup.complete"}]
    assert "state" not in scope


async def test_middleware_resets_context_after_response() -> None:
    sent: list[Message] = []
    seen: list[str | None] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(request_id_context.get())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message: Message) -> None:
        sent.append(message)

    token = request_id_context.set("outer")
    try:
        await RequestContextMiddleware(app)(
            http_scope([(b"x-request-id", b"inner")]), receive, send
        )
        assert request_id_context.get() == "outer"
    finally:
        request_id_context.reset(token)
    assert seen == ["inner"]
    assert (b"x-request-id", b"inner") in sent[0]["headers"]


async def test_middleware_replaces_duplicate_response_id() -> None:
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-request-id", b"old"), (b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestContextMiddleware(app)(http_scope([(b"x-request-id", b"new")]), receive, send)
    headers = sent[0]["headers"]
    assert [value for key, value in headers if key == b"x-request-id"] == [b"new"]
    assert (b"content-type", b"text/plain") in headers


async def test_middleware_does_not_send_second_response_after_failure() -> None:
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("stream broke")

    async def send(message: Message) -> None:
        sent.append(message)

    with pytest.raises(RuntimeError, match="stream broke"):
        await RequestContextMiddleware(app)(http_scope(), receive, send)
    assert len(sent) == 1
    assert request_id_context.get() is None


async def test_middleware_does_not_convert_cancellation_to_500() -> None:
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise asyncio.CancelledError()

    async def send(message: Message) -> None:
        sent.append(message)

    with pytest.raises(asyncio.CancelledError):
        await RequestContextMiddleware(app)(http_scope(), receive, send)
    assert sent == []
    assert request_id_context.get() is None


async def test_unexpected_exception_is_json_error() -> None:
    import json

    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise ValueError("internal details")

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestContextMiddleware(app)(http_scope([(b"x-request-id", b"error-id")]), receive, send)
    assert sent[0]["status"] == 500
    body: dict[str, Any] = json.loads(sent[1]["body"])
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["request_id"] == "error-id"
    assert "internal details" not in str(body)
