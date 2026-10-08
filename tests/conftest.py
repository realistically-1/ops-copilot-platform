import socket
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from app.core.config import Settings
from app.models.alert import Alert
from app.models.logs import LogQuery


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not open external network connections")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    yield


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_api_key="test-key")


@pytest.fixture
def alert() -> Alert:
    return Alert(
        service="payment-service",
        environment="prod",
        alert="HTTP 5xx error rate > 15%",
        timestamp=datetime(2026, 10, 8, 4, 5, tzinfo=UTC),
    )


@pytest.fixture
def query() -> LogQuery:
    return LogQuery(
        service="payment-service",
        environment="prod",
        start=datetime(2026, 10, 8, 3, 50, tzinfo=UTC),
        end=datetime(2026, 10, 8, 4, 10, tzinfo=UTC),
        keywords=["error", "timeout"],
        limit=30,
    )
