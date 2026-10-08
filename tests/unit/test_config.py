from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://compatible.invalid/v1/")
    monkeypatch.setenv("LLM_API_KEY", "private-key")
    monkeypatch.setenv("LLM_MODEL", "compatible-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "https://compatible.invalid/v1"
    assert settings.llm_model == "compatible-model"
    assert settings.llm_timeout_seconds == 12.5
    assert settings.llm_configured
    assert "private-key" not in repr(settings)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/socket",
        "not-a-url",
        "https://",
        "ftp://host",
        "https://user:pass@host/v1",
        "https://host/v1?api_key=secret",
        "https://host/v1#fragment",
    ],
)
def test_invalid_base_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_base_url=url)


@pytest.mark.parametrize(
    "key,value",
    [
        ("llm_timeout_seconds", 0),
        ("llm_timeout_seconds", 181),
        ("agent_max_steps", 0),
        ("agent_max_steps", 101),
        ("agent_max_tool_calls", -1),
        ("agent_max_tool_calls", 13),
        ("log_result_limit", 0),
        ("log_result_limit", 101),
        ("log_window_before_minutes", 0),
        ("log_window_before_minutes", 121),
        ("log_window_after_minutes", -1),
        ("log_window_after_minutes", 61),
        ("llm_max_tokens", 100),
        ("llm_model", ""),
        ("log_level", "TRACE"),
    ],
)
def test_settings_validate_limits(key: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{key: value})


@pytest.mark.parametrize("key", ["", " ", "\t"])
def test_empty_key_is_not_ready(key: str) -> None:
    assert Settings(_env_file=None, llm_api_key=key).llm_configured is False


def test_local_http_endpoint_is_supported() -> None:
    settings = Settings(_env_file=None, llm_base_url="http://127.0.0.1:8080/v1")
    assert settings.llm_base_url == "http://127.0.0.1:8080/v1"


def test_dotenv_loads_and_environment_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("LLM_MODEL=file-model\nAGENT_MAX_STEPS=16\nUNRELATED=ignored\n")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    settings = Settings(_env_file=dotenv)
    assert settings.llm_model == "environment-model"
    assert settings.agent_max_steps == 16


def test_cached_settings_can_be_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    try:
        monkeypatch.setenv("LLM_MODEL", "first")
        first = get_settings()
        monkeypatch.setenv("LLM_MODEL", "second")
        assert get_settings() is first
        assert get_settings().llm_model == "first"
        get_settings.cache_clear()
        assert get_settings().llm_model == "second"
    finally:
        get_settings.cache_clear()
