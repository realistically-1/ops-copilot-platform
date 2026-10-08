from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="ops-copilot-platform", min_length=1, max_length=100)
    app_environment: Literal["local", "test", "staging", "prod"] = "local"
    diagnosis_store_capacity: int = Field(default=500, ge=1, le=10000)
    diagnosis_retention_seconds: int = Field(default=3600, ge=60, le=86400)
    llm_max_prompt_chars: int = Field(default=200000, ge=1000, le=1000000)
    llm_max_response_bytes: int = Field(default=2000000, ge=1000, le=10000000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = Field(default="gpt-4o-mini", min_length=1)
    llm_timeout_seconds: float = Field(default=30, gt=0, le=180)
    llm_max_tokens: int = Field(default=2500, ge=256, le=16000)
    agent_max_steps: int = Field(default=18, ge=1, le=100)
    agent_max_tool_calls: int = Field(default=4, ge=0, le=12)
    log_window_before_minutes: int = Field(default=15, ge=1, le=120)
    log_window_after_minutes: int = Field(default=5, ge=0, le=60)
    log_result_limit: int = Field(default=30, ge=1, le=100)

    @field_validator("llm_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("LLM_BASE_URL must be an HTTP(S) URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("LLM_BASE_URL must not contain credentials, query, or fragment")
        try:
            _ = parts.port
        except ValueError as exc:
            raise ValueError("LLM_BASE_URL contains an invalid port") from exc
        return value

    @field_validator("llm_model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 200 or any(ord(char) < 32 for char in value):
            raise ValueError("LLM_MODEL must be a non-empty printable model identifier")
        return value

    @field_validator("llm_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if any(ord(char) < 32 or ord(char) == 127 for char in key):
            raise ValueError("LLM_API_KEY must not contain control characters")
        return SecretStr(key)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
