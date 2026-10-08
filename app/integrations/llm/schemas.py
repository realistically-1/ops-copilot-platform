from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CompletionUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def consistent_usage(self) -> Self:
        if self.total_tokens is not None:
            for count in (self.prompt_tokens, self.completion_tokens):
                if count is not None and count > self.total_tokens:
                    raise ValueError("token component exceeds total")
        return self


class CompletionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content: str | None
    refusal: str | None = None


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    finish_reason: str
    message: CompletionMessage


class CompletionEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = None
    model: str | None = None
    choices: list[CompletionChoice] = Field(min_length=1)
    usage: CompletionUsage | None = None

    def structured_content(self) -> str:
        choice = self.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError("model did not finish normally")
        if choice.message.refusal:
            raise ValueError("model refused the request")
        content = choice.message.content
        if content is None or not content.strip():
            raise ValueError("model content is empty")
        return content


class ModelCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    response_schema: str
    status: Literal["succeeded", "failed", "cancelled"]
    duration_ms: float = Field(ge=0)
    prompt_characters: int = Field(ge=0)
    provider_request_id: str | None = None
    usage: CompletionUsage | None = None
    error_code: str | None = None
