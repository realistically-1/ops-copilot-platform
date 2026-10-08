from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from app.models.alert import EnvironmentName, ServiceName
from app.models.diagnosis import Evidence, Text
from app.models.logs import LogQuery


class SearchLogsCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["search_logs"]
    arguments: LogQuery


class ServiceInfoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: ServiceName
    environment: EnvironmentName


class ServiceInfoCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["get_service_info"]
    arguments: ServiceInfoArguments


ToolCall = Annotated[SearchLogsCall | ServiceInfoCall, Field(discriminator="name")]


class ReasoningDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tool", "diagnose"]
    rationale: Text
    tool_call: ToolCall | None

    @model_validator(mode="after")
    def consistent_action(self) -> Self:
        if (self.action == "tool") != (self.tool_call is not None):
            raise ValueError("tool action requires a tool call; diagnose requires null")
        return self


class ToolObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["search_logs", "get_service_info"]
    status: Literal["ok", "empty", "error"]
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    error_code: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)
