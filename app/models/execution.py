from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field, model_validator

from app.integrations.llm.schemas import ModelCallRecord
from app.models.alert import Alert
from app.models.context import DiagnosisContext, DiagnosisOptions
from app.models.diagnosis import DiagnosisResult


class ExecutionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class ExecutionError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    stage: str | None = None


class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int = Field(ge=1)
    node: Literal["parse_alert", "reason", "tools", "diagnose", "generate_report"]
    completed_at: AwareDatetime
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    tool: str | None = None
    outcome: Literal["ok", "empty", "error"] | None = None
    evidence_count: int = Field(default=0, ge=0)


class ExecutionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    model_calls: list[ModelCallRecord] = Field(default_factory=list, max_length=100)
    request_id: str | None = None
    steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    last_node: str | None = None
    timeline: list[ExecutionStep] = Field(default_factory=list, max_length=100)
    fixture_context: bool = True

    @model_validator(mode="after")
    def consistent_progress(self) -> Self:
        if self.tool_calls > self.steps:
            raise ValueError("tool count cannot exceed completed steps")
        if len(self.timeline) > self.steps:
            raise ValueError("timeline cannot exceed completed steps")
        previous_elapsed = 0.0
        for index, step in enumerate(self.timeline, start=1):
            if step.sequence != index:
                raise ValueError("timeline sequence must be contiguous")
            if step.elapsed_ms < previous_elapsed:
                raise ValueError("timeline elapsed duration cannot decrease")
            previous_elapsed = step.elapsed_ms
        if self.timeline and self.last_node != self.timeline[-1].node:
            raise ValueError("last node differs from the timeline")
        return self

    @computed_field
    @property
    def usage(self) -> ModelUsageSummary:
        return ModelUsageSummary.from_calls(self.model_calls)


class DiagnosisExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    diagnosis_id: UUID
    created_at: AwareDatetime
    updated_at: AwareDatetime
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    status: ExecutionStatus = ExecutionStatus.CREATED
    alert: Alert
    options: DiagnosisOptions = Field(default_factory=DiagnosisOptions)
    context: DiagnosisContext | None = None
    metadata: ExecutionMetadata
    result: DiagnosisResult | None = None
    error: ExecutionError | None = None

    @model_validator(mode="after")
    def check_lifecycle(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes creation")
        if self.started_at and self.started_at < self.created_at:
            raise ValueError("started_at precedes creation")
        if self.finished_at and self.finished_at < (self.started_at or self.created_at):
            raise ValueError("finished_at precedes execution")
        if self.status == ExecutionStatus.CREATED and self.started_at is not None:
            raise ValueError("created execution cannot have a start time")
        if self.status == ExecutionStatus.RUNNING and self.started_at is None:
            raise ValueError("running execution needs a start time")
        if self.status.terminal != (self.finished_at is not None):
            raise ValueError("only terminal executions have a finish time")
        if (self.status == ExecutionStatus.SUCCEEDED) != (self.result is not None):
            raise ValueError("only succeeded executions have a result")
        if (self.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}) != (
            self.error is not None
        ):
            raise ValueError("failed and cancelled executions require an error")
        return self


class ExecutionFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str | None = None
    environment: str | None = None
    status: ExecutionStatus | None = None
    created_after: AwareDatetime | None = None
    created_before: AwareDatetime | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def ordered_dates(self) -> Self:
        if self.created_after and self.created_before and self.created_after >= self.created_before:
            raise ValueError("created_after must precede created_before")
        return self

    def matches(self, execution: DiagnosisExecution) -> bool:
        if self.service is not None and execution.alert.service != self.service:
            return False
        if self.environment is not None and execution.alert.environment != self.environment:
            return False
        if self.status is not None and execution.status != self.status:
            return False
        if self.created_after and execution.created_at < self.created_after:
            return False
        return not (self.created_before and execution.created_at >= self.created_before)


class ExecutionPage(BaseModel):
    items: list[DiagnosisExecution]
    total: int
    offset: int
    limit: int
    has_more: bool


class StoreSummary(BaseModel):
    capacity: int
    total: int
    active: int
    terminal: int
    retention_seconds: int
    oldest_created_at: datetime | None


class ModelUsageSummary(BaseModel):
    calls: int
    succeeded: int
    failed: int
    cancelled: int
    calls_with_usage: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    duration_ms: float

    @classmethod
    def from_calls(cls, calls: list[ModelCallRecord]) -> ModelUsageSummary:
        usage = [call.usage for call in calls if call.usage is not None]
        prompt = [item.prompt_tokens for item in usage if item.prompt_tokens is not None]
        completion = [
            item.completion_tokens for item in usage if item.completion_tokens is not None
        ]
        total = [item.total_tokens for item in usage if item.total_tokens is not None]
        return cls(
            calls=len(calls),
            succeeded=sum(call.status == "succeeded" for call in calls),
            failed=sum(call.status == "failed" for call in calls),
            cancelled=sum(call.status == "cancelled" for call in calls),
            calls_with_usage=len(usage),
            prompt_tokens=sum(prompt) if prompt and len(prompt) == len(calls) else None,
            completion_tokens=sum(completion)
            if completion and len(completion) == len(calls)
            else None,
            total_tokens=sum(total) if total and len(total) == len(calls) else None,
            duration_ms=round(sum(call.duration_ms for call in calls), 2),
        )
