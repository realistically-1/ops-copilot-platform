from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.alert import Alert
from app.models.context import DiagnosisOptions
from app.models.diagnosis import DiagnosisResult
from app.models.execution import ExecutionPage


class DiagnosisRequest(Alert):
    options: DiagnosisOptions = Field(default_factory=DiagnosisOptions)


class ExecutionSummary(BaseModel):
    steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    duration_ms: float = Field(ge=0)


class DiagnosisResponse(BaseModel):
    diagnosis_id: UUID
    service: str
    environment: str
    created_at: datetime
    result: DiagnosisResult
    execution: ExecutionSummary


class DiagnosisListItem(BaseModel):
    diagnosis_id: UUID
    service: str
    environment: str
    alert: str
    status: str
    created_at: datetime
    finished_at: datetime | None
    summary: str | None
    severity: str | None
    confidence: float | None
    error_code: str | None
    steps: int
    tool_calls: int
    duration_ms: float


class DiagnosisListResponse(BaseModel):
    items: list[DiagnosisListItem]
    total: int
    offset: int
    limit: int
    has_more: bool

    @classmethod
    def from_page(cls, page: ExecutionPage) -> DiagnosisListResponse:
        return cls(
            items=[
                DiagnosisListItem(
                    diagnosis_id=item.diagnosis_id,
                    service=item.alert.service,
                    environment=item.alert.environment,
                    alert=item.alert.alert,
                    status=item.status,
                    created_at=item.created_at,
                    finished_at=item.finished_at,
                    summary=item.result.summary if item.result else None,
                    severity=item.result.severity if item.result else None,
                    confidence=item.result.confidence if item.result else None,
                    error_code=item.error.code if item.error else None,
                    steps=item.metadata.steps,
                    tool_calls=item.metadata.tool_calls,
                    duration_ms=item.metadata.duration_ms,
                )
                for item in page.items
            ],
            total=page.total,
            offset=page.offset,
            limit=page.limit,
            has_more=page.has_more,
        )
