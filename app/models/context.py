from typing import Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.alert import Alert
from app.models.logs import LogLevel, LogQuery, TimeRange
from app.models.service import ServiceInfo


class DiagnosisOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookback_minutes: int | None = Field(default=None, ge=1, le=120)
    lookahead_minutes: int | None = Field(default=None, ge=0, le=60)
    keywords: list[str] = Field(default_factory=list, max_length=8)
    log_levels: list[LogLevel] = Field(default_factory=list, max_length=5)

    @field_validator("keywords")
    @classmethod
    def valid_keywords(cls, values: list[str]) -> list[str]:
        return LogQuery.clean_keywords(values)


class DiagnosisContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    diagnosis_id: UUID
    alert: Alert
    collected_at: AwareDatetime
    window: TimeRange
    service_info: ServiceInfo | None
    initial_keywords: list[str]
    initial_levels: list[LogLevel]
    available_components: list[str]
    data_gaps: list[str]

    @model_validator(mode="after")
    def consistent_scope(self) -> Self:
        if not self.window.start <= self.alert.timestamp <= self.window.end:
            raise ValueError("context window must include the alert timestamp")
        if self.service_info is not None:
            if self.service_info.service != self.alert.service:
                raise ValueError("context service differs from alert")
            if self.service_info.environment != self.alert.environment:
                raise ValueError("context environment differs from alert")
        if self.alert.service not in self.available_components:
            raise ValueError("context must include the alert service")
        if len(self.available_components) != len(set(self.available_components)):
            raise ValueError("context components must be unique")
        allowed = {self.alert.service}
        if self.service_info:
            allowed.update(self.service_info.dependencies)
        if set(self.available_components) != allowed:
            raise ValueError("context components differ from the catalog snapshot")
        return self

    def prompt_context(self) -> dict[str, object]:
        service = self.service_info
        return {
            "diagnosis_id": str(self.diagnosis_id),
            "collected_at": self.collected_at.isoformat(),
            "window": self.window.model_dump(mode="json"),
            "service_summary": {
                "owner": service.owner,
                "tier": service.tier,
                "version": service.deploy_version,
                "dependencies": service.dependencies,
                "source": service.source,
                "simulated": service.simulated,
            }
            if service
            else None,
            "initial_keywords": self.initial_keywords,
            "initial_levels": self.initial_levels,
            "available_components": self.available_components,
            "data_gaps": self.data_gaps,
        }
