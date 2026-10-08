from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.models.alert import EnvironmentName, ServiceName

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Severity(StrEnum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1, max_length=250)
    source: Literal["alert", "search_logs", "get_service_info"]
    summary: Text
    reference: str = Field(min_length=1, max_length=1000)
    simulated: bool = False
    observed_at: AwareDatetime | None = None
    service: ServiceName | None = None
    environment: EnvironmentName | None = None
    attributes: dict[str, str] = Field(default_factory=dict, max_length=40)


class AffectedComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: ServiceName
    role: Literal["primary", "dependency"]
    impact: Text
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)


class ReportAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_count: int = Field(default=0, ge=0)
    simulated_evidence_count: int = Field(default=0, ge=0)
    independent_sources: list[str] = Field(default_factory=list)
    tool_failures: int = Field(default=0, ge=0)
    evidence_status: Literal["alert_only", "simulated_context", "observed_context"] = "alert_only"
    confidence_note: str = "Confidence is an uncalibrated model estimate."
    limitations: list[str] = Field(default_factory=list)


class ReportEvent(BaseModel):
    timestamp: AwareDatetime
    source: Literal["alert", "search_logs", "get_service_info"]
    service: str | None
    description: Text
    evidence_id: str
    simulated: bool


class DiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_cause: Text
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence: list[Evidence] = Field(max_length=30)
    recommended_actions: list[Text] = Field(min_length=1, max_length=12)
    data_gaps: list[Text] = Field(max_length=20)
    summary: Text | None = None
    severity: Severity = Severity.UNKNOWN
    affected_components: list[AffectedComponent] = Field(default_factory=list, max_length=20)
    assessment: ReportAssessment = Field(default_factory=ReportAssessment)
    timeline: list[ReportEvent] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def unique_evidence(self) -> Self:
        ids = [item.id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")
        names = [item.name for item in self.affected_components]
        if len(names) != len(set(names)):
            raise ValueError("affected component names must be unique")
        if sum(item.role == "primary" for item in self.affected_components) > 1:
            raise ValueError("only one primary component is allowed")
        for component in self.affected_components:
            if set(component.evidence_ids) - set(ids):
                raise ValueError("component references evidence missing from the report")
        evidence_by_id = {item.id: item for item in self.evidence}
        timeline_ids: set[str] = set()
        for event in self.timeline:
            source = evidence_by_id.get(event.evidence_id)
            if source is None:
                raise ValueError("timeline event references missing evidence")
            if event.evidence_id in timeline_ids:
                raise ValueError("timeline evidence IDs must be unique")
            timeline_ids.add(event.evidence_id)
            if event.timestamp != source.observed_at or event.source != source.source:
                raise ValueError("timeline event does not match its evidence")
            if event.simulated != source.simulated:
                raise ValueError("timeline cannot change evidence provenance")
        if self.summary is None:
            self.summary = self.root_cause
        return self


class DiagnosisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_cause: Text
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_ids: list[str] = Field(max_length=30)
    recommended_actions: list[Text] = Field(min_length=1, max_length=12)
    data_gaps: list[Text] = Field(max_length=20)
    summary: Text | None = None
    severity: Severity = Severity.UNKNOWN
    affected_components: list[AffectedComponent] = Field(default_factory=list, max_length=20)
