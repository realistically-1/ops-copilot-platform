from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator

ServiceName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$"
    ),
]
EnvironmentName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=40, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$"
    ),
]


class AlertSignal(BaseModel):
    metric: Literal[
        "http_error_rate", "request_latency", "cpu_utilization", "memory_utilization", "unknown"
    ]
    comparison: Literal[">", ">=", "<", "<="] | None = None
    threshold: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    unit: Literal["percent", "milliseconds"] | None = None
    percentile: int | None = Field(default=None, ge=1, le=100)
    http_status_codes: list[int] = Field(default_factory=list)


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    service: ServiceName
    environment: EnvironmentName
    alert: str = Field(min_length=1, max_length=4000)
    timestamp: AwareDatetime

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


class ParsedAlert(BaseModel):
    service: str
    environment: str
    summary: str
    category: Literal["availability", "latency", "resource", "unknown"]
    keywords: list[str]
    signal: AlertSignal
    window_start: AwareDatetime
    window_end: AwareDatetime
