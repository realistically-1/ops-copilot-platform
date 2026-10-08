from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.alert import EnvironmentName, ServiceName


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return list(type(self)).index(self)


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start: AwareDatetime
    end: AwareDatetime

    @field_validator("start", "end")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.start >= self.end:
            raise ValueError("start must be earlier than end")
        if self.seconds > 10800:
            raise ValueError("log search window cannot exceed three hours")
        return self

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp < self.end

    def contains_range(self, other: TimeRange) -> bool:
        return self.start <= other.start and other.end <= self.end


class LogQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    service: ServiceName
    environment: EnvironmentName
    start: AwareDatetime
    end: AwareDatetime
    keywords: list[str] = Field(default_factory=list, max_length=8)
    levels: list[LogLevel] = Field(default_factory=list, max_length=5)
    match_mode: Literal["any", "all"] = "any"
    order: Literal["asc", "desc"] = "asc"
    limit: int = Field(default=30, ge=1, le=100)

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value or len(value) > 100:
                raise ValueError("keywords must contain 1 to 100 characters")
            if value.casefold() not in {item.casefold() for item in cleaned}:
                cleaned.append(value)
        return cleaned

    @field_validator("levels")
    @classmethod
    def unique_levels(cls, levels: list[LogLevel]) -> list[LogLevel]:
        return sorted(set(levels), key=lambda level: level.rank)

    @model_validator(mode="after")
    def check_window(self) -> Self:
        TimeRange(start=self.start, end=self.end)
        return self

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(start=self.start, end=self.end)

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "service": self.service,
                "environment": self.environment,
                "start": self.start.astimezone(UTC).isoformat(),
                "end": self.end.astimezone(UTC).isoformat(),
                "keywords": sorted(keyword.casefold() for keyword in self.keywords),
                "levels": sorted(self.levels),
                "match_mode": self.match_mode,
                "order": self.order,
                "limit": self.limit,
            },
            sort_keys=True,
        )


class LogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1, max_length=150)
    service: ServiceName
    environment: EnvironmentName
    timestamp: AwareDatetime
    level: LogLevel
    message: str = Field(min_length=1, max_length=1500)
    attributes: dict[str, str] = Field(default_factory=dict, max_length=40)

    @property
    def searchable_text(self) -> str:
        return " ".join([self.level, self.message, *self.attributes.values()]).casefold()

    def matches(self, query: LogQuery) -> bool:
        if self.service != query.service or self.environment != query.environment:
            return False
        if not query.time_range.contains(self.timestamp):
            return False
        if query.levels and self.level not in query.levels:
            return False
        if not query.keywords:
            return True
        checks = [keyword.casefold() in self.searchable_text for keyword in query.keywords]
        return all(checks) if query.match_mode == "all" else any(checks)


class LogStatistics(BaseModel):
    level_counts: dict[LogLevel, int] = Field(default_factory=dict)
    first_seen: AwareDatetime | None = None
    last_seen: AwareDatetime | None = None
    error_count: int = Field(default=0, ge=0)
    returned_count: int = Field(default=0, ge=0)
    distinct_instances: list[str] = Field(default_factory=list)


class LogPattern(BaseModel):
    message: str
    level: LogLevel
    count: int = Field(ge=1)
    sample_ids: list[str]
    first_seen: AwareDatetime
    last_seen: AwareDatetime


class LogFacet(BaseModel):
    value: str
    count: int = Field(ge=1)
    sample_ids: list[str] = Field(max_length=3)


class LogSearchResult(BaseModel):
    entries: list[LogEntry]
    total: int = Field(ge=0)
    truncated: bool
    backend: str
    simulated: bool
    statistics: LogStatistics | None = None
    facets: dict[str, list[LogFacet]] = Field(default_factory=dict)
    patterns: list[LogPattern] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    queried_range: TimeRange | None = None

    @model_validator(mode="after")
    def consistent_total(self) -> Self:
        if self.total < len(self.entries):
            raise ValueError("total cannot be smaller than returned entries")
        if self.truncated != (self.total > len(self.entries)):
            raise ValueError("truncation flag disagrees with total")
        return self
