from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.alert import EnvironmentName, ServiceName


class ServiceOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    team: str = Field(min_length=1, max_length=100)
    contact: str | None = Field(default=None, max_length=200)
    escalation: str | None = Field(default=None, max_length=200)


class RuntimeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    language: str = Field(min_length=1, max_length=80)
    version: str | None = Field(default=None, max_length=80)
    framework: str | None = Field(default=None, max_length=100)

    @property
    def display_name(self) -> str:
        return " ".join(value for value in (self.language, self.version) if value)


class DependencyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: ServiceName
    kind: Literal["service", "database", "external"] = "service"
    critical: bool = True
    purpose: str = Field(default="", max_length=300)


class EnvironmentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: EnvironmentName
    current_version: str = Field(min_length=1, max_length=100)
    region: str | None = Field(default=None, max_length=100)
    replicas: int | None = Field(default=None, ge=0, le=10000)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=30)
    dependencies: list[DependencyInfo] = Field(default_factory=list, max_length=50)

    @field_validator("dependencies")
    @classmethod
    def unique_dependencies(cls, values: list[DependencyInfo]) -> list[DependencyInfo]:
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("dependency names must be unique")
        return values

    @field_validator("metadata")
    @classmethod
    def bounded_metadata(cls, values: dict[str, str]) -> dict[str, str]:
        for key, value in values.items():
            if not key.strip() or len(key) > 80 or len(value) > 1000:
                raise ValueError("metadata key or value exceeds allowed size")
        return values


class CatalogService(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: ServiceName
    description: str = Field(default="", max_length=1000)
    owner: ServiceOwner
    runtime: RuntimeInfo
    tier: Literal["critical", "standard", "internal"] = "standard"
    tags: list[str] = Field(default_factory=list, max_length=20)
    environments: list[EnvironmentInfo] = Field(min_length=1, max_length=20)
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def valid_environment_names(self) -> Self:
        names = [item.name for item in self.environments]
        if len(names) != len(set(names)):
            raise ValueError("environment names must be unique")
        for environment in self.environments:
            if any(item.name == self.name for item in environment.dependencies):
                raise ValueError("service cannot directly depend on itself")
        return self

    def environment(self, name: str) -> EnvironmentInfo | None:
        return next((item for item in self.environments if item.name == name), None)


class ServiceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    service: ServiceName
    environment: EnvironmentName
    owner: str = Field(min_length=1)
    language: str = Field(min_length=1)
    dependencies: list[str]
    deploy_version: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    tier: Literal["critical", "standard", "internal"] = "standard"
    description: str = ""
    contact: str | None = None
    dependency_details: list[DependencyInfo] = Field(default_factory=list)
    catalog_updated_at: AwareDatetime | None = None
    source: str = "fixture-catalog"
    simulated: bool = True


class ServiceCatalogEntry(BaseModel):
    service: CatalogService
    source: str
    simulated: bool
    warnings: list[str] = Field(default_factory=list)


class ServiceSummary(BaseModel):
    name: str
    owner: str
    language: str
    tier: str
    environments: list[str]
    tags: list[str]


class ServicePage(BaseModel):
    items: list[ServiceSummary]
    total: int
    offset: int
    limit: int


class DependencyResolution(BaseModel):
    name: str
    kind: str
    critical: bool
    registered: bool
    available_in_environment: bool
    owner: str | None = None
    current_version: str | None = None
