from pydantic import BaseModel, ConfigDict, Field

from app.models.alert import EnvironmentName


class CatalogFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    environment: EnvironmentName | None = None
    owner: str | None = Field(default=None, min_length=1, max_length=100)
    keyword: str | None = Field(default=None, min_length=1, max_length=100)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
