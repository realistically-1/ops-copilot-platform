from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from importlib.resources import files

from pydantic import TypeAdapter

from app.models.service import (
    CatalogService,
    DependencyInfo,
    EnvironmentInfo,
    RuntimeInfo,
    ServiceInfo,
    ServiceOwner,
)


class FakeCatalogClient:
    source = "fixture-catalog"
    simulated = True

    def __init__(
        self,
        records: list[CatalogService],
        before_read: Callable[[str | None], None] | None = None,
    ) -> None:
        self._records: dict[str, CatalogService] = {}
        for record in records:
            if record.name in self._records:
                raise ValueError(f"duplicate catalog service: {record.name}")
            self._records[record.name] = record.model_copy(deep=True)
        self._before_read = before_read

    @classmethod
    def from_fixture(cls) -> FakeCatalogClient:
        raw = files("app.data").joinpath("catalog.json").read_text(encoding="utf-8")
        return cls(TypeAdapter(list[CatalogService]).validate_json(raw))

    @classmethod
    def from_legacy(cls, services: list[ServiceInfo]) -> FakeCatalogClient:
        grouped: dict[str, list[ServiceInfo]] = {}
        keys: set[tuple[str, str]] = set()
        for service in services:
            key = (service.service, service.environment)
            if key in keys:
                raise ValueError(f"duplicate service catalog entry: {key}")
            keys.add(key)
            grouped.setdefault(service.service, []).append(service)
        records = []
        for name, environments in grouped.items():
            first = environments[0]
            records.append(
                CatalogService(
                    name=name,
                    owner=ServiceOwner(team=first.owner),
                    runtime=RuntimeInfo(language=first.language),
                    updated_at=datetime(2026, 10, 8, tzinfo=UTC),
                    environments=[
                        EnvironmentInfo(
                            name=item.environment,
                            current_version=item.deploy_version,
                            metadata=item.metadata,
                            dependencies=[
                                DependencyInfo(name=value) for value in item.dependencies
                            ],
                        )
                        for item in environments
                    ],
                )
            )
        return cls(records)

    def get(self, service: str) -> CatalogService | None:
        if self._before_read:
            self._before_read(service)
        record = self._records.get(service)
        return record.model_copy(deep=True) if record else None

    def list_services(self) -> list[CatalogService]:
        if self._before_read:
            self._before_read(None)
        return [self._records[name].model_copy(deep=True) for name in sorted(self._records)]
