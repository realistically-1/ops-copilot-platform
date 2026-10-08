from __future__ import annotations

from app.core.exceptions import NotFoundError
from app.integrations.catalog.client import CatalogClient
from app.integrations.catalog.fake import FakeCatalogClient
from app.models.service import (
    CatalogService,
    DependencyResolution,
    ServiceCatalogEntry,
    ServiceInfo,
    ServicePage,
    ServiceSummary,
)


class ServiceCatalog:
    def __init__(self, services: list[ServiceInfo] | CatalogClient) -> None:
        # Older callers still supply flat environment records.
        self._client = (
            FakeCatalogClient.from_legacy(services) if isinstance(services, list) else services
        )

    @classmethod
    def from_fixture(cls) -> ServiceCatalog:
        return cls(FakeCatalogClient.from_fixture())

    def get(self, service: str, environment: str) -> ServiceInfo | None:
        record = self._client.get(service)
        if record is None:
            return None
        target = record.environment(environment)
        if target is None:
            return None
        metadata = dict(target.metadata)
        if target.region is not None:
            metadata["region"] = target.region
        if target.replicas is not None:
            metadata["replicas"] = str(target.replicas)
        return ServiceInfo(
            service=record.name,
            environment=target.name,
            owner=record.owner.team,
            contact=record.owner.contact,
            language=record.runtime.display_name,
            dependencies=[item.name for item in target.dependencies],
            dependency_details=target.dependencies,
            deploy_version=target.current_version,
            metadata=metadata,
            description=record.description,
            tier=record.tier,
            catalog_updated_at=record.updated_at,
            source=self._client.source,
            simulated=self._client.simulated,
        )

    def describe(self, service: str, environment: str | None = None) -> ServiceCatalogEntry:
        record = self._client.get(service)
        if record is None:
            raise NotFoundError("Service is not registered.")
        if environment is not None:
            target = record.environment(environment)
            if target is None:
                raise NotFoundError("Service environment is not registered.")
            record = record.model_copy(update={"environments": [target]}, deep=True)
        warnings = self._warnings(record)
        return ServiceCatalogEntry(
            service=record,
            source=self._client.source,
            simulated=self._client.simulated,
            warnings=warnings,
        )

    def list(
        self,
        *,
        environment: str | None = None,
        owner: str | None = None,
        keyword: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> ServicePage:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("invalid catalog page")
        records = self._client.list_services()
        selected: list[CatalogService] = []
        for record in records:
            if environment is not None and record.environment(environment) is None:
                continue
            if owner is not None and record.owner.team != owner:
                continue
            searchable = " ".join([record.name, record.description, *record.tags]).casefold()
            if keyword and keyword.casefold() not in searchable:
                continue
            selected.append(record)
        selected.sort(key=lambda record: record.name)
        return ServicePage(
            items=[
                ServiceSummary(
                    name=record.name,
                    owner=record.owner.team,
                    language=record.runtime.display_name,
                    tier=record.tier,
                    environments=sorted(item.name for item in record.environments),
                    tags=record.tags,
                )
                for record in selected[offset : offset + limit]
            ],
            total=len(selected),
            offset=offset,
            limit=limit,
        )

    def dependencies(self, service: str, environment: str) -> list[DependencyResolution]:
        info = self.get(service, environment)
        if info is None:
            raise NotFoundError("Service environment is not registered.")
        resolved = []
        for dependency in info.dependency_details:
            record = self._client.get(dependency.name) if dependency.kind == "service" else None
            target = record.environment(environment) if record else None
            resolved.append(
                DependencyResolution(
                    name=dependency.name,
                    kind=dependency.kind,
                    critical=dependency.critical,
                    registered=record is not None,
                    available_in_environment=target is not None,
                    owner=record.owner.team if record else None,
                    current_version=target.current_version if target else None,
                )
            )
        return resolved

    def _warnings(self, record: CatalogService) -> list[str]:
        warnings = []
        if self._client.simulated:
            warnings.append("Catalog data is a static fixture snapshot.")
        if record.owner.contact is None:
            warnings.append("Owner contact is not recorded.")
        for environment in record.environments:
            if environment.replicas == 0:
                warnings.append(f"{environment.name} has zero replicas in the catalog snapshot.")
            if environment.region is None:
                warnings.append(f"{environment.name} has no recorded region.")
        return warnings
