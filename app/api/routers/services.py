from typing import Annotated

from fastapi import APIRouter, Query

from app.api.dependencies import ServiceCatalogDependency
from app.api.schemas.catalog import CatalogFilter
from app.api.schemas.errors import ErrorResponse
from app.models.alert import EnvironmentName, ServiceName
from app.models.service import DependencyResolution, ServiceCatalogEntry, ServicePage

router = APIRouter(prefix="/api/v1/services", tags=["services"])


@router.get("", response_model=ServicePage)
def list_services(
    catalog: ServiceCatalogDependency,
    filters: Annotated[CatalogFilter, Query()],
) -> ServicePage:
    return catalog.list(**filters.model_dump())


@router.get(
    "/{service_name}", response_model=ServiceCatalogEntry, responses={404: {"model": ErrorResponse}}
)
def get_service(
    service_name: ServiceName,
    catalog: ServiceCatalogDependency,
    environment: EnvironmentName | None = None,
) -> ServiceCatalogEntry:
    return catalog.describe(service_name, environment)


@router.get(
    "/{service_name}/dependencies",
    response_model=list[DependencyResolution],
    responses={404: {"model": ErrorResponse}},
)
def get_dependencies(
    service_name: ServiceName,
    catalog: ServiceCatalogDependency,
    environment: EnvironmentName = "prod",
) -> list[DependencyResolution]:
    return catalog.dependencies(service_name, environment)
