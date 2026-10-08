from typing import Annotated

from fastapi import Depends, Request

from app.services.catalog import ServiceCatalog
from app.services.diagnosis import DiagnosisService


def get_diagnosis_service(request: Request) -> DiagnosisService:
    return request.app.state.diagnosis_service


DiagnosisServiceDependency = Annotated[DiagnosisService, Depends(get_diagnosis_service)]


def get_service_catalog(request: Request) -> ServiceCatalog:
    return request.app.state.service_catalog


ServiceCatalogDependency = Annotated[ServiceCatalog, Depends(get_service_catalog)]
