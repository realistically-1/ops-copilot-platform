from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import __version__
from app.core.exceptions import CatalogError
from app.services.catalog import ServiceCatalog
from app.stores.diagnoses import InMemoryDiagnosisStore

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessCheck(BaseModel):
    name: str
    status: Literal["ready", "not_ready"]
    message: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_configured", "not_ready"]
    version: str
    environment: str
    checks: list[ReadinessCheck]


@router.get("/healthz", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/readyz", response_model=ReadinessResponse, responses={503: {"model": ReadinessResponse}}
)
def readiness(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    configured = settings.llm_configured
    checks = [
        ReadinessCheck(
            name="llm_configuration",
            status="ready" if configured else "not_ready",
            message="API key configured; provider connectivity is not probed."
            if configured
            else "API key missing.",
        )
    ]
    catalog: ServiceCatalog | None = getattr(request.app.state, "service_catalog", None)
    if catalog is None:
        checks.append(
            ReadinessCheck(
                name="catalog", status="not_ready", message="Catalog is not initialized."
            )
        )
    else:
        try:
            page = catalog.list(limit=1)
            checks.append(
                ReadinessCheck(
                    name="catalog",
                    status="ready" if page.total else "not_ready",
                    message=f"{page.total} catalog services loaded.",
                )
            )
        except CatalogError:
            checks.append(
                ReadinessCheck(
                    name="catalog", status="not_ready", message="Catalog is unavailable."
                )
            )
    store: InMemoryDiagnosisStore | None = getattr(request.app.state, "diagnosis_store", None)
    if store is None:
        checks.append(
            ReadinessCheck(
                name="diagnosis_store", status="not_ready", message="Store is not initialized."
            )
        )
    else:
        summary = store.summary()
        has_capacity = summary.active < summary.capacity
        checks.append(
            ReadinessCheck(
                name="diagnosis_store",
                status="ready" if has_capacity else "not_ready",
                message="Capacity available."
                if has_capacity
                else "All retained executions are active and capacity is exhausted.",
            )
        )
    logs_loaded = getattr(request.app.state, "log_service", None) is not None
    checks.append(
        ReadinessCheck(
            name="logs",
            status="ready" if logs_loaded else "not_ready",
            message="Log service initialized."
            if logs_loaded
            else "Log service is not initialized.",
        )
    )
    ready = all(check.status == "ready" for check in checks)
    status = "ready" if ready else "not_ready" if configured else "not_configured"
    body = ReadinessResponse(
        status=status,
        version=__version__,
        environment=settings.app_environment,
        checks=checks,
    )
    return JSONResponse(body.model_dump(), status_code=200 if ready else 503)
