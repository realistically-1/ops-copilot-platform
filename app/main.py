from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app import __version__
from app.agents.graph import DiagnosisAgent
from app.agents.tools.search_logs import SearchLogsTool
from app.agents.tools.service_info import GetServiceInfoTool
from app.api.errors import install_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routers import diagnosis, health, services
from app.api.routers import logs as log_routes
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.integrations.llm.client import OpenAICompatibleLLMClient
from app.integrations.logs.fake import FakeLogClient
from app.services.catalog import ServiceCatalog
from app.services.context import DiagnosisContextBuilder
from app.services.diagnosis import DiagnosisService
from app.services.logs import LogSearchService
from app.stores.diagnoses import InMemoryDiagnosisStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        async with httpx.AsyncClient(follow_redirects=False) as http_client:
            llm = OpenAICompatibleLLMClient(settings, http_client)
            logs = FakeLogClient.from_fixture()
            catalog = ServiceCatalog.from_fixture()
            log_service = LogSearchService(logs, settings.log_result_limit)
            store = InMemoryDiagnosisStore(
                settings.diagnosis_store_capacity, settings.diagnosis_retention_seconds
            )
            builder = DiagnosisContextBuilder(catalog, settings)
            agent = DiagnosisAgent(
                llm, SearchLogsTool(log_service), GetServiceInfoTool(catalog), settings
            )
            application.state.log_service = log_service
            application.state.diagnosis_store = store
            application.state.service_catalog = catalog
            application.state.diagnosis_service = DiagnosisService(agent, store, builder, settings)
            yield

    application = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    application.state.settings = settings
    application.add_middleware(RequestContextMiddleware)
    install_exception_handlers(application)
    application.include_router(diagnosis.router)
    application.include_router(health.router)
    application.include_router(services.router)
    application.include_router(log_routes.router)
    return application


app = create_app()
