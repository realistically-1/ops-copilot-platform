import asyncio
import logging
from collections.abc import Callable
from time import perf_counter
from uuid import UUID

from app.agents.graph import DiagnosisAgent
from app.api.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse, ExecutionSummary
from app.core.config import Settings
from app.core.exceptions import ApplicationError
from app.core.logging import request_id_context
from app.integrations.llm.recording import capture_model_calls
from app.models.alert import Alert
from app.models.execution import (
    DiagnosisExecution,
    ExecutionError,
    ExecutionFilter,
    ExecutionPage,
    ExecutionStep,
)
from app.services.context import DiagnosisContextBuilder
from app.stores.diagnoses import InMemoryDiagnosisStore

logger = logging.getLogger(__name__)


class DiagnosisService:
    def __init__(
        self,
        agent: DiagnosisAgent,
        store: InMemoryDiagnosisStore | None = None,
        context_builder: DiagnosisContextBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._agent = agent
        self._store = store if store is not None else InMemoryDiagnosisStore()
        self._context_builder = context_builder
        self._model = settings.llm_model if settings else "unspecified"

    def get(self, diagnosis_id: UUID) -> DiagnosisExecution:
        return self._store.get(diagnosis_id)

    def list(self, filters: ExecutionFilter) -> ExecutionPage:
        return self._store.list(filters)

    async def diagnose(
        self,
        request: DiagnosisRequest,
        on_created: Callable[[UUID], None] | None = None,
    ) -> DiagnosisResponse:
        started = perf_counter()
        alert = Alert.model_validate(request.model_dump(exclude={"options"}))
        execution = self._store.create(
            alert, self._model, request_id_context.get(), request.options
        )
        diagnosis_id = execution.diagnosis_id
        if on_created:
            on_created(diagnosis_id)
        logger.info(
            "diagnosis.started",
            extra={
                "diagnosis_id": str(diagnosis_id),
                "service": request.service,
                "environment": request.environment,
            },
        )
        self._store.start(diagnosis_id)

        def progress(step: ExecutionStep, tool_calls: int) -> None:
            self._store.record_step(diagnosis_id, step, tool_calls)

        try:
            with capture_model_calls() as model_calls:
                try:
                    if self._context_builder is not None:
                        context = self._context_builder.build(diagnosis_id, alert, request.options)
                        self._store.attach_context(diagnosis_id, context)
                        run = await self._agent.run(alert, context=context, on_progress=progress)
                    else:
                        run = await self._agent.run(alert)
                finally:
                    self._store.record_model_calls(diagnosis_id, model_calls)
            duration_ms = round((perf_counter() - started) * 1000, 2)
            saved = self._store.succeed(
                diagnosis_id, run.result, run.steps, run.tool_calls, duration_ms
            )
        except asyncio.CancelledError:
            self._store.fail(
                diagnosis_id,
                ExecutionError(
                    code="request_cancelled", message="The requesting task was cancelled."
                ),
                (perf_counter() - started) * 1000,
                cancelled=True,
            )
            raise
        except ApplicationError as exc:
            self._store.fail(
                diagnosis_id,
                ExecutionError(code=exc.code, message=exc.public_message),
                (perf_counter() - started) * 1000,
            )
            logger.warning(
                "diagnosis.failed",
                extra={"diagnosis_id": str(diagnosis_id), "error_code": exc.code},
            )
            raise
        except Exception:
            self._store.fail(
                diagnosis_id,
                ExecutionError(code="internal_error", message="An unexpected error occurred."),
                (perf_counter() - started) * 1000,
            )
            raise
        response = DiagnosisResponse(
            diagnosis_id=diagnosis_id,
            service=request.service,
            environment=request.environment,
            created_at=saved.created_at,
            result=run.result,
            execution=ExecutionSummary(
                steps=run.steps, tool_calls=run.tool_calls, duration_ms=duration_ms
            ),
        )
        logger.info(
            "diagnosis.completed",
            extra={
                "diagnosis_id": str(diagnosis_id),
                "service": request.service,
                "duration_ms": duration_ms,
                "step_count": run.steps,
            },
        )
        return response
