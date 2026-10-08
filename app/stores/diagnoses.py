from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from threading import RLock
from uuid import UUID, uuid4

from app.core.exceptions import ExecutionConflictError, NotFoundError, StoreCapacityError
from app.integrations.llm.schemas import ModelCallRecord
from app.models.alert import Alert
from app.models.context import DiagnosisContext, DiagnosisOptions
from app.models.diagnosis import DiagnosisResult
from app.models.execution import (
    DiagnosisExecution,
    ExecutionError,
    ExecutionFilter,
    ExecutionMetadata,
    ExecutionPage,
    ExecutionStatus,
    ExecutionStep,
    StoreSummary,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class InMemoryDiagnosisStore:
    def __init__(
        self,
        capacity: int = 500,
        retention_seconds: int = 3600,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if capacity < 1 or retention_seconds < 1:
            raise ValueError("store capacity and retention must be positive")
        self._records: dict[UUID, DiagnosisExecution] = {}
        self._capacity = capacity
        self._retention = retention_seconds
        self._clock = clock
        self._lock = RLock()

    def create(
        self,
        alert: Alert,
        model: str,
        request_id: str | None,
        options: DiagnosisOptions | None = None,
    ) -> DiagnosisExecution:
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._make_room()
            record = DiagnosisExecution(
                diagnosis_id=uuid4(),
                created_at=now,
                updated_at=now,
                alert=alert.model_copy(deep=True),
                options=options.model_copy(deep=True) if options else DiagnosisOptions(),
                metadata=ExecutionMetadata(model=model, request_id=request_id),
            )
            self._records[record.diagnosis_id] = record
            return record.model_copy(deep=True)

    def get(self, diagnosis_id: UUID) -> DiagnosisExecution:
        with self._lock:
            self._expire(self._clock())
            return self._require(diagnosis_id).model_copy(deep=True)

    def start(self, diagnosis_id: UUID) -> DiagnosisExecution:
        with self._lock:
            record = self._require_status(diagnosis_id, ExecutionStatus.CREATED)
            now = max(self._clock(), record.updated_at)
            return self._save(
                record, status=ExecutionStatus.RUNNING, started_at=now, updated_at=now
            )

    def attach_context(self, diagnosis_id: UUID, context: DiagnosisContext) -> None:
        with self._lock:
            record = self._require_status(diagnosis_id, ExecutionStatus.RUNNING)
            if context.diagnosis_id != diagnosis_id or context.alert != record.alert:
                raise ExecutionConflictError("Context does not belong to this execution.")
            self._save(record, context=context.model_copy(deep=True))

    def record_step(self, diagnosis_id: UUID, step: ExecutionStep, tool_calls: int) -> None:
        with self._lock:
            record = self._require_status(diagnosis_id, ExecutionStatus.RUNNING)
            metadata = record.metadata.model_copy(deep=True)
            if step.sequence != metadata.steps + 1:
                raise ExecutionConflictError("Execution step sequence is not contiguous.")
            if tool_calls < metadata.tool_calls:
                raise ExecutionConflictError("Tool count cannot decrease.")
            metadata.timeline.append(step.model_copy(deep=True))
            metadata.steps = step.sequence
            metadata.tool_calls = tool_calls
            metadata.last_node = step.node
            metadata.duration_ms = max(metadata.duration_ms, step.elapsed_ms)
            self._save(record, metadata=metadata, updated_at=max(self._clock(), record.updated_at))

    def succeed(
        self,
        diagnosis_id: UUID,
        result: DiagnosisResult,
        steps: int,
        tool_calls: int,
        duration_ms: float,
    ) -> DiagnosisExecution:
        with self._lock:
            record = self._require_status(diagnosis_id, ExecutionStatus.RUNNING)
            metadata = self._finish_metadata(record, steps, tool_calls, duration_ms)
            now = max(self._clock(), record.updated_at)
            return self._save(
                record,
                status=ExecutionStatus.SUCCEEDED,
                result=result.model_copy(deep=True),
                metadata=metadata,
                finished_at=now,
                updated_at=now,
            )

    def fail(
        self,
        diagnosis_id: UUID,
        error: ExecutionError,
        duration_ms: float,
        *,
        cancelled: bool = False,
    ) -> DiagnosisExecution:
        with self._lock:
            record = self._require(diagnosis_id)
            if record.status.terminal:
                raise ExecutionConflictError("Execution is already terminal.")
            metadata = self._finish_metadata(
                record, record.metadata.steps, record.metadata.tool_calls, duration_ms
            )
            now = max(self._clock(), record.updated_at)
            return self._save(
                record,
                status=ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.FAILED,
                error=error.model_copy(deep=True),
                metadata=metadata,
                finished_at=now,
                updated_at=now,
            )

    def record_model_calls(self, diagnosis_id: UUID, calls: list[ModelCallRecord]) -> None:
        with self._lock:
            record = self._require(diagnosis_id)
            metadata = record.metadata.model_copy(deep=True)
            metadata.model_calls = [call.model_copy(deep=True) for call in calls]
            self._save(record, metadata=metadata)

    def list(self, filters: ExecutionFilter) -> ExecutionPage:
        with self._lock:
            self._expire(self._clock())
            matches = [record for record in self._records.values() if filters.matches(record)]
            matches.sort(
                key=lambda record: (record.created_at, str(record.diagnosis_id)), reverse=True
            )
            selected = matches[filters.offset : filters.offset + filters.limit]
            return ExecutionPage(
                items=[record.model_copy(deep=True) for record in selected],
                total=len(matches),
                offset=filters.offset,
                limit=filters.limit,
                has_more=filters.offset + len(selected) < len(matches),
            )

    def summary(self) -> StoreSummary:
        with self._lock:
            self._expire(self._clock())
            records = list(self._records.values())
            active = sum(not record.status.terminal for record in records)
            return StoreSummary(
                capacity=self._capacity,
                total=len(records),
                active=active,
                terminal=len(records) - active,
                retention_seconds=self._retention,
                oldest_created_at=min((record.created_at for record in records), default=None),
            )

    def _require(self, diagnosis_id: UUID) -> DiagnosisExecution:
        try:
            return self._records[diagnosis_id]
        except KeyError as exc:
            raise NotFoundError("Diagnosis is absent or has expired.") from exc

    def _require_status(self, diagnosis_id: UUID, status: ExecutionStatus) -> DiagnosisExecution:
        record = self._require(diagnosis_id)
        if record.status != status:
            raise ExecutionConflictError(f"Expected {status}, found {record.status}.")
        return record

    def _save(self, record: DiagnosisExecution, **changes: object) -> DiagnosisExecution:
        updated = DiagnosisExecution.model_validate(
            {**record.model_dump(exclude_computed_fields=True), **changes}
        )
        self._records[record.diagnosis_id] = updated
        return updated.model_copy(deep=True)

    @staticmethod
    def _finish_metadata(
        record: DiagnosisExecution, steps: int, tool_calls: int, duration_ms: float
    ) -> ExecutionMetadata:
        metadata = record.metadata.model_dump(exclude_computed_fields=True)
        if steps < record.metadata.steps or tool_calls < record.metadata.tool_calls:
            raise ExecutionConflictError("Execution counters cannot decrease.")
        metadata.update(steps=steps, tool_calls=tool_calls, duration_ms=duration_ms)
        return ExecutionMetadata.model_validate(metadata)

    def _expire(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._retention)
        expired = [
            key
            for key, record in self._records.items()
            if record.finished_at is not None and record.finished_at <= cutoff
        ]
        for key in expired:
            del self._records[key]

    def _make_room(self) -> None:
        if len(self._records) < self._capacity:
            return
        terminal = [record for record in self._records.values() if record.status.terminal]
        if not terminal:
            raise StoreCapacityError()
        oldest = min(terminal, key=lambda record: (record.finished_at, str(record.diagnosis_id)))
        del self._records[oldest.diagnosis_id]
