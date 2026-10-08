from datetime import timedelta

import pytest

from app.core.exceptions import ExecutionConflictError, NotFoundError, StoreCapacityError
from app.models.alert import Alert
from app.models.diagnosis import DiagnosisResult
from app.models.execution import ExecutionError, ExecutionFilter, ExecutionStatus, ExecutionStep
from app.stores.diagnoses import InMemoryDiagnosisStore


def test_store_transitions_copies_and_retention(alert: Alert) -> None:
    now = alert.timestamp
    store = InMemoryDiagnosisStore(capacity=2, retention_seconds=60, clock=lambda: now)
    created = store.create(alert, "model", "request-1")
    assert created.status == ExecutionStatus.CREATED
    store.start(created.diagnosis_id)
    store.record_step(
        created.diagnosis_id,
        ExecutionStep(
            sequence=1,
            node="parse_alert",
            completed_at=now,
            elapsed_ms=2,
            evidence_count=1,
        ),
        0,
    )
    result = DiagnosisResult(
        root_cause="unknown",
        confidence=0.1,
        evidence=[],
        recommended_actions=["inspect"],
        data_gaps=["no live data"],
    )
    finished = store.succeed(created.diagnosis_id, result, 4, 0, 5)
    assert finished.status == ExecutionStatus.SUCCEEDED
    assert finished.metadata.last_node == "parse_alert"
    assert finished.started_at == now
    finished.result.recommended_actions.clear()
    assert store.get(created.diagnosis_id).result.recommended_actions == ["inspect"]
    with pytest.raises(ExecutionConflictError):
        store.start(created.diagnosis_id)
    now += timedelta(seconds=60)
    with pytest.raises(NotFoundError):
        store.get(created.diagnosis_id)


def test_store_never_evicts_active_execution(alert: Alert) -> None:
    now = alert.timestamp
    store = InMemoryDiagnosisStore(capacity=1, retention_seconds=60, clock=lambda: now)
    first = store.create(alert, "model", None)
    store.start(first.diagnosis_id)
    now += timedelta(hours=1)
    with pytest.raises(StoreCapacityError):
        store.create(alert, "model", None)
    assert store.get(first.diagnosis_id).status == ExecutionStatus.RUNNING
    store.fail(first.diagnosis_id, ExecutionError(code="failed", message="Failed"), 10)
    second = store.create(alert, "model", None)
    assert second.diagnosis_id != first.diagnosis_id
    with pytest.raises(NotFoundError):
        store.get(first.diagnosis_id)


def test_store_filter_paging_and_cancelled_status(alert: Alert) -> None:
    store = InMemoryDiagnosisStore()
    first = store.create(alert, "model", None)
    store.fail(
        first.diagnosis_id, ExecutionError(code="cancelled", message="Cancelled"), 1, cancelled=True
    )
    store.create(alert.model_copy(update={"environment": "staging"}), "model", None)
    page = store.list(ExecutionFilter(status=ExecutionStatus.CANCELLED, limit=1))
    assert page.total == 1
    assert page.items[0].diagnosis_id == first.diagnosis_id
    assert not page.has_more
    assert store.list(ExecutionFilter(environment="staging")).total == 1
    assert store.list(ExecutionFilter(limit=1)).has_more
    assert store.list(ExecutionFilter(offset=9)).items == []
    summary = store.summary()
    assert (summary.total, summary.active, summary.terminal) == (2, 1, 1)


def test_store_rejects_invalid_progress_atomically(alert: Alert) -> None:
    store = InMemoryDiagnosisStore()
    record = store.create(alert, "model", None)
    store.start(record.diagnosis_id)
    with pytest.raises(ExecutionConflictError):
        store.record_step(
            record.diagnosis_id,
            ExecutionStep(
                sequence=2,
                node="reason",
                completed_at=record.created_at,
                elapsed_ms=5,
            ),
            0,
        )
    assert store.get(record.diagnosis_id).metadata.timeline == []
