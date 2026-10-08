from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.integrations.llm.schemas import ModelCallRecord

_call_records: ContextVar[list[ModelCallRecord] | None] = ContextVar(
    "model_call_records", default=None
)


@contextmanager
def capture_model_calls() -> Iterator[list[ModelCallRecord]]:
    records: list[ModelCallRecord] = []
    token = _call_records.set(records)
    try:
        yield records
    finally:
        _call_records.reset(token)


def record_model_call(record: ModelCallRecord) -> None:
    records = _call_records.get()
    if records is not None:
        records.append(record)
