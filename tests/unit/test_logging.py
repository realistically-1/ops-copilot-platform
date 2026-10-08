import json
import logging
import sys

from app.core.logging import JsonFormatter, request_id_context


def test_json_formatter_includes_context_and_event_fields() -> None:
    record = logging.LogRecord("test", logging.INFO, "test.py", 1, "completed", (), None)
    record.service = "payment-service"
    record.duration_ms = 12.3
    record.status_code = 200
    token = request_id_context.set("request-123")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_context.reset(token)
    assert payload["request_id"] == "request-123"
    assert payload["service"] == "payment-service"
    assert payload["duration_ms"] == 12.3
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["timestamp"].endswith("+00:00")


def test_json_formatter_does_not_copy_arbitrary_fields() -> None:
    record = logging.LogRecord("test", logging.INFO, "test.py", 1, "completed", (), None)
    record.api_key = "private-key"
    record.prompt = "private-input"
    output = JsonFormatter().format(record)
    assert "private-key" not in output
    assert "private-input" not in output


def test_exception_logging_omits_exception_body() -> None:
    try:
        raise RuntimeError("Authorization: Bearer private-key")
    except RuntimeError:
        record = logging.LogRecord(
            "test", logging.ERROR, "test.py", 1, "failed", (), sys.exc_info()
        )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["exception_type"] == "RuntimeError"
    assert "private-key" not in json.dumps(payload)


def test_logging_outside_request_has_null_request_id() -> None:
    record = logging.LogRecord("test", logging.INFO, "test.py", 1, "startup", (), None)
    assert json.loads(JsonFormatter().format(record))["request_id"] is None


def test_log_message_supports_unicode_and_arguments() -> None:
    record = logging.LogRecord("test", logging.INFO, "test.py", 1, "完成 %s", ("检查",), None)
    output = JsonFormatter().format(record)
    assert json.loads(output)["message"] == "完成 检查"
    assert "完成" in output
