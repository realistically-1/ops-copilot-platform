import pytest

from app.agents.alert_signals import extract_signal
from app.agents.nodes.parse_alert import ParseAlertNode
from app.agents.state import DiagnosisState
from app.core.config import Settings
from app.models.alert import Alert


@pytest.mark.parametrize(
    "text,metric,threshold,operator",
    [
        ("HTTP 5xx error rate > 15%", "http_error_rate", 15, ">"),
        ("HTTP error rate >= 0.5%", "http_error_rate", 0.5, ">="),
        ("CPU usage > 90%", "cpu_utilization", 90, ">"),
        ("memory available < 5%", "memory_utilization", 5, "<"),
        ("内存剩余 <= 10%", "memory_utilization", 10, "<="),
        ("错误率 > 1.5%", "http_error_rate", 1.5, ">"),
    ],
)
def test_percentage_signal(text: str, metric: str, threshold: float, operator: str) -> None:
    signal = extract_signal(text)
    assert signal.metric == metric
    assert signal.threshold == threshold
    assert signal.comparison == operator
    assert signal.unit == "percent"


@pytest.mark.parametrize(
    "text,threshold,percentile",
    [
        ("P99 latency > 200ms", 200, 99),
        ("p95 latency >= 1.5s", 1500, 95),
        ("request latency > 3s", 3000, None),
        ("P100 latency > 1ms", 1, 100),
        ("P999 latency > 1ms", 1, None),
    ],
)
def test_latency_signal(text: str, threshold: float, percentile: int | None) -> None:
    signal = extract_signal(text)
    assert signal.metric == "request_latency"
    assert signal.threshold == threshold
    assert signal.percentile == percentile
    assert signal.unit == "milliseconds"


def test_http_status_codes_are_unique_and_sorted() -> None:
    signal = extract_signal("HTTP status 503 and 500, repeated 503, successes 200")
    assert signal.http_status_codes == [500, 503]


def test_non_http_numbers_are_not_status_codes() -> None:
    signal = extract_signal("memory usage > 500ms")
    assert signal.http_status_codes == []
    assert signal.threshold is None
    assert signal.unit is None


@pytest.mark.parametrize(
    "text",
    [
        "CPU > 101%",
        "memory > 102%",
        "HTTP error rate > 1000%",
        "latency > 50%",
        "CPU > 3s",
        "unknown metric > 50%",
        "p99 latency > 20",
        "CPU > -5%",
    ],
)
def test_unsupported_threshold_is_not_invented(text: str) -> None:
    signal = extract_signal(text)
    assert signal.threshold is None
    assert signal.comparison is None
    assert signal.unit is None


def test_unknown_alert_stays_unknown() -> None:
    signal = extract_signal("Business KPI anomaly detected")
    assert signal.metric == "unknown"
    assert signal.threshold is None
    assert signal.http_status_codes == []


def test_percentage_threshold_at_zero() -> None:
    signal = extract_signal("HTTP error rate > 0%")
    assert signal.threshold == 0
    assert signal.unit == "percent"


def test_percentage_threshold_at_one_hundred() -> None:
    signal = extract_signal("CPU >= 100%")
    assert signal.threshold == 100
    assert signal.comparison == ">="


def test_threshold_overflow_is_not_infinity() -> None:
    signal = extract_signal("p99 latency > " + "9" * 400 + "s")
    assert signal.threshold is None
    assert signal.unit is None


def test_parser_keeps_original_alert_alongside_signal(alert: Alert, settings: Settings) -> None:
    state: DiagnosisState = {"alert": alert, "steps": 0}
    parsed = ParseAlertNode(settings)(state)["parsed"]
    assert parsed.summary == alert.alert
    assert parsed.signal.metric == "http_error_rate"
    assert parsed.signal.threshold == 15
    assert parsed.signal.unit == "percent"
