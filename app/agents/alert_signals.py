import re
from typing import Literal

from app.models.alert import AlertSignal

_THRESHOLD = re.compile(r"(?P<operator>>=|<=|>|<)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|ms|s)\b")
_PERCENT = re.compile(r"(?P<operator>>=|<=|>|<)\s*(?P<value>\d+(?:\.\d+)?)\s*%")
_STATUS = re.compile(r"\b[45]\d{2}\b")
_PERCENTILE = re.compile(r"\bp(\d{1,3})\b", re.IGNORECASE)


def extract_signal(text: str) -> AlertSignal:
    normalized = text.casefold()
    metric = _metric(normalized)
    signal = AlertSignal(metric=metric)
    if "http" in normalized or "status" in normalized:
        signal.http_status_codes = sorted({int(match) for match in _STATUS.findall(text)})
    if metric == "request_latency":
        percentile = _PERCENTILE.search(text)
        if percentile and 1 <= int(percentile.group(1)) <= 100:
            signal.percentile = int(percentile.group(1))
    match = _PERCENT.search(text) or _THRESHOLD.search(text)
    if match is None:
        return signal
    unit = match.groupdict().get("unit", "%")
    value = float(match.group("value"))
    if unit == "%" and metric in {"http_error_rate", "cpu_utilization", "memory_utilization"}:
        if value > 100:
            return signal
        signal.unit = "percent"
    elif unit in {"ms", "s"} and metric == "request_latency":
        signal.unit = "milliseconds"
        value *= 1000 if unit == "s" else 1
    else:
        return signal
    if value == float("inf"):
        return signal.model_copy(update={"unit": None})
    signal.threshold = value
    match match.group("operator"):
        case ">":
            signal.comparison = ">"
        case ">=":
            signal.comparison = ">="
        case "<":
            signal.comparison = "<"
        case "<=":
            signal.comparison = "<="
    return signal


def _metric(
    text: str,
) -> Literal[
    "http_error_rate", "request_latency", "cpu_utilization", "memory_utilization", "unknown"
]:
    if any(token in text for token in ("5xx", "error rate", "错误率")):
        return "http_error_rate"
    if any(token in text for token in ("latency", "p99", "p95", "延迟")):
        return "request_latency"
    if "cpu" in text:
        return "cpu_utilization"
    if "memory" in text or "内存" in text:
        return "memory_utilization"
    return "unknown"
