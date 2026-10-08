from datetime import timedelta
from typing import Literal

from app.agents.alert_signals import extract_signal
from app.agents.state import DiagnosisState
from app.core.config import Settings
from app.models.alert import ParsedAlert
from app.models.diagnosis import Evidence


class ParseAlertNode:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def __call__(self, state: DiagnosisState) -> DiagnosisState:
        alert = state["alert"]
        text = alert.alert.casefold()
        category: Literal["availability", "latency", "resource", "unknown"] = "unknown"
        keywords: list[str] = []
        if any(token in text for token in ("5xx", "500", "503", "error rate", "错误率")):
            category = "availability"
            keywords = ["error", "500", "503", "timeout"]
        elif any(token in text for token in ("latency", "p99", "timeout", "延迟", "超时")):
            category = "latency"
            keywords = ["timeout", "slow", "latency"]
        elif any(token in text for token in ("memory", "cpu", "oom", "内存")):
            category = "resource"
            keywords = ["memory", "cpu", "oom"]
        parsed = ParsedAlert(
            service=alert.service,
            environment=alert.environment,
            summary=alert.alert,
            category=category,
            keywords=keywords,
            signal=extract_signal(alert.alert),
            window_start=alert.timestamp
            - timedelta(minutes=self._settings.log_window_before_minutes),
            window_end=alert.timestamp + timedelta(minutes=self._settings.log_window_after_minutes),
        )
        context = state.get("context")
        if context is not None:
            parsed.window_start = context.window.start
            parsed.window_end = context.window.end
            if context.initial_keywords:
                parsed.keywords = context.initial_keywords
        return {
            "parsed": parsed,
            "evidence": [
                Evidence(
                    id="alert:input",
                    source="alert",
                    summary=alert.alert[:2000],
                    observed_at=alert.timestamp,
                    service=alert.service,
                    environment=alert.environment,
                    reference=f"alert/{alert.service}/{alert.environment}/{alert.timestamp.isoformat()}",
                )
            ],
            "steps": state["steps"] + 1,
        }
