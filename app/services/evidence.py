from app.models.alert import Alert
from app.models.diagnosis import Evidence
from app.models.logs import TimeRange
from app.models.tools import ToolObservation


class EvidenceConflict(ValueError):
    pass


def merge_observation(
    current: list[Evidence],
    observation: ToolObservation,
    alert: Alert,
    window: TimeRange,
) -> list[Evidence]:
    if observation.status != "ok" and observation.evidence:
        raise EvidenceConflict("Unsuccessful observation contains evidence.")
    indexed = {item.id: item for item in current}
    if len(indexed) != len(current):
        raise EvidenceConflict("Existing evidence IDs are not unique.")
    for item in observation.evidence:
        _validate_scope(item, observation.tool, alert, window)
        prior = indexed.get(item.id)
        if prior is not None and prior != item:
            raise EvidenceConflict("An evidence ID was reused for different content.")
        indexed[item.id] = item
    return list(indexed.values())


def _validate_scope(evidence: Evidence, source: str, alert: Alert, window: TimeRange) -> None:
    if evidence.source != source:
        raise EvidenceConflict("Evidence source differs from the observing tool.")
    if evidence.service is not None and evidence.service != alert.service:
        raise EvidenceConflict("Evidence belongs to another service.")
    if evidence.environment is not None and evidence.environment != alert.environment:
        raise EvidenceConflict("Evidence belongs to another environment.")
    if evidence.source == "search_logs":
        if evidence.observed_at is not None and not window.contains(evidence.observed_at):
            raise EvidenceConflict("Log evidence is outside the investigation window.")
    if evidence.id == "alert:input":
        raise EvidenceConflict("Tool evidence cannot replace the original alert.")


def rejected_observation(observation: ToolObservation, reason: str) -> ToolObservation:
    return ToolObservation(
        tool=observation.tool,
        status="error",
        summary="Tool returned inconsistent evidence.",
        error_code="invalid_tool_evidence",
        data_gaps=[*observation.data_gaps, reason],
    )
