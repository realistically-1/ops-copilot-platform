from typing import TypedDict

from app.models.alert import Alert, ParsedAlert
from app.models.context import DiagnosisContext
from app.models.diagnosis import DiagnosisDraft, DiagnosisResult, Evidence
from app.models.tools import ReasoningDecision, ToolObservation


class DiagnosisState(TypedDict, total=False):
    context: DiagnosisContext
    alert: Alert
    parsed: ParsedAlert
    decision: ReasoningDecision
    observations: list[ToolObservation]
    evidence: list[Evidence]
    data_gaps: list[str]
    tool_calls: int
    executed_calls: list[str]
    steps: int
    draft: DiagnosisDraft
    result: DiagnosisResult
