from app.agents.state import DiagnosisState
from app.services.reports import DiagnosisReportBuilder


def generate_report(state: DiagnosisState) -> DiagnosisState:
    result = DiagnosisReportBuilder().build(
        alert=state["alert"],
        draft=state["draft"],
        evidence=state["evidence"],
        observations=state["observations"],
        known_gaps=state["data_gaps"],
        context=state.get("context"),
    )
    return {"result": result, "steps": state["steps"] + 1}
