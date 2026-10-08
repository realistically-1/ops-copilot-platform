from app.agents.prompts import diagnosis_messages
from app.agents.state import DiagnosisState
from app.core.exceptions import LLMResponseError
from app.integrations.llm.client import LLMClient
from app.models.diagnosis import DiagnosisDraft


class DiagnoseNode:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def __call__(self, state: DiagnosisState) -> DiagnosisState:
        draft = await self._llm.complete_structured(diagnosis_messages(state), DiagnosisDraft)
        available = {item.id for item in state["evidence"]}
        if set(draft.evidence_ids) - available:
            raise LLMResponseError("Diagnosis referenced evidence that was never observed.")
        return {"draft": draft, "steps": state["steps"] + 1}
