from app.agents.prompts import reasoning_messages
from app.agents.state import DiagnosisState
from app.core.config import Settings
from app.integrations.llm.client import LLMClient
from app.models.tools import ReasoningDecision


class ReasonNode:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._max_calls = settings.agent_max_tool_calls

    async def __call__(self, state: DiagnosisState) -> DiagnosisState:
        if state["tool_calls"] >= self._max_calls:
            return {
                "decision": ReasoningDecision(
                    action="diagnose", rationale="Tool budget exhausted.", tool_call=None
                ),
                "data_gaps": [
                    *state["data_gaps"],
                    "Investigation reached the configured tool budget.",
                ],
                "steps": state["steps"] + 1,
            }
        decision = await self._llm.complete_structured(
            reasoning_messages(state, self._max_calls), ReasoningDecision
        )
        return {"decision": decision, "steps": state["steps"] + 1}


def after_reason(state: DiagnosisState) -> str:
    return "tools" if state["decision"].action == "tool" else "diagnose"
