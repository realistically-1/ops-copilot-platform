from collections import deque
from collections.abc import Sequence

from pydantic import BaseModel

from app.agents.graph import DiagnosisAgent
from app.agents.tools.search_logs import SearchLogsTool
from app.agents.tools.service_info import GetServiceInfoTool
from app.core.config import Settings
from app.integrations.llm.client import ChatMessage
from app.integrations.logs.fake import FakeLogClient
from app.models.diagnosis import DiagnosisDraft
from app.models.logs import LogQuery
from app.models.tools import (
    ReasoningDecision,
    SearchLogsCall,
    ServiceInfoArguments,
    ServiceInfoCall,
)
from app.services.catalog import ServiceCatalog


class ScriptedLLM:
    def __init__(self, responses: Sequence[BaseModel | Exception]) -> None:
        self.responses = deque(responses)
        self.calls: list[tuple[list[ChatMessage], type[BaseModel]]] = []

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[ChatMessage], response_model: type[T]
    ) -> T:
        self.calls.append((list(messages), response_model))
        if not self.responses:
            raise AssertionError("Unexpected model call")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response.model_dump())


def finish() -> ReasoningDecision:
    return ReasoningDecision(action="diagnose", rationale="Sufficient context.", tool_call=None)


def search(query: LogQuery) -> ReasoningDecision:
    return ReasoningDecision(
        action="tool",
        rationale="Look for errors around the alert.",
        tool_call=SearchLogsCall(name="search_logs", arguments=query),
    )


def service_info(service: str = "payment-service", environment: str = "prod") -> ReasoningDecision:
    return ReasoningDecision(
        action="tool",
        rationale="Check deployment and dependencies.",
        tool_call=ServiceInfoCall(
            name="get_service_info",
            arguments=ServiceInfoArguments(service=service, environment=environment),
        ),
    )


def draft(evidence_ids: list[str] | None = None) -> DiagnosisDraft:
    return DiagnosisDraft(
        root_cause="Database pool exhaustion is a likely cause; requires live verification.",
        confidence=0.7,
        evidence_ids=evidence_ids if evidence_ids is not None else ["alert:input"],
        recommended_actions=["Inspect database pool metrics and recent configuration changes."],
        data_gaps=["Database metrics were not available."],
    )


def make_agent(
    llm: ScriptedLLM,
    settings: Settings,
    logs: FakeLogClient | None = None,
    catalog: ServiceCatalog | None = None,
) -> DiagnosisAgent:
    return DiagnosisAgent(
        llm,
        SearchLogsTool(logs or FakeLogClient.from_fixture()),
        GetServiceInfoTool(catalog or ServiceCatalog.from_fixture()),
        settings,
    )
