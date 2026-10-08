from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.diagnose import DiagnoseNode
from app.agents.nodes.parse_alert import ParseAlertNode
from app.agents.nodes.reason import ReasonNode, after_reason
from app.agents.nodes.report import generate_report
from app.agents.nodes.tools import ToolsNode
from app.agents.state import DiagnosisState
from app.agents.tools.search_logs import SearchLogsTool
from app.agents.tools.service_info import GetServiceInfoTool
from app.core.config import Settings
from app.core.exceptions import AgentLimitError
from app.integrations.llm.client import LLMClient
from app.models.alert import Alert
from app.models.context import DiagnosisContext
from app.models.diagnosis import DiagnosisResult
from app.models.execution import ExecutionStep


@dataclass(frozen=True)
class AgentRun:
    result: DiagnosisResult
    steps: int
    tool_calls: int


class DiagnosisAgent:
    def __init__(
        self,
        llm: LLMClient,
        search_logs: SearchLogsTool,
        get_service_info: GetServiceInfoTool,
        settings: Settings,
    ) -> None:
        graph = StateGraph(DiagnosisState)
        graph.add_node("parse_alert", ParseAlertNode(settings))
        graph.add_node("reason", ReasonNode(llm, settings))
        graph.add_node("tools", ToolsNode(search_logs, get_service_info, settings))
        graph.add_node("diagnose", DiagnoseNode(llm))
        graph.add_node("generate_report", generate_report)
        graph.add_edge(START, "parse_alert")
        graph.add_edge("parse_alert", "reason")
        graph.add_conditional_edges(
            "reason", after_reason, {"tools": "tools", "diagnose": "diagnose"}
        )
        graph.add_edge("tools", "reason")
        graph.add_edge("diagnose", "generate_report")
        graph.add_edge("generate_report", END)
        self._graph = graph.compile()
        self._max_steps = settings.agent_max_steps

    async def run(
        self,
        alert: Alert,
        *,
        context: DiagnosisContext | None = None,
        on_progress: Callable[[ExecutionStep, int], None] | None = None,
    ) -> AgentRun:
        initial: DiagnosisState = {
            "alert": alert,
            "observations": [],
            "evidence": [],
            "data_gaps": [],
            "tool_calls": 0,
            "executed_calls": [],
            "steps": 0,
        }
        if context is not None:
            initial["context"] = context
            initial["data_gaps"] = list(context.data_gaps)
        state = initial.copy()
        started = perf_counter()
        try:
            async for updates in self._graph.astream(
                initial, {"recursion_limit": self._max_steps}, stream_mode="updates"
            ):
                for node, update in updates.items():
                    state.update(update)
                    if on_progress is not None:
                        observation = state["observations"][-1] if node == "tools" else None
                        step = ExecutionStep.model_validate(
                            {
                                "sequence": state["steps"],
                                "node": node,
                                "completed_at": datetime.now(UTC),
                                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                                "tool": observation.tool if observation else None,
                                "outcome": observation.status if observation else None,
                                "evidence_count": len(state["evidence"]),
                            }
                        )
                        on_progress(step, state["tool_calls"])
        except GraphRecursionError as exc:
            raise AgentLimitError() from exc
        return AgentRun(
            result=state["result"], steps=state["steps"], tool_calls=state["tool_calls"]
        )
