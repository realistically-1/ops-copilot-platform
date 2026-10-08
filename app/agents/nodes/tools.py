import json
import logging

from app.agents.state import DiagnosisState
from app.agents.tools.search_logs import SearchLogsTool
from app.agents.tools.service_info import GetServiceInfoTool
from app.core.config import Settings
from app.models.logs import TimeRange
from app.models.tools import SearchLogsCall, ToolCall, ToolObservation
from app.services.evidence import EvidenceConflict, merge_observation, rejected_observation

logger = logging.getLogger(__name__)


class ToolsNode:
    def __init__(
        self, search_logs: SearchLogsTool, get_service_info: GetServiceInfoTool, settings: Settings
    ) -> None:
        self._search_logs = search_logs
        self._get_service_info = get_service_info
        self._log_limit = settings.log_result_limit

    async def __call__(self, state: DiagnosisState) -> DiagnosisState:
        call = state["decision"].tool_call
        if call is None:
            raise ValueError("tools node requires a tool call")
        if isinstance(call, SearchLogsCall):
            effective = call.arguments.model_copy(
                update={"limit": min(call.arguments.limit, self._log_limit)}
            )
            key = "search_logs:" + effective.fingerprint()
        else:
            key = json.dumps(call.model_dump(mode="json"), sort_keys=True)
        rejection = self._reject_call(call, state, key)
        if rejection:
            observation = ToolObservation(
                tool=call.name,
                status="error",
                summary=rejection,
                data_gaps=[rejection],
                error_code="invalid_tool_request",
            )
        else:
            try:
                if isinstance(call, SearchLogsCall):
                    query = call.arguments.model_copy(
                        update={"limit": min(call.arguments.limit, self._log_limit)}
                    )
                    observation = await self._search_logs(query)
                else:
                    observation = self._get_service_info(call.arguments)
            except Exception:
                # CancelledError must propagate.
                logger.exception("tool.unexpected_failure", extra={"tool": call.name})
                observation = ToolObservation(
                    tool=call.name,
                    status="error",
                    summary="Tool failed unexpectedly.",
                    data_gaps=[f"{call.name} did not return usable context."],
                    error_code="tool_failed",
                )
        try:
            evidence = merge_observation(
                state["evidence"],
                observation,
                state["alert"],
                TimeRange(start=state["parsed"].window_start, end=state["parsed"].window_end),
            )
        except EvidenceConflict as exc:
            observation = rejected_observation(observation, str(exc))
            evidence = list(state["evidence"])
        return {
            "observations": [*state["observations"], observation],
            "evidence": evidence,
            "data_gaps": list(dict.fromkeys([*state["data_gaps"], *observation.data_gaps])),
            "tool_calls": state["tool_calls"] + 1,
            "executed_calls": [*state["executed_calls"], key],
            "steps": state["steps"] + 1,
        }

    @staticmethod
    def _reject_call(call: ToolCall, state: DiagnosisState, key: str) -> str | None:
        if key in state["executed_calls"]:
            return "Identical tool request already attempted; no new context was obtained."
        alert = state["alert"]
        if (
            call.arguments.service != alert.service
            or call.arguments.environment != alert.environment
        ):
            return "Tool request is outside the alert service or environment."
        if isinstance(call, SearchLogsCall):
            parsed = state["parsed"]
            window = TimeRange(start=parsed.window_start, end=parsed.window_end)
            if not window.contains_range(call.arguments.time_range):
                return "Log search is outside the investigation time window."
        return None
