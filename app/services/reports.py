from collections.abc import Iterable

from app.core.exceptions import LLMResponseError
from app.models.alert import Alert
from app.models.context import DiagnosisContext
from app.models.diagnosis import (
    AffectedComponent,
    DiagnosisDraft,
    DiagnosisResult,
    Evidence,
    ReportAssessment,
    ReportEvent,
)
from app.models.tools import ToolObservation


class DiagnosisReportBuilder:
    def build(
        self,
        alert: Alert,
        draft: DiagnosisDraft,
        evidence: list[Evidence],
        observations: list[ToolObservation],
        known_gaps: list[str],
        context: DiagnosisContext | None = None,
    ) -> DiagnosisResult:
        available = self._index_evidence(evidence)
        cited = self._resolve_citations(draft, available)
        components = self._components(alert, draft, cited, context)
        assessment = self._assess(cited, observations)
        gaps = self._data_gaps(draft, known_gaps, cited, observations, context)
        actions = self._unique_text(draft.recommended_actions)
        summary = draft.summary or draft.root_cause
        return DiagnosisResult(
            root_cause=draft.root_cause,
            confidence=draft.confidence,
            summary=summary,
            severity=draft.severity,
            evidence=cited,
            recommended_actions=actions,
            data_gaps=gaps,
            affected_components=components,
            assessment=assessment,
            timeline=self._timeline(cited),
        )

    @staticmethod
    def _index_evidence(evidence: list[Evidence]) -> dict[str, Evidence]:
        indexed: dict[str, Evidence] = {}
        for item in evidence:
            prior = indexed.get(item.id)
            if prior is not None and prior != item:
                raise LLMResponseError("Evidence ID refers to conflicting observations.")
            indexed[item.id] = item
        return indexed

    @staticmethod
    def _resolve_citations(draft: DiagnosisDraft, available: dict[str, Evidence]) -> list[Evidence]:
        citations = list(dict.fromkeys(draft.evidence_ids))
        if set(citations) - available.keys():
            raise LLMResponseError("Report references unavailable evidence.")
        return [available[key] for key in citations]

    @staticmethod
    def _components(
        alert: Alert,
        draft: DiagnosisDraft,
        cited: list[Evidence],
        context: DiagnosisContext | None,
    ) -> list[AffectedComponent]:
        known = set(context.available_components) if context else {alert.service}
        cited_ids = {item.id for item in cited}
        result: list[AffectedComponent] = []
        names: set[str] = set()
        for component in draft.affected_components:
            if component.name not in known:
                raise LLMResponseError("Affected component is not in the investigation context.")
            expected_role = "primary" if component.name == alert.service else "dependency"
            if component.role != expected_role:
                raise LLMResponseError("Affected component has an inconsistent role.")
            if component.name in names:
                raise LLMResponseError("Affected component is repeated.")
            if set(component.evidence_ids) - cited_ids:
                raise LLMResponseError("Affected component cites evidence absent from the report.")
            if component.role == "dependency" and not component.evidence_ids:
                raise LLMResponseError("Dependency impact requires an evidence reference.")
            names.add(component.name)
            result.append(component)
        if alert.service not in names:
            result.insert(
                0,
                AffectedComponent(
                    name=alert.service,
                    role="primary",
                    impact="Service named in the incoming alert; impact requires verification.",
                    evidence_ids=["alert:input"] if "alert:input" in cited_ids else [],
                ),
            )
        if len(result) > 20:
            raise LLMResponseError("Too many affected components.")
        return result

    @staticmethod
    def _assess(evidence: list[Evidence], observations: list[ToolObservation]) -> ReportAssessment:
        independent = [item for item in evidence if item.source != "alert"]
        simulated = sum(item.simulated for item in evidence)
        failures = sum(item.status == "error" for item in observations)
        limitations: list[str] = []
        if not independent:
            status = "alert_only"
            limitations.append("The report has no independent tool evidence.")
        elif all(item.simulated for item in independent):
            status = "simulated_context"
            limitations.append("All cited tool evidence comes from simulated data.")
        else:
            status = "observed_context"
        if failures:
            limitations.append(f"{failures} tool requests did not complete successfully.")
        if simulated:
            limitations.append("Fixture evidence cannot establish a live production root cause.")
        return ReportAssessment(
            evidence_count=len(evidence),
            simulated_evidence_count=simulated,
            independent_sources=sorted({item.source for item in independent}),
            tool_failures=failures,
            evidence_status=status,
            limitations=limitations,
        )

    def _data_gaps(
        self,
        draft: DiagnosisDraft,
        known: list[str],
        cited: list[Evidence],
        observations: list[ToolObservation],
        context: DiagnosisContext | None,
    ) -> list[str]:
        mandatory: list[str] = []
        if not any(item.source != "alert" for item in cited):
            mandatory.append("No independent tool evidence was cited in the diagnosis.")
        if any(item.simulated for item in cited):
            mandatory.append("Cited fixture evidence needs verification against live systems.")
        for observation in observations:
            mandatory.extend(observation.data_gaps)
        if context:
            mandatory.extend(context.data_gaps)
        combined = self._unique_text([*mandatory, *known, *draft.data_gaps])
        if len(combined) <= 20:
            return combined
        return [*combined[:19], f"{len(combined) - 19} additional data gaps were omitted."]

    @staticmethod
    def _unique_text(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = value.strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result

    @staticmethod
    def _timeline(evidence: list[Evidence]) -> list[ReportEvent]:
        events = [
            ReportEvent(
                timestamp=item.observed_at,
                source=item.source,
                service=item.service,
                description=item.summary,
                evidence_id=item.id,
                simulated=item.simulated,
            )
            for item in evidence
            if item.observed_at is not None
        ]
        events.sort(key=lambda event: (event.timestamp, event.evidence_id))
        return events
