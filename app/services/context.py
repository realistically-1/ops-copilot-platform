import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import CatalogError, DomainValidationError
from app.models.alert import Alert
from app.models.context import DiagnosisContext, DiagnosisOptions
from app.models.logs import TimeRange
from app.models.service import ServiceInfo
from app.services.catalog import ServiceCatalog

logger = logging.getLogger(__name__)


class DiagnosisContextBuilder:
    def __init__(self, catalog: ServiceCatalog, settings: Settings) -> None:
        self._catalog = catalog
        self._settings = settings

    def build(
        self, diagnosis_id: UUID, alert: Alert, options: DiagnosisOptions
    ) -> DiagnosisContext:
        before = options.lookback_minutes
        after = options.lookahead_minutes
        before = self._settings.log_window_before_minutes if before is None else before
        after = self._settings.log_window_after_minutes if after is None else after
        try:
            window = TimeRange(
                start=alert.timestamp - timedelta(minutes=before),
                end=alert.timestamp + timedelta(minutes=after),
            )
        except (OverflowError, ValueError) as exc:
            raise DomainValidationError(
                "Alert timestamp cannot form a valid investigation window."
            ) from exc
        gaps: list[str] = []
        service_info = None
        try:
            service_info = self._catalog.get(alert.service, alert.environment)
        except CatalogError:
            logger.warning("context.catalog_unavailable", extra={"service": alert.service})
            gaps.append("Initial service metadata could not be loaded.")
        if service_info is None:
            gaps.append("No initial service catalog snapshot is available for this environment.")
        elif service_info.simulated:
            gaps.append("Initial service context is a fixture snapshot, not live deployment state.")
        if service_info:
            gaps.extend(self._catalog_gaps(service_info, alert))
        components = [alert.service]
        if service_info:
            components.extend(service_info.dependencies)
        return DiagnosisContext(
            diagnosis_id=diagnosis_id,
            alert=alert,
            collected_at=datetime.now(UTC),
            window=window,
            service_info=service_info,
            initial_keywords=options.keywords,
            initial_levels=options.log_levels,
            available_components=list(dict.fromkeys(components)),
            data_gaps=gaps,
        )

    @staticmethod
    def _catalog_gaps(info: ServiceInfo, alert: Alert) -> list[str]:
        gaps: list[str] = []
        if info.catalog_updated_at is not None:
            age = (alert.timestamp - info.catalog_updated_at).total_seconds()
            if age < 0:
                gaps.append(
                    "Catalog snapshot postdates the alert; historical version is uncertain."
                )
            elif age > 86400:
                gaps.append("Catalog snapshot is more than one day older than the alert.")
        if not info.contact:
            gaps.append("Service owner contact is not recorded in the catalog.")
        if not info.dependencies:
            gaps.append(
                "Catalog lists no dependencies; this does not prove the service is independent."
            )
        if "region" not in info.metadata:
            gaps.append("Deployment region is not recorded in the catalog.")
        if info.metadata.get("replicas") == "0":
            gaps.append(
                "Catalog snapshot records zero replicas; current runtime state was not queried."
            )
        return gaps
