import logging

from app.core.exceptions import CatalogError
from app.models.diagnosis import Evidence
from app.models.tools import ServiceInfoArguments, ToolObservation
from app.services.catalog import ServiceCatalog

logger = logging.getLogger(__name__)


class GetServiceInfoTool:
    name = "get_service_info"
    description = (
        "Get owner, language, dependencies, version and metadata for a service environment."
    )

    def __init__(self, catalog: ServiceCatalog) -> None:
        self._catalog = catalog

    def __call__(self, arguments: ServiceInfoArguments) -> ToolObservation:
        try:
            info = self._catalog.get(arguments.service, arguments.environment)
        except CatalogError:
            logger.warning(
                "service_info.failed",
                extra={"tool_name": self.name, "service": arguments.service},
            )
            return ToolObservation(
                tool=self.name,
                status="error",
                summary="Service catalog lookup failed.",
                data_gaps=["Service metadata could not be retrieved."],
                error_code="catalog_unavailable",
            )
        if info is None:
            return ToolObservation(
                tool=self.name,
                status="empty",
                summary="Service is not present in the catalog for this environment.",
                data_gaps=[
                    f"No service catalog entry for {arguments.service}/{arguments.environment}."
                ],
            )
        fields = {
            "owner": info.owner,
            "tier": info.tier,
            "language": info.language,
            "dependencies": ", ".join(info.dependencies) or "none recorded",
            "deploy_version": info.deploy_version,
            **{f"metadata.{key}": value for key, value in sorted(info.metadata.items())},
        }
        evidence = [
            Evidence(
                id=f"service:{info.service}:{info.environment}:{key}",
                source=self.name,
                summary=f"{key}: {value}"[:2000],
                reference=f"{info.source}/{info.service}/{info.environment}#{key}",
                simulated=info.simulated,
                observed_at=info.catalog_updated_at,
                service=info.service,
                environment=info.environment,
            )
            for key, value in fields.items()
        ]
        gaps = []
        if info.simulated:
            gaps.append(
                "Service metadata is a fixture snapshot and may not reflect current deployment."
            )
        if any(len(f"{key}: {value}") > 2000 for key, value in fields.items()):
            gaps.append("Some catalog evidence summaries were shortened; consult service details.")
        logger.info("service_info.completed", extra={"service": info.service})
        return ToolObservation(
            tool=self.name,
            status="ok",
            summary=f"Catalog metadata for {info.service}/{info.environment}.",
            evidence=evidence,
            data_gaps=gaps,
            details={"service": info.model_dump(mode="json")},
        )
