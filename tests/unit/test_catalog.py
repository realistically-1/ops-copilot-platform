import pytest

from app.models.service import ServiceInfo
from app.services.catalog import ServiceCatalog


def test_catalog_returns_environment_specific_version() -> None:
    catalog = ServiceCatalog.from_fixture()
    production = catalog.get("payment-service", "prod")
    staging = catalog.get("payment-service", "staging")
    assert production is not None and staging is not None
    assert production.deploy_version == "2026.10.08-rc2"
    assert staging.deploy_version == "2026.10.08-rc3"
    assert production.metadata["replicas"] == "4"
    assert staging.metadata["replicas"] == "1"


def test_unknown_service_is_not_an_exception() -> None:
    catalog = ServiceCatalog.from_fixture()
    assert catalog.get("unknown-service", "prod") is None
    assert catalog.get("payment-service", "unknown-env") is None


def test_catalog_rejects_duplicate_key() -> None:
    service = ServiceInfo(
        service="one",
        environment="prod",
        owner="team",
        language="Python",
        dependencies=[],
        deploy_version="1",
    )
    with pytest.raises(ValueError, match="duplicate"):
        ServiceCatalog([service, service])


def test_catalog_copies_mutable_fields() -> None:
    catalog = ServiceCatalog.from_fixture()
    first = catalog.get("payment-service", "prod")
    assert first is not None
    first.dependencies.clear()
    first.metadata["region"] = "tampered"
    second = catalog.get("payment-service", "prod")
    assert second is not None
    assert "payment-db" in second.dependencies
    assert second.metadata["region"] == "cn-east-1"


def test_catalog_copies_inputs() -> None:
    service = ServiceInfo(
        service="one",
        environment="prod",
        owner="team",
        language="Python",
        dependencies=["db"],
        deploy_version="1",
    )
    catalog = ServiceCatalog([service])
    service.dependencies.append("new")
    stored = catalog.get("one", "prod")
    assert stored is not None
    assert stored.dependencies == ["db"]
