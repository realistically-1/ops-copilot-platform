from typing import Protocol

from app.models.service import CatalogService


class CatalogClient(Protocol):
    source: str
    simulated: bool

    def get(self, service: str) -> CatalogService | None: ...

    def list_services(self) -> list[CatalogService]: ...
