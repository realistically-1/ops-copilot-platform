import asyncio
from collections import deque
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest

from app.core.config import Settings
from app.main import create_app
from app.models.alert import Alert
from app.models.execution import ExecutionFilter, ExecutionStatus
from app.models.logs import LogLevel, LogQuery
from tests.helpers import draft, finish, search, service_info
from tests.unit.test_llm_client import completion

pytestmark = pytest.mark.integration


async def test_diagnosis_round_trip_has_context_progress_and_usage(
    settings: Settings, alert: Alert, query: LogQuery
) -> None:
    replies = deque([service_info(), search(query), finish(), draft(["alert:input", "log:p003"])])

    def handler(request: httpx.Request) -> httpx.Response:
        body = completion(replies.popleft().model_dump_json())
        body.update(
            id="completion-test",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        return httpx.Response(200, json=body)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(settings)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    with (
        patch("app.main.httpx.AsyncClient", return_value=upstream),
        patch("app.main.configure_logging"),
    ):
        async with app.router.lifespan_context(app), client:
            response = await client.post("/api/v1/diagnosis", json=alert.model_dump(mode="json"))
            assert response.status_code == 200, response.text
            saved = await client.get(response.headers["location"])
            assert saved.status_code == 200
            record = saved.json()
            assert record["status"] == "succeeded"
            assert record["context"]["service_info"]["owner"] == "payments-platform"
            assert record["metadata"]["steps"] == 8
            assert len(record["metadata"]["timeline"]) == 8
            assert record["metadata"]["usage"]["total_tokens"] == 60
            assert record["result"]["assessment"]["evidence_status"] == "simulated_context"
            assert record["result"]["timeline"][0]["evidence_id"] == "log:p003"
            page = await client.get("/api/v1/diagnoses", params={"status": "succeeded", "limit": 1})
            assert page.status_code == 200
            assert page.json()["total"] == 1
            assert "evidence" not in page.json()["items"][0]
            missing = await client.get(f"/api/v1/diagnoses/{uuid4()}")
            assert missing.status_code == 404
            invalid = await client.get("/api/v1/diagnoses", params={"limit": 0})
            assert invalid.status_code == 422


async def test_failed_diagnosis_is_queryable_without_private_errors(
    settings: Settings, alert: Alert
) -> None:
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="private"))
    )
    app = create_app(settings)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    with (
        patch("app.main.httpx.AsyncClient", return_value=upstream),
        patch("app.main.configure_logging"),
    ):
        async with app.router.lifespan_context(app), client:
            response = await client.post("/api/v1/diagnosis", json=alert.model_dump(mode="json"))
            assert response.status_code == 502
            diagnosis_id = response.json()["error"]["diagnosis_id"]
            saved = await client.get(f"/api/v1/diagnoses/{diagnosis_id}")
            assert saved.json()["status"] == "failed"
            assert saved.json()["result"] is None
            assert saved.json()["metadata"]["steps"] == 1
            assert saved.json()["metadata"]["usage"]["failed"] == 1
            assert saved.json()["metadata"]["usage"]["total_tokens"] is None
            assert "private" not in saved.text


async def test_catalog_and_log_api(settings: Settings, query: LogQuery) -> None:
    app = create_app(settings)
    with patch("app.main.configure_logging"):
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                service = await client.get(
                    "/api/v1/services/payment-service", params={"environment": "prod"}
                )
                assert service.status_code == 200
                assert service.json()["service"]["runtime"]["version"] == "3.12"
                assert (
                    service.json()["service"]["environments"][0]["current_version"]
                    == "2026.10.08-rc2"
                )
                dependencies = await client.get("/api/v1/services/payment-service/dependencies")
                items = {item["name"]: item for item in dependencies.json()}
                assert items["ledger-service"]["available_in_environment"]
                assert not items["payment-db"]["registered"]
                page = await client.get("/api/v1/services", params={"owner": "payments-platform"})
                assert page.json()["total"] == 1
                assert (await client.get("/api/v1/services/missing")).status_code == 404
                query.levels = [LogLevel.ERROR]
                query.keywords = ["database", "timeout"]
                query.match_mode = "all"
                logs = await client.post("/api/v1/logs/search", json=query.model_dump(mode="json"))
                assert logs.status_code == 200
                assert logs.json()["total"] == 2
                assert logs.json()["statistics"]["error_count"] == 2
                assert logs.json()["facets"]["dependency"][0]["value"] == "payment-db"


async def test_running_query_and_request_cancellation(settings: Settings, alert: Alert) -> None:
    app = create_app(settings)
    entered = asyncio.Event()

    async def wait_for_model(*args: object, **kwargs: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    with patch("app.main.configure_logging"):
        async with app.router.lifespan_context(app):
            service = app.state.diagnosis_service
            with patch.object(service._agent, "run", new=AsyncMock(side_effect=wait_for_model)):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    task = asyncio.create_task(
                        client.post("/api/v1/diagnosis", json=alert.model_dump(mode="json"))
                    )
                    try:
                        await asyncio.wait_for(entered.wait(), 2)
                        page = await client.get("/api/v1/diagnoses", params={"status": "running"})
                        assert page.json()["total"] == 1
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        records = service.list(ExecutionFilter(status=ExecutionStatus.CANCELLED))
                        assert records.total == 1
                        assert records.items[0].error.code == "request_cancelled"
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
