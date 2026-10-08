from datetime import timedelta

from app.integrations.logs.fake import FakeLogClient
from app.models.logs import LogEntry, LogQuery


async def test_fixture_search_returns_relevant_logs(query: LogQuery) -> None:
    result = await FakeLogClient.from_fixture().search(query)
    assert result.total == 3
    assert {entry.id for entry in result.entries} == {"p003", "p004", "p006"}
    assert result.simulated is True
    assert not result.truncated
    assert all(entry.service == query.service for entry in result.entries)
    assert all(entry.environment == query.environment for entry in result.entries)


async def test_search_keywords_are_or_case_insensitive(query: LogQuery) -> None:
    query.keywords = ["DEPLOYMENT", "pool exhausted"]
    result = await FakeLogClient.from_fixture().search(query)
    assert [entry.id for entry in result.entries] == ["p001", "p004"]


async def test_search_matches_attributes(query: LogQuery) -> None:
    query.keywords = ["sample-a01"]
    result = await FakeLogClient.from_fixture().search(query)
    assert [entry.id for entry in result.entries] == ["p003"]


async def test_search_without_keywords_returns_all_in_scope(query: LogQuery) -> None:
    query.keywords = []
    result = await FakeLogClient.from_fixture().search(query)
    assert result.total == 6
    assert [entry.id for entry in result.entries] == [f"p00{i}" for i in range(1, 7)]


async def test_search_filters_environment(query: LogQuery) -> None:
    query.environment = "staging"
    query.keywords = []
    result = await FakeLogClient.from_fixture().search(query)
    assert [entry.id for entry in result.entries] == ["s001"]


async def test_search_filters_service(query: LogQuery) -> None:
    query.service = "ledger-service"
    query.keywords = []
    result = await FakeLogClient.from_fixture().search(query)
    assert [entry.id for entry in result.entries] == ["l001"]


async def test_time_range_is_half_open(query: LogQuery) -> None:
    entries = [
        LogEntry(
            id=str(index),
            service=query.service,
            environment=query.environment,
            timestamp=time,
            level="INFO",
            message="record",
        )
        for index, time in enumerate(
            [
                query.start - timedelta(microseconds=1),
                query.start,
                query.end - timedelta(microseconds=1),
                query.end,
            ]
        )
    ]
    query.keywords = []
    result = await FakeLogClient(entries).search(query)
    assert [entry.id for entry in result.entries] == ["1", "2"]


async def test_search_limit_preserves_total(query: LogQuery) -> None:
    query.limit = 1
    result = await FakeLogClient.from_fixture().search(query)
    assert len(result.entries) == 1
    assert result.total == 3
    assert result.truncated
    assert result.entries[0].id == "p003"


async def test_no_matches_is_empty_success(query: LogQuery) -> None:
    query.keywords = ["nonexistent-token"]
    result = await FakeLogClient.from_fixture().search(query)
    assert result.entries == []
    assert result.total == 0
    assert result.truncated is False


async def test_fixture_does_not_shift_to_request_date(query: LogQuery) -> None:
    query.start += timedelta(days=1)
    query.end += timedelta(days=1)
    result = await FakeLogClient.from_fixture().search(query)
    assert result.total == 0


async def test_results_do_not_mutate_backend(query: LogQuery) -> None:
    client = FakeLogClient.from_fixture()
    first = await client.search(query)
    first.entries[0].attributes["dependency"] = "tampered"
    first.entries.clear()
    second = await client.search(query)
    assert second.entries[0].attributes["dependency"] == "payment-db"


async def test_input_entries_are_copied(query: LogQuery) -> None:
    item = LogEntry(
        id="a",
        service=query.service,
        environment=query.environment,
        timestamp=query.start,
        level="ERROR",
        message="error",
        attributes={"pod": "one"},
    )
    client = FakeLogClient([item])
    item.attributes["pod"] = "changed"
    result = await client.search(query)
    assert result.entries[0].attributes == {"pod": "one"}


async def test_equal_timestamps_have_stable_id_order(query: LogQuery) -> None:
    entries = [
        LogEntry(
            id=key,
            service=query.service,
            environment=query.environment,
            timestamp=query.start,
            level="ERROR",
            message="error",
        )
        for key in ["b", "a"]
    ]
    result = await FakeLogClient(entries).search(query)
    assert [entry.id for entry in result.entries] == ["a", "b"]
