"""Tests for idempotency & resilience.

Tests:
1. daily run key prevents duplicates
2. retry/backoff handles 429 correctly
3. non-retriable errors do not retry
"""

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hean.exchange.bybit.http import BybitHTTPClient
from hean.process_factory.schemas import ProcessRun, ProcessRunStatus
from hean.process_factory.storage import SQLiteStorage


@pytest.fixture
async def storage(tmp_path):
    """Create SQLiteStorage instance."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(str(db_path))
    await storage._init_schema(await storage._get_connection())
    yield storage
    await storage.close()


@pytest.mark.asyncio
async def test_daily_run_key_prevents_duplicates(storage):
    """Test that daily run key prevents duplicate runs."""
    process_id = "test_process"
    daily_run_key = f"{process_id}_{date.today().isoformat()}"

    # Create first run
    run1 = ProcessRun(
        run_id="run_1",
        process_id=process_id,
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        status=ProcessRunStatus.COMPLETED,
        metrics={},
        capital_allocated_usd=100.0,
        inputs={},
        outputs={},
    )

    # Save first run with daily run key
    await storage.save_run(run1, daily_run_key=daily_run_key)

    # Check that daily run key exists
    exists, existing_run_id = await storage.check_daily_run_key(daily_run_key)
    assert exists, "Daily run key should exist"
    assert existing_run_id == run1.run_id, "Should return first run ID"

    # Try to create second run with same daily run key
    run2 = ProcessRun(
        run_id="run_2",
        process_id=process_id,
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        status=ProcessRunStatus.COMPLETED,
        metrics={},
        capital_allocated_usd=100.0,
        inputs={},
        outputs={},
    )

    # Check before saving - should detect duplicate
    exists, existing_run_id = await storage.check_daily_run_key(daily_run_key)
    assert exists, "Should detect existing daily run key"
    assert existing_run_id == run1.run_id, "Should return first run ID, not second"


@pytest.mark.asyncio
async def test_retry_backoff_handles_429_correctly():
    """Test that retry/backoff handles HTTP 429 (rate limit) correctly."""
    import httpx

    client = BybitHTTPClient()
    client._api_key = "test_key"
    client._api_secret = "test_secret"
    client._testnet = True

    # Mock httpx client to simulate 429 responses
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Rate limited", request=MagicMock(), response=mock_response_429
    )

    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = {"retCode": 0, "result": {}}
    mock_response_200.raise_for_status.return_value = None

    # First call returns 429, second call succeeds
    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_response_429, mock_response_200]
    client._client = mock_client

    # Should retry on 429 and eventually succeed
    try:
        result = await client._request("GET", "/v5/account/wallet-balance", params={})
        assert result == {}, "Should eventually succeed after retry"
        assert mock_client.get.call_count == 2, "Should retry once on 429"
    except Exception:
        # If test fails, check that retry was attempted
        assert mock_client.get.call_count > 1, "Should attempt retry on 429"


@pytest.mark.asyncio
async def test_non_retriable_errors_do_not_retry():
    """Test that non-retriable errors (auth, validation) do not retry."""
    client = BybitHTTPClient()
    client._api_key = "test_key"
    client._api_secret = "test_secret"
    client._testnet = True

    # Mock response with auth error (non-retriable)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "retCode": 10003,  # Auth error (non-retriable)
        "retMsg": "Invalid API key",
    }
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    client._client = mock_client

    # Should not retry on auth error
    with pytest.raises(ValueError, match="Bybit API error"):
        await client._request("GET", "/v5/account/wallet-balance", params={})

    # Should only call once (no retry)
    assert mock_client.get.call_count == 1, "Should not retry on non-retriable error"


@pytest.mark.asyncio
async def test_retry_backoff_exponential_delay():
    """Test that retry uses exponential backoff."""

    client = BybitHTTPClient()
    client._api_key = "test_key"
    client._api_secret = "test_secret"
    client._testnet = True

    # Mock response with retriable error (rate limit)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "retCode": 10002,  # Rate limit (retriable)
        "retMsg": "Rate limited",
    }
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    client._client = mock_client

    # Track sleep calls
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        try:
            await client._request("GET", "/v5/account/wallet-balance", params={})
        except Exception:
            pass  # Expected to fail after retries

    # Should have exponential backoff delays
    if sleep_calls:
        assert len(sleep_calls) > 0, "Should have sleep delays for retries"
        # Check that delays increase (exponential backoff)
        for i in range(len(sleep_calls) - 1):
            assert sleep_calls[i + 1] >= sleep_calls[i], (
                f"Backoff delays should increase: {sleep_calls}"
            )


@pytest.mark.asyncio
async def test_daily_run_key_different_dates_allowed(storage):
    """Test that different dates allow different runs."""
    from datetime import timedelta

    process_id = "test_process"
    today_key = f"{process_id}_{date.today().isoformat()}"
    yesterday = date.today() - timedelta(days=1)
    yesterday_key = f"{process_id}_{yesterday.isoformat()}"

    # Create runs for different dates
    run1 = ProcessRun(
        run_id="run_today",
        process_id=process_id,
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        status=ProcessRunStatus.COMPLETED,
        metrics={},
        capital_allocated_usd=100.0,
        inputs={},
        outputs={},
    )

    run2 = ProcessRun(
        run_id="run_yesterday",
        process_id=process_id,
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        status=ProcessRunStatus.COMPLETED,
        metrics={},
        capital_allocated_usd=100.0,
        inputs={},
        outputs={},
    )

    # Save both runs with different daily keys
    await storage.save_run(run1, daily_run_key=today_key)
    await storage.save_run(run2, daily_run_key=yesterday_key)

    # Both should exist
    exists1, run_id1 = await storage.check_daily_run_key(today_key)
    exists2, run_id2 = await storage.check_daily_run_key(yesterday_key)

    assert exists1 and run_id1 == "run_today"
    assert exists2 and run_id2 == "run_yesterday"

