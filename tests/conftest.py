"""Shared test fixtures and markers."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked @pytest.mark.bybit when no valid API key is available."""
    api_key = os.environ.get("BYBIT_API_KEY", "")
    if api_key and api_key not in ("test_key", "your_key_here", ""):
        return  # Real API key present, don't skip

    skip_bybit = pytest.mark.skip(reason="No valid BYBIT_API_KEY; skipping live Bybit tests")
    for item in items:
        if "bybit" in item.keywords:
            item.add_marker(skip_bybit)


@pytest.fixture(autouse=True)
def _reset_price_anomaly_detector():
    """Reset the module-level PriceAnomalyDetector singleton between tests.

    Without this, price history from earlier tests causes z-score anomaly
    detection to fire on normal prices, blocking tick processing in strategies.
    """
    from hean.risk.price_anomaly_detector import price_anomaly_detector

    price_anomaly_detector._price_history.clear()
    price_anomaly_detector._last_prices.clear()
    price_anomaly_detector._last_update_time.clear()
    price_anomaly_detector._anomaly_count.clear()
    price_anomaly_detector._blocked_until.clear()
