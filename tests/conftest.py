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
