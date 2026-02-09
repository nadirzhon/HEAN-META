"""
Pytest configuration and fixtures for SYMBIONT X tests
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'symbol': 'BTCUSDT',
        'price': 50000.0,
        'bid': 49995.0,
        'ask': 50005.0,
        'volume': 1000.0,
        'high': 51000.0,
        'low': 49000.0,
        'timestamp': 1234567890000,
        'timestamp_ns': 1234567890000000000
    }


@pytest.fixture
def sample_genome():
    """Sample genome for testing"""
    from hean.symbiont_x.genome_lab import create_random_genome
    return create_random_genome("TestStrategy")


@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing"""
    from hean.symbiont_x.capital_allocator import Portfolio
    return Portfolio(
        portfolio_id=str(uuid.uuid4()),
        name="Test Portfolio",
        total_capital=10000.0
    )


@pytest.fixture
def sample_decision():
    """Sample decision for testing"""
    from hean.symbiont_x.decision_ledger import Decision, DecisionType
    return Decision(
        decision_id=str(uuid.uuid4()),
        decision_type=DecisionType.OPEN_POSITION,
        reason="Test decision",
        strategy_id="TestStrategy",
        symbol="BTCUSDT"
    )


@pytest.fixture
def sample_event_envelope():
    """Sample event envelope for testing"""
    from hean.symbiont_x.nervous_system import EventEnvelope, EventType
    return EventEnvelope(
        event_type=EventType.MARKET_DATA,
        symbol="BTCUSDT",
        data={'price': 50000.0}
    )


@pytest.fixture
def sample_ohlcv_history():
    """Sample OHLCV history for testing (20 candles)"""
    import random
    base_price = 50000.0
    history = []

    for i in range(20):
        open_price = base_price + random.uniform(-500, 500)
        close_price = open_price + random.uniform(-300, 300)
        high_price = max(open_price, close_price) + random.uniform(0, 200)
        low_price = min(open_price, close_price) - random.uniform(0, 200)
        volume = random.uniform(800, 1200)

        history.append({
            'timestamp': 1234567890000 + i * 60000,  # 1 minute apart
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

        base_price = close_price  # Next candle starts where this one ended

    return history
