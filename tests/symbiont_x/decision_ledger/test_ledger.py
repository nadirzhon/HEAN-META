"""
Unit tests for Decision Ledger
"""

import pytest
from hean.symbiont_x.decision_ledger import (
    DecisionLedger,
    Decision,
    DecisionType,
    DecisionOutcome
)
import uuid


class TestDecisionLedger:
    """Test DecisionLedger class"""

    def test_ledger_creation(self):
        """Test creating a ledger"""
        ledger = DecisionLedger()
        assert ledger is not None
        assert ledger.total_decisions == 0

    def test_record_decision(self, sample_decision):
        """Test recording a decision"""
        ledger = DecisionLedger()

        ledger.record_decision(sample_decision)

        assert ledger.total_decisions == 1
        assert sample_decision.decision_id in ledger.decision_index

    def test_get_decision_by_id(self, sample_decision):
        """Test retrieving decision by ID"""
        ledger = DecisionLedger()
        ledger.record_decision(sample_decision)

        retrieved = ledger.get_decision(sample_decision.decision_id)

        assert retrieved is not None
        assert retrieved.decision_id == sample_decision.decision_id
        assert retrieved.decision_type == sample_decision.decision_type

    def test_get_decisions_by_strategy(self):
        """Test filtering decisions by strategy"""
        ledger = DecisionLedger()

        # Create decisions for different strategies
        for i in range(5):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.OPEN_POSITION,
                reason=f"Test {i}",
                strategy_id=f"Strategy{i % 2}",  # Strategy0 or Strategy1
                symbol="BTCUSDT"
            )
            ledger.record_decision(decision)

        # Get decisions for Strategy0 using the get_decisions method
        strategy0_decisions = ledger.get_decisions(strategy_id="Strategy0")

        assert len(strategy0_decisions) == 3  # 0, 2, 4
        assert all(d.strategy_id == "Strategy0" for d in strategy0_decisions)

    def test_get_decisions_by_symbol(self):
        """Test filtering decisions by symbol"""
        ledger = DecisionLedger()

        symbols = ["BTCUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT", "BTCUSDT"]

        for i, symbol in enumerate(symbols):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.OPEN_POSITION,
                reason=f"Test {i}",
                strategy_id="TestStrategy",
                symbol=symbol
            )
            ledger.record_decision(decision)

        # Get BTC decisions using get_decisions method
        btc_decisions = ledger.get_decisions(symbol="BTCUSDT")

        assert len(btc_decisions) == 3
        assert all(d.symbol == "BTCUSDT" for d in btc_decisions)

    def test_get_decisions_by_type(self):
        """Test filtering decisions by type"""
        ledger = DecisionLedger()

        decision_types = [
            DecisionType.OPEN_POSITION,
            DecisionType.CLOSE_POSITION,
            DecisionType.OPEN_POSITION,
            DecisionType.CANCEL_ORDER,
            DecisionType.OPEN_POSITION
        ]

        for i, dec_type in enumerate(decision_types):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=dec_type,
                reason=f"Test {i}",
                strategy_id="TestStrategy",
                symbol="BTCUSDT"
            )
            ledger.record_decision(decision)

        # Get OPEN_POSITION decisions using get_decisions method
        open_decisions = ledger.get_decisions(decision_type=DecisionType.OPEN_POSITION)

        assert len(open_decisions) == 3
        assert all(d.decision_type == DecisionType.OPEN_POSITION for d in open_decisions)

    def test_get_recent_decisions(self):
        """Test getting recent N decisions"""
        ledger = DecisionLedger()

        # Add 10 decisions
        for i in range(10):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.OPEN_POSITION,
                reason=f"Test {i}",
                strategy_id="TestStrategy",
                symbol="BTCUSDT"
            )
            ledger.record_decision(decision)

        # Get last 5 using get_recent_decisions
        recent = ledger.get_recent_decisions(n=5)

        assert len(recent) == 5

    def test_calculate_statistics(self):
        """Test calculating statistics including success rate"""
        ledger = DecisionLedger()

        # Add 10 decisions: 7 success, 3 failure
        for i in range(10):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.OPEN_POSITION,
                reason=f"Test {i}",
                strategy_id="TestStrategy",
                symbol="BTCUSDT"
            )
            # Set outcome directly on decision
            outcome = DecisionOutcome.SUCCESS if i < 7 else DecisionOutcome.FAILURE
            decision.mark_completed(outcome)
            ledger.record_decision(decision)

        stats = ledger.get_statistics()

        assert stats['success_rate'] == 0.7  # 7/10

    def test_calculate_pnl_from_statistics(self):
        """Test calculating total PnL from statistics"""
        ledger = DecisionLedger()

        pnls = [100.0, -50.0, 75.0, -25.0, 200.0]

        for i, pnl in enumerate(pnls):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.CLOSE_POSITION,
                reason=f"Test {i}",
                strategy_id="TestStrategy",
                symbol="BTCUSDT",
                pnl_impact=pnl  # Set pnl_impact directly
            )
            ledger.record_decision(decision)

        stats = ledger.get_statistics()

        assert stats['total_pnl_impact'] == sum(pnls)  # 300.0

    def test_ledger_persistence(self, sample_decision):
        """Test that ledger maintains decision order"""
        ledger = DecisionLedger()

        decision_ids = []
        for i in range(5):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.OPEN_POSITION,
                reason=f"Test {i}",
                strategy_id="TestStrategy",
                symbol="BTCUSDT"
            )
            ledger.record_decision(decision)
            decision_ids.append(decision.decision_id)

        # Get all decisions from the deque
        all_decisions = list(ledger.decisions)

        # Check order is preserved
        assert [d.decision_id for d in all_decisions] == decision_ids

    def test_ledger_statistics(self):
        """Test ledger statistics calculation"""
        ledger = DecisionLedger()

        # Add mix of decisions
        for i in range(20):
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.OPEN_POSITION if i % 2 == 0 else DecisionType.CLOSE_POSITION,
                reason=f"Test {i}",
                strategy_id=f"Strategy{i % 3}",
                symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT"
            )
            ledger.record_decision(decision)

        stats = ledger.get_statistics()

        assert stats['total_decisions'] == 20
        assert 'decisions_by_type' in stats
        assert 'total_strategies' in stats

    def test_get_related_decisions(self):
        """Test getting related decisions"""
        ledger = DecisionLedger()

        # Create parent decision
        parent = Decision(
            decision_id="parent-1",
            decision_type=DecisionType.OPEN_POSITION,
            reason="Parent decision",
            strategy_id="TestStrategy",
            symbol="BTCUSDT"
        )
        ledger.record_decision(parent)

        # Create child decision
        child = Decision(
            decision_id="child-1",
            decision_type=DecisionType.CLOSE_POSITION,
            reason="Child decision",
            strategy_id="TestStrategy",
            symbol="BTCUSDT",
            parent_decision_id="parent-1"
        )
        ledger.record_decision(child)

        # Get related decisions
        related = ledger.get_related_decisions("child-1")

        assert len(related) == 1
        assert related[0].decision_id == "parent-1"

    def test_get_decision_chain(self):
        """Test getting decision chain from parent to child"""
        ledger = DecisionLedger()

        # Create chain of decisions
        d1 = Decision(
            decision_id="d1",
            decision_type=DecisionType.OPEN_POSITION,
            reason="First",
            strategy_id="TestStrategy",
            symbol="BTCUSDT"
        )
        ledger.record_decision(d1)

        d2 = Decision(
            decision_id="d2",
            decision_type=DecisionType.MODIFY_POSITION,
            reason="Second",
            strategy_id="TestStrategy",
            symbol="BTCUSDT",
            parent_decision_id="d1"
        )
        ledger.record_decision(d2)

        d3 = Decision(
            decision_id="d3",
            decision_type=DecisionType.CLOSE_POSITION,
            reason="Third",
            strategy_id="TestStrategy",
            symbol="BTCUSDT",
            parent_decision_id="d2"
        )
        ledger.record_decision(d3)

        # Get chain from last decision
        chain = ledger.get_decision_chain("d3")

        assert len(chain) == 3
        assert [d.decision_id for d in chain] == ["d1", "d2", "d3"]
