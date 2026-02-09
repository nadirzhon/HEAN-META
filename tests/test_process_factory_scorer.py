"""Tests for Process Factory scorer."""


from hean.process_factory.schemas import Opportunity, OpportunitySource
from hean.process_factory.scorer import rank_opportunities, score_opportunity


def test_score_opportunity():
    """Test opportunity scoring."""
    opp = Opportunity(
        id="test",
        source=OpportunitySource.TRADING,
        expected_profit_usd=100.0,
        time_hours=1.0,
        risk_factor=1.0,
        complexity=1,
        confidence=1.0,
    )
    score = score_opportunity(opp)
    assert score > 0
    assert score == 100.0 / (1.0 * 1.0 * 1.0)  # 100.0


def test_score_opportunity_with_factors():
    """Test opportunity scoring with various factors."""
    # High profit, low time, low risk = high score
    opp1 = Opportunity(
        id="test1",
        source=OpportunitySource.TRADING,
        expected_profit_usd=100.0,
        time_hours=1.0,
        risk_factor=1.0,
        complexity=1,
        confidence=1.0,
    )
    score1 = score_opportunity(opp1)

    # Lower profit, higher time, higher risk = lower score
    opp2 = Opportunity(
        id="test2",
        source=OpportunitySource.TRADING,
        expected_profit_usd=50.0,
        time_hours=10.0,
        risk_factor=3.0,
        complexity=5,
        confidence=0.5,
    )
    score2 = score_opportunity(opp2)

    assert score1 > score2


def test_rank_opportunities():
    """Test opportunity ranking."""
    opportunities = [
        Opportunity(
            id="low",
            source=OpportunitySource.TRADING,
            expected_profit_usd=10.0,
            time_hours=10.0,
            risk_factor=2.0,
            complexity=2,
            confidence=0.5,
        ),
        Opportunity(
            id="high",
            source=OpportunitySource.TRADING,
            expected_profit_usd=100.0,
            time_hours=1.0,
            risk_factor=1.0,
            complexity=1,
            confidence=1.0,
        ),
    ]
    ranked = rank_opportunities(opportunities)
    assert len(ranked) == 2
    assert ranked[0][0].id == "high"  # Higher score first
    assert ranked[1][0].id == "low"


def test_rank_opportunities_min_score():
    """Test opportunity ranking with minimum score filter."""
    opportunities = [
        Opportunity(
            id="low",
            source=OpportunitySource.TRADING,
            expected_profit_usd=1.0,
            time_hours=100.0,
            risk_factor=5.0,
            complexity=5,
            confidence=0.1,
        ),
        Opportunity(
            id="high",
            source=OpportunitySource.TRADING,
            expected_profit_usd=100.0,
            time_hours=1.0,
            risk_factor=1.0,
            complexity=1,
            confidence=1.0,
        ),
    ]
    ranked = rank_opportunities(opportunities, min_score=10.0)
    assert len(ranked) == 1
    assert ranked[0][0].id == "high"

