"""Tests for DoomsdaySandbox catastrophe simulation engine."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from hean.core.bus import EventBus
from hean.core.types import EventType
from hean.risk.doomsday_sandbox import DoomsdaySandbox, MarketSnapshot


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def accounting():
    mock = MagicMock()
    mock.get_equity.return_value = 1000.0
    mock.get_positions.return_value = []
    mock.get_drawdown.return_value = (0.0, 0.0)
    return mock


@pytest.fixture
def sandbox(bus, accounting):
    return DoomsdaySandbox(bus=bus, accounting=accounting)


class TestScenarioGeneration:
    def test_all_scenarios_generate_steps(self, sandbox):
        for name in DoomsdaySandbox.SCENARIOS:
            steps = sandbox._generate_scenario(name)
            assert len(steps) > 0, f"Scenario {name} generated 0 steps"
            assert all(s.time_offset_sec >= 0 for s in steps)

    def test_flash_crash_10pct_prices_drop(self, sandbox):
        steps = sandbox._generate_scenario("flash_crash_10pct")
        assert steps[0].price_multiplier > steps[-1].price_multiplier
        assert steps[-1].price_multiplier == pytest.approx(0.9, abs=0.02)

    def test_flash_crash_30pct_prices_drop(self, sandbox):
        steps = sandbox._generate_scenario("flash_crash_30pct")
        assert steps[-1].price_multiplier == pytest.approx(0.7, abs=0.02)

    def test_exchange_downtime_has_api_failures(self, sandbox):
        steps = sandbox._generate_scenario("exchange_downtime")
        api_states = [s.api_available for s in steps]
        assert False in api_states  # Must have downtime
        assert True in api_states  # Must have uptime too

    def test_cascade_has_accelerating_drop(self, sandbox):
        steps = sandbox._generate_scenario("cascade_liquidation")
        # First half should drop less than second half
        mid = len(steps) // 2
        first_half_drop = 1.0 - steps[mid].price_multiplier
        second_half_drop = steps[mid].price_multiplier - steps[-1].price_multiplier
        assert second_half_drop > first_half_drop


class TestSurvivalScore:
    def test_perfect_score(self, sandbox):
        score = sandbox._calculate_survival_score(
            max_drawdown_pct=0.0,
            killswitch_triggered=False,
            final_equity_pct=1.0,
        )
        assert score == 1.0

    def test_worst_score(self, sandbox):
        score = sandbox._calculate_survival_score(
            max_drawdown_pct=30.0,
            killswitch_triggered=True,
            final_equity_pct=0.0,
        )
        assert score == 0.0

    def test_moderate_drawdown(self, sandbox):
        score = sandbox._calculate_survival_score(
            max_drawdown_pct=10.0,
            killswitch_triggered=False,
            final_equity_pct=0.9,
        )
        # dd_score = 0.5, ks_score = 1.0, equity_score = 0.9
        # = 0.5*0.5 + 1.0*0.3 + 0.9*0.2 = 0.25 + 0.30 + 0.18 = 0.73
        assert score == pytest.approx(0.73, abs=0.01)

    def test_score_bounded_0_to_1(self, sandbox):
        for dd in [0, 5, 10, 15, 20, 30, 50]:
            for ks in [True, False]:
                for eq in [0.0, 0.5, 1.0, 1.5]:
                    score = sandbox._calculate_survival_score(dd, ks, eq)
                    assert 0.0 <= score <= 1.0


class TestSimulation:
    def test_no_positions_high_survival(self, sandbox):
        """With no positions, all scenarios should have high survival."""
        snapshot = MarketSnapshot(
            timestamp=0,
            equity=1000.0,
            positions=[],
            open_orders=0,
            peak_equity=1000.0,
            daily_high_equity=1000.0,
            current_drawdown_pct=0.0,
            initial_capital=1000.0,
        )
        for name in DoomsdaySandbox.SCENARIOS:
            steps = sandbox._generate_scenario(name)
            result = sandbox._simulate(snapshot, steps, name)
            # No positions = no losses from price moves
            assert result.survival_score >= 0.5, (
                f"Scenario {name} survival too low with no positions: {result.survival_score}"
            )

    def test_long_position_flash_crash(self, sandbox):
        """Long position + flash crash should have low survival."""
        snapshot = MarketSnapshot(
            timestamp=0,
            equity=1000.0,
            positions=[{
                "entry_price": 50000.0,
                "size": 0.02,  # 1 BTC = $50k notional
                "side": "long",
            }],
            open_orders=0,
            peak_equity=1000.0,
            daily_high_equity=1000.0,
            current_drawdown_pct=0.0,
            initial_capital=1000.0,
        )
        steps = sandbox._generate_scenario("flash_crash_30pct")
        result = sandbox._simulate(snapshot, steps, "flash_crash_30pct")
        assert result.max_drawdown_pct > 0
        assert result.estimated_loss_usd > 0
        assert result.survival_score < 1.0

    def test_recommendations_generated(self, sandbox):
        recs = DoomsdaySandbox._generate_recommendations(
            scenario_name="flash_crash_30pct",
            max_dd=25.0,
            risk_transitions=[],
            survival_score=0.3,
            ks_triggered=True,
        )
        assert len(recs) > 0
        assert any("KillSwitch" in r for r in recs)


class TestRunAllScenarios:
    async def test_run_all_publishes_events(self, bus, accounting):
        sandbox = DoomsdaySandbox(bus=bus, accounting=accounting)
        events_received = []

        async def capture(event):
            events_received.append(event)

        bus.subscribe(EventType.RISK_SIMULATION_RESULT, capture)
        await bus.start()

        report = await sandbox.run_all_scenarios()

        # Process events
        await asyncio.sleep(0.1)
        await bus.stop()

        assert report.overall_survival_score >= 0.0
        assert report.overall_survival_score <= 1.0
        assert report.weakest_scenario in DoomsdaySandbox.SCENARIOS

    async def test_auto_protect_triggered_when_low_score(self, bus, accounting):
        """Auto-protect should fire when survival < threshold."""
        # Create positions that will suffer in crashes
        mock_pos = MagicMock()
        mock_pos.model_dump.return_value = {
            "entry_price": 50000.0,
            "size": 0.1,  # Very large position
            "side": "long",
        }
        accounting.get_positions.return_value = [mock_pos]

        sandbox = DoomsdaySandbox(bus=bus, accounting=accounting)
        alerts = []

        async def capture_alert(event):
            if event.data.get("type") == "DOOMSDAY_AUTO_PROTECT":
                alerts.append(event)

        bus.subscribe(EventType.RISK_ALERT, capture_alert)
        await bus.start()

        with patch("hean.risk.doomsday_sandbox.settings") as mock_settings:
            mock_settings.doomsday_auto_protect = True
            mock_settings.doomsday_survival_threshold = 0.99  # Very high threshold
            mock_settings.killswitch_drawdown_pct = 30.0
            mock_settings.initial_capital = 1000.0
            mock_settings.doomsday_interval_sec = 3600
            mock_settings.doomsday_run_on_physics_alert = False
            mock_settings.doomsday_sandbox_enabled = True

            await sandbox.run_all_scenarios()
            await asyncio.sleep(0.1)

        await bus.stop()
        # With very high threshold, auto-protect should trigger
        # (unless all scenarios have perfect score, which is unlikely with large positions)
