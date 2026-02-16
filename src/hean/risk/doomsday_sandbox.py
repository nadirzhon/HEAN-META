"""DoomsdaySandbox — Catastrophe simulation engine for resilience scoring.

Runs 7 catastrophic market scenarios against the system's CURRENT portfolio
state using SHADOW copies of RiskGovernor and KillSwitch. Never mutates live state.

Publishes RISK_SIMULATION_RESULT events with survival scores (0.0-1.0).
Optionally auto-triggers SOFT_BRAKE when survival < threshold.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


@dataclass
class MarketSnapshot:
    """Snapshot of current market state for simulation baseline."""

    timestamp: float
    equity: float
    positions: list[dict[str, Any]]
    open_orders: int
    peak_equity: float
    daily_high_equity: float
    current_drawdown_pct: float
    initial_capital: float


@dataclass
class SimStep:
    """A single step in scenario simulation."""

    time_offset_sec: float
    price_multiplier: float = 1.0  # 0.7 = -30% crash
    volatility_multiplier: float = 1.0  # 5.0 = 5x vol spike
    spread_multiplier: float = 1.0  # 10.0 = 10x spread
    funding_rate_delta: float = 0.0  # +0.02 = +2% funding shock
    api_available: bool = True  # False = exchange down


@dataclass
class ScenarioResult:
    """Result of a single scenario simulation."""

    scenario_name: str
    duration_sec: int
    max_drawdown_pct: float
    risk_state_transitions: list[tuple[float, str, str]]  # [(time, old, new)]
    killswitch_triggered: bool
    killswitch_trigger_time_sec: float | None
    positions_at_risk: int
    estimated_loss_usd: float
    estimated_loss_pct: float
    survival_score: float  # 0.0-1.0
    final_equity: float
    recommendations: list[str]


@dataclass
class SandboxReport:
    """Full report from running all scenarios."""

    timestamp: float
    scenarios: dict[str, ScenarioResult]
    overall_survival_score: float
    weakest_scenario: str
    auto_protect_triggered: bool


class DoomsdaySandbox:
    """Catastrophe simulation engine using shadow risk instances."""

    SCENARIOS = [
        "flash_crash_10pct",
        "flash_crash_30pct",
        "volatility_spike_5x",
        "spread_blowout_50x",
        "funding_shock_2pct",
        "exchange_downtime",
        "cascade_liquidation",
    ]

    def __init__(
        self,
        bus: EventBus,
        accounting: PortfolioAccounting,
    ) -> None:
        self._bus = bus
        self._accounting = accounting
        self._running = False
        self._periodic_task: asyncio.Task | None = None
        self._last_run: float = 0.0
        self._last_report: SandboxReport | None = None

    async def start(self) -> None:
        """Start the doomsday sandbox."""
        self._running = True
        if settings.doomsday_run_on_physics_alert:
            self._bus.subscribe(EventType.PHYSICS_UPDATE, self._on_physics)
        self._periodic_task = asyncio.create_task(self._periodic_loop())
        logger.info(
            "DoomsdaySandbox started (interval=%ds, auto_protect=%s, threshold=%.2f)",
            settings.doomsday_interval_sec,
            settings.doomsday_auto_protect,
            settings.doomsday_survival_threshold,
        )

    async def stop(self) -> None:
        """Stop the doomsday sandbox."""
        self._running = False
        if self._periodic_task:
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass
        if settings.doomsday_run_on_physics_alert:
            self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._on_physics)
        logger.info("DoomsdaySandbox stopped")

    def get_last_report(self) -> SandboxReport | None:
        """Get the last simulation report."""
        return self._last_report

    # --- Event handlers ---

    async def _on_physics(self, event: Event) -> None:
        """Auto-run on dangerous physics signals."""
        physics = event.data.get("physics", {})
        ssd_mode = physics.get("ssd_mode", "normal")
        # Trigger on SSD laplace (high conviction moment) or silent (noise danger)
        if ssd_mode in ("laplace", "silent"):
            # Rate limit: max once per 5 minutes from physics triggers
            now = time.time()
            if now - self._last_run > 300:
                logger.info("DoomsdaySandbox triggered by physics alert (ssd_mode=%s)", ssd_mode)
                await self.run_all_scenarios()

    async def _periodic_loop(self) -> None:
        """Run stress tests on a schedule."""
        while self._running:
            try:
                await asyncio.sleep(settings.doomsday_interval_sec)
                if not self._running:
                    break
                await self.run_all_scenarios()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("DoomsdaySandbox periodic error")

    # --- Public API ---

    async def run_scenario(self, scenario_name: str) -> ScenarioResult:
        """Run a single catastrophe scenario."""
        snapshot = self._capture_snapshot()
        steps = self._generate_scenario(scenario_name)
        result = self._simulate(snapshot, steps, scenario_name)

        await self._bus.publish(
            Event(
                event_type=EventType.RISK_SIMULATION_RESULT,
                data={
                    "scenario": scenario_name,
                    "survival_score": result.survival_score,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "killswitch_triggered": result.killswitch_triggered,
                    "estimated_loss_pct": result.estimated_loss_pct,
                    "recommendations": result.recommendations,
                },
            )
        )
        return result

    async def run_all_scenarios(self) -> SandboxReport:
        """Run all scenarios and publish report."""
        self._last_run = time.time()
        results: dict[str, ScenarioResult] = {}

        for name in self.SCENARIOS:
            try:
                results[name] = await self.run_scenario(name)
            except Exception:
                logger.exception("Scenario %s failed", name)

        if not results:
            report = SandboxReport(
                timestamp=time.time(),
                scenarios={},
                overall_survival_score=1.0,
                weakest_scenario="none",
                auto_protect_triggered=False,
            )
            self._last_report = report
            return report

        # Overall survival = average of all scenario scores
        scores = [r.survival_score for r in results.values()]
        overall = sum(scores) / len(scores)

        # Find weakest
        weakest = min(results, key=lambda k: results[k].survival_score)

        # Auto-protect
        auto_triggered = False
        if settings.doomsday_auto_protect and overall < settings.doomsday_survival_threshold:
            auto_triggered = True
            await self._trigger_preventive_brake(overall, weakest)

        report = SandboxReport(
            timestamp=time.time(),
            scenarios=results,
            overall_survival_score=overall,
            weakest_scenario=weakest,
            auto_protect_triggered=auto_triggered,
        )
        self._last_report = report

        logger.info(
            "DoomsdaySandbox complete: overall=%.2f, weakest=%s (%.2f), auto_protect=%s",
            overall,
            weakest,
            results[weakest].survival_score,
            auto_triggered,
        )
        return report

    # --- Simulation engine ---

    def _capture_snapshot(self) -> MarketSnapshot:
        """Capture current system state for simulation."""
        equity = self._accounting.get_equity()
        positions = self._accounting.get_positions()
        _, dd_pct = self._accounting.get_drawdown(equity)

        pos_dicts = []
        for p in positions:
            d = p.model_dump() if hasattr(p, "model_dump") else p.dict()
            pos_dicts.append(d)

        return MarketSnapshot(
            timestamp=time.time(),
            equity=equity,
            positions=pos_dicts,
            open_orders=0,
            peak_equity=max(equity, settings.initial_capital),
            daily_high_equity=equity,
            current_drawdown_pct=dd_pct,
            initial_capital=settings.initial_capital,
        )

    def _generate_scenario(self, name: str) -> list[SimStep]:
        """Generate a time-series of simulated market conditions."""
        steps: list[SimStep] = []

        if name == "flash_crash_10pct":
            # 10% drop over 60 seconds (1 sec steps)
            for i in range(60):
                mult = 1.0 - (0.10 * (i / 60))
                steps.append(SimStep(time_offset_sec=float(i), price_multiplier=mult))

        elif name == "flash_crash_30pct":
            # 30% drop over 120 seconds
            for i in range(120):
                mult = 1.0 - (0.30 * (i / 120))
                steps.append(SimStep(time_offset_sec=float(i), price_multiplier=mult))

        elif name == "volatility_spike_5x":
            # Vol ramps from 1x to 5x over 60 seconds, with oscillating price
            for i in range(120):
                vol_mult = 1.0 + 4.0 * min(1.0, i / 60)
                # Price oscillates with increasing amplitude
                import math
                price_osc = 1.0 + 0.02 * vol_mult * math.sin(i * 0.5)
                steps.append(SimStep(
                    time_offset_sec=float(i),
                    price_multiplier=price_osc,
                    volatility_multiplier=vol_mult,
                ))

        elif name == "spread_blowout_50x":
            # Spread widens from 1x to 50x over 60 seconds
            for i in range(60):
                spread_mult = 1.0 + 49.0 * (i / 60)
                # Price also drops slightly due to liquidity crisis
                price_mult = 1.0 - 0.05 * (i / 60)
                steps.append(SimStep(
                    time_offset_sec=float(i),
                    price_multiplier=price_mult,
                    spread_multiplier=spread_mult,
                ))

        elif name == "funding_shock_2pct":
            # Funding rate jumps to 2% instantly, price adjusts
            for i in range(60):
                # Funding pressure causes a 5% price drop over the period
                price_mult = 1.0 - 0.05 * (i / 60)
                steps.append(SimStep(
                    time_offset_sec=float(i),
                    price_multiplier=price_mult,
                    funding_rate_delta=0.02,
                ))

        elif name == "exchange_downtime":
            # API goes down for 120 seconds, then comes back
            for i in range(180):
                api_up = i < 10 or i > 130  # First 10s up, then down for 120s, then back
                # Price drifts -3% during downtime (can't manage positions)
                if 10 <= i <= 130:
                    price_mult = 1.0 - 0.03 * ((i - 10) / 120)
                else:
                    price_mult = 0.97 if i > 130 else 1.0
                steps.append(SimStep(
                    time_offset_sec=float(i),
                    price_multiplier=price_mult,
                    api_available=api_up,
                ))

        elif name == "cascade_liquidation":
            # Progressive drops with accelerating slippage
            for i in range(120):
                # Nonlinear drop: accelerates as liquidations cascade
                t = i / 120
                price_mult = 1.0 - 0.25 * (t ** 1.5)  # 25% drop, accelerating
                spread_mult = 1.0 + 20.0 * (t ** 2)  # Spread widens quadratically
                steps.append(SimStep(
                    time_offset_sec=float(i),
                    price_multiplier=price_mult,
                    spread_multiplier=spread_mult,
                ))

        return steps

    def _simulate(
        self,
        snapshot: MarketSnapshot,
        steps: list[SimStep],
        scenario_name: str,
    ) -> ScenarioResult:
        """Simulate system response to scenario using shadow state."""
        if not steps:
            return ScenarioResult(
                scenario_name=scenario_name,
                duration_sec=0,
                max_drawdown_pct=0.0,
                risk_state_transitions=[],
                killswitch_triggered=False,
                killswitch_trigger_time_sec=None,
                positions_at_risk=len(snapshot.positions),
                estimated_loss_usd=0.0,
                estimated_loss_pct=0.0,
                survival_score=1.0,
                final_equity=snapshot.equity,
                recommendations=[],
            )

        # Shadow state for RiskGovernor thresholds
        # Instead of creating real instances (which need complex deps),
        # we simulate the logic directly
        risk_state = "NORMAL"
        ks_triggered = False
        ks_trigger_time: float | None = None
        risk_transitions: list[tuple[float, str, str]] = []

        sim_equity = snapshot.equity
        max_dd = 0.0
        peak = snapshot.peak_equity

        for step in steps:
            t = step.time_offset_sec

            # Apply price shock to all positions
            unrealized = 0.0
            for pos in snapshot.positions:
                entry = pos.get("entry_price", 0.0)
                size = pos.get("size", 0.0)
                side = pos.get("side", "long")
                sim_price = entry * step.price_multiplier

                if side == "long":
                    unrealized += (sim_price - entry) * size
                else:
                    unrealized += (entry - sim_price) * size

            # Add spread cost (simulated as additional loss)
            if step.spread_multiplier > 1.0:
                spread_cost = snapshot.equity * 0.001 * (step.spread_multiplier - 1.0)
                unrealized -= spread_cost

            # Add funding cost
            if step.funding_rate_delta != 0.0:
                notional = sum(
                    p.get("entry_price", 0.0) * p.get("size", 0.0)
                    for p in snapshot.positions
                )
                funding_cost = abs(notional * step.funding_rate_delta)
                unrealized -= funding_cost

            sim_equity = snapshot.equity + unrealized

            # Track drawdown from peak
            if sim_equity > peak:
                peak = sim_equity
            dd_pct = ((peak - sim_equity) / peak * 100) if peak > 0 else 0.0
            max_dd = max(max_dd, dd_pct)

            # Simulate RiskGovernor thresholds
            old_state = risk_state
            if dd_pct >= 20.0:
                risk_state = "HARD_STOP"
            elif dd_pct >= 15.0:
                risk_state = "QUARANTINE"
            elif dd_pct >= 10.0:
                risk_state = "SOFT_BRAKE"

            if risk_state != old_state:
                risk_transitions.append((t, old_state, risk_state))

            # Simulate KillSwitch (30% from initial)
            if snapshot.initial_capital > 0:
                dd_from_initial = (
                    (snapshot.initial_capital - sim_equity) / snapshot.initial_capital * 100
                )
                if dd_from_initial >= settings.killswitch_drawdown_pct:
                    if not ks_triggered:
                        ks_triggered = True
                        ks_trigger_time = t

            # Stop simulation on HARD_STOP or KillSwitch
            if risk_state == "HARD_STOP" or ks_triggered:
                break

        # Calculate results
        final_loss_usd = snapshot.equity - sim_equity
        final_loss_pct = (final_loss_usd / snapshot.equity * 100) if snapshot.equity > 0 else 0.0

        survival = self._calculate_survival_score(
            max_drawdown_pct=max_dd,
            killswitch_triggered=ks_triggered,
            final_equity_pct=sim_equity / snapshot.equity if snapshot.equity > 0 else 0.0,
        )

        recommendations = self._generate_recommendations(
            scenario_name=scenario_name,
            max_dd=max_dd,
            risk_transitions=risk_transitions,
            survival_score=survival,
            ks_triggered=ks_triggered,
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            duration_sec=len(steps),
            max_drawdown_pct=max_dd,
            risk_state_transitions=risk_transitions,
            killswitch_triggered=ks_triggered,
            killswitch_trigger_time_sec=ks_trigger_time,
            positions_at_risk=len(snapshot.positions),
            estimated_loss_usd=max(0.0, final_loss_usd),
            estimated_loss_pct=max(0.0, final_loss_pct),
            survival_score=survival,
            final_equity=sim_equity,
            recommendations=recommendations,
        )

    # --- Scoring ---

    @staticmethod
    def _calculate_survival_score(
        max_drawdown_pct: float,
        killswitch_triggered: bool,
        final_equity_pct: float,
    ) -> float:
        """Calculate survival score (0.0-1.0).

        Components:
        - Drawdown severity: 0.5 weight
        - Killswitch avoidance: 0.3 weight
        - Final equity preservation: 0.2 weight
        """
        dd_score = max(0.0, 1.0 - (max_drawdown_pct / 20.0))
        ks_score = 0.0 if killswitch_triggered else 1.0
        equity_score = max(0.0, min(1.0, final_equity_pct))

        survival = (dd_score * 0.5) + (ks_score * 0.3) + (equity_score * 0.2)
        return max(0.0, min(1.0, survival))

    @staticmethod
    def _generate_recommendations(
        scenario_name: str,
        max_dd: float,
        risk_transitions: list[tuple[float, str, str]],
        survival_score: float,
        ks_triggered: bool,
    ) -> list[str]:
        """Generate actionable recommendations from scenario results."""
        recs: list[str] = []

        if ks_triggered:
            recs.append("CRITICAL: KillSwitch triggered. Reduce position sizes or add hedges.")

        if max_dd > 15.0:
            recs.append(f"High drawdown risk ({max_dd:.1f}%). Consider tighter stop-losses.")

        if "flash_crash" in scenario_name and survival_score < 0.5:
            recs.append("Vulnerable to flash crashes. Add circuit-breaker or reduce leverage.")

        if "spread_blowout" in scenario_name and survival_score < 0.6:
            recs.append("Liquidity risk detected. Reduce position sizes in thin markets.")

        if "exchange_downtime" in scenario_name and survival_score < 0.7:
            recs.append("Exchange downtime exposure high. Add trailing stops or hedge positions.")

        if "cascade" in scenario_name and survival_score < 0.5:
            recs.append("Cascade liquidation risk. Diversify across uncorrelated assets.")

        if "funding" in scenario_name and survival_score < 0.6:
            recs.append("Funding rate shock vulnerability. Monitor funding before taking positions.")

        if not recs and survival_score > 0.8:
            recs.append("System resilience is strong for this scenario.")

        return recs

    async def _trigger_preventive_brake(self, overall_score: float, weakest: str) -> None:
        """Trigger preventive SOFT_BRAKE via risk alert."""
        logger.warning(
            "DoomsdaySandbox AUTO-PROTECT: survival=%.2f < threshold=%.2f, weakest=%s",
            overall_score,
            settings.doomsday_survival_threshold,
            weakest,
        )
        await self._bus.publish(
            Event(
                event_type=EventType.RISK_ALERT,
                data={
                    "type": "DOOMSDAY_AUTO_PROTECT",
                    "overall_survival_score": overall_score,
                    "threshold": settings.doomsday_survival_threshold,
                    "weakest_scenario": weakest,
                    "recommendation": "SOFT_BRAKE",
                },
            )
        )
