"""Aggregated Health Score - Single 0-100 metric for system health.

Combines multiple health indicators into one actionable score:
- 90-100: Excellent - All systems optimal
- 70-89: Good - Minor issues, trading normally
- 50-69: Degraded - Some concerns, reduced capacity
- 30-49: Warning - Significant issues, limited trading
- 0-29: Critical - Major problems, trading paused
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"  # 70-89
    DEGRADED = "degraded"  # 50-69
    WARNING = "warning"  # 30-49
    CRITICAL = "critical"  # 0-29


@dataclass
class HealthComponent:
    """Individual health component."""

    name: str
    score: float  # 0-100
    weight: float  # 0-1, contribution to overall score
    status: HealthStatus
    details: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def is_stale(self, max_age_seconds: int = 60) -> bool:
        """Check if this component data is stale."""
        age = datetime.utcnow() - self.last_updated
        return age > timedelta(seconds=max_age_seconds)


@dataclass
class HealthReport:
    """Complete health report."""

    overall_score: float  # 0-100
    status: HealthStatus
    components: dict[str, HealthComponent]
    timestamp: datetime
    recommendations: list[str]


def _score_to_status(score: float) -> HealthStatus:
    """Convert score to status."""
    if score >= 90:
        return HealthStatus.EXCELLENT
    elif score >= 70:
        return HealthStatus.GOOD
    elif score >= 50:
        return HealthStatus.DEGRADED
    elif score >= 30:
        return HealthStatus.WARNING
    else:
        return HealthStatus.CRITICAL


class HealthScoreCalculator:
    """Calculates aggregated health score from multiple components.

    Components and their weights:
    - Exchange connectivity (20%): API responsiveness, WebSocket health
    - Risk state (25%): Drawdown, killswitch status, position limits
    - Execution quality (15%): Slippage, latency, fill rates
    - Strategy health (20%): Signal generation, win rate, rejection rate
    - System resources (10%): Memory, CPU, disk
    - Data freshness (10%): Price staleness, heartbeat
    """

    def __init__(self) -> None:
        """Initialize health score calculator."""
        self._components: dict[str, HealthComponent] = {}

        # Component weights (must sum to 1.0)
        self._weights = {
            "exchange": 0.20,
            "risk": 0.25,
            "execution": 0.15,
            "strategy": 0.20,
            "system": 0.10,
            "data": 0.10,
        }

        # Initialize with unknown state
        for name, weight in self._weights.items():
            self._components[name] = HealthComponent(
                name=name,
                score=50.0,  # Unknown = degraded
                weight=weight,
                status=HealthStatus.DEGRADED,
                details="Not yet measured",
            )

    def update_exchange_health(
        self,
        ws_connected: bool,
        api_response_time_ms: float,
        api_error_rate: float,
    ) -> None:
        """Update exchange connectivity health.

        Args:
            ws_connected: WebSocket connection status
            api_response_time_ms: Average API response time
            api_error_rate: Error rate (0-1)
        """
        score = 100.0

        # WebSocket penalty
        if not ws_connected:
            score -= 40  # Major penalty for no WebSocket

        # API response time penalty (>500ms is concerning)
        if api_response_time_ms > 1000:
            score -= 30
        elif api_response_time_ms > 500:
            score -= 15
        elif api_response_time_ms > 200:
            score -= 5

        # Error rate penalty
        score -= api_error_rate * 100 * 0.3  # 30% weight for errors

        score = max(0, min(100, score))
        status = _score_to_status(score)

        details = (
            f"WS: {'connected' if ws_connected else 'disconnected'}, "
            f"API: {api_response_time_ms:.0f}ms, errors: {api_error_rate*100:.1f}%"
        )

        self._components["exchange"] = HealthComponent(
            name="exchange",
            score=score,
            weight=self._weights["exchange"],
            status=status,
            details=details,
        )

    def update_risk_health(
        self,
        drawdown_pct: float,
        killswitch_triggered: bool,
        risk_state: str,
        position_utilization: float,
    ) -> None:
        """Update risk management health.

        Args:
            drawdown_pct: Current drawdown percentage
            killswitch_triggered: Whether killswitch is active
            risk_state: Current risk governor state (NORMAL, SOFT_BRAKE, etc.)
            position_utilization: Position limit usage (0-1)
        """
        score = 100.0

        # Killswitch is critical
        if killswitch_triggered:
            score = 0
        else:
            # Drawdown penalty
            if drawdown_pct >= 15:
                score -= 50
            elif drawdown_pct >= 10:
                score -= 30
            elif drawdown_pct >= 5:
                score -= 15
            elif drawdown_pct >= 2:
                score -= 5

            # Risk state penalty
            risk_penalties = {
                "NORMAL": 0,
                "SOFT_BRAKE": 20,
                "QUARANTINE": 40,
                "HARD_STOP": 60,
            }
            score -= risk_penalties.get(risk_state, 30)

            # Position utilization (>80% is concerning)
            if position_utilization > 0.9:
                score -= 20
            elif position_utilization > 0.8:
                score -= 10

        score = max(0, min(100, score))
        status = _score_to_status(score)

        details = (
            f"DD: {drawdown_pct:.1f}%, state: {risk_state}, "
            f"pos util: {position_utilization*100:.0f}%"
        )
        if killswitch_triggered:
            details = "KILLSWITCH TRIGGERED"

        self._components["risk"] = HealthComponent(
            name="risk",
            score=score,
            weight=self._weights["risk"],
            status=status,
            details=details,
        )

    def update_execution_health(
        self,
        avg_slippage_bps: float,
        avg_latency_ms: float,
        fill_rate: float,
        rejection_rate: float,
    ) -> None:
        """Update execution quality health.

        Args:
            avg_slippage_bps: Average slippage in basis points
            avg_latency_ms: Average execution latency
            fill_rate: Order fill rate (0-1)
            rejection_rate: Order rejection rate (0-1)
        """
        score = 100.0

        # Slippage penalty (>10 bps is bad)
        if avg_slippage_bps > 20:
            score -= 30
        elif avg_slippage_bps > 10:
            score -= 15
        elif avg_slippage_bps > 5:
            score -= 5

        # Latency penalty
        if avg_latency_ms > 500:
            score -= 25
        elif avg_latency_ms > 200:
            score -= 10
        elif avg_latency_ms > 100:
            score -= 5

        # Fill rate bonus/penalty
        if fill_rate < 0.5:
            score -= 30
        elif fill_rate < 0.7:
            score -= 15
        elif fill_rate < 0.9:
            score -= 5

        # Rejection rate penalty
        score -= rejection_rate * 100 * 0.2

        score = max(0, min(100, score))
        status = _score_to_status(score)

        details = (
            f"Slip: {avg_slippage_bps:.1f}bps, lat: {avg_latency_ms:.0f}ms, "
            f"fill: {fill_rate*100:.0f}%, rej: {rejection_rate*100:.1f}%"
        )

        self._components["execution"] = HealthComponent(
            name="execution",
            score=score,
            weight=self._weights["execution"],
            status=status,
            details=details,
        )

    def update_strategy_health(
        self,
        signals_per_hour: float,
        win_rate: float,
        signal_rejection_rate: float,
        active_strategies: int,
    ) -> None:
        """Update strategy health.

        Args:
            signals_per_hour: Signal generation rate
            win_rate: Strategy win rate (0-1)
            signal_rejection_rate: Signal rejection rate (0-1)
            active_strategies: Number of active strategies
        """
        score = 100.0

        # No active strategies is bad
        if active_strategies == 0:
            score -= 40

        # Signal rate (too low = not finding opportunities)
        if signals_per_hour < 0.1:
            score -= 20
        elif signals_per_hour < 0.5:
            score -= 10

        # Win rate
        if win_rate < 0.3:
            score -= 30
        elif win_rate < 0.4:
            score -= 15
        elif win_rate < 0.5:
            score -= 5

        # High rejection rate indicates poor signal quality
        if signal_rejection_rate > 0.9:
            score -= 25
        elif signal_rejection_rate > 0.7:
            score -= 10

        score = max(0, min(100, score))
        status = _score_to_status(score)

        details = (
            f"Signals/hr: {signals_per_hour:.1f}, WR: {win_rate*100:.0f}%, "
            f"rej: {signal_rejection_rate*100:.0f}%, active: {active_strategies}"
        )

        self._components["strategy"] = HealthComponent(
            name="strategy",
            score=score,
            weight=self._weights["strategy"],
            status=status,
            details=details,
        )

    def update_system_health(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
    ) -> None:
        """Update system resources health.

        Args:
            cpu_percent: CPU usage (0-100)
            memory_percent: Memory usage (0-100)
            disk_percent: Disk usage (0-100)
        """
        score = 100.0

        # CPU penalty
        if cpu_percent > 90:
            score -= 30
        elif cpu_percent > 75:
            score -= 15
        elif cpu_percent > 60:
            score -= 5

        # Memory penalty
        if memory_percent > 90:
            score -= 30
        elif memory_percent > 80:
            score -= 15
        elif memory_percent > 70:
            score -= 5

        # Disk penalty
        if disk_percent > 95:
            score -= 25
        elif disk_percent > 90:
            score -= 10

        score = max(0, min(100, score))
        status = _score_to_status(score)

        details = f"CPU: {cpu_percent:.0f}%, Mem: {memory_percent:.0f}%, Disk: {disk_percent:.0f}%"

        self._components["system"] = HealthComponent(
            name="system",
            score=score,
            weight=self._weights["system"],
            status=status,
            details=details,
        )

    def update_data_health(
        self,
        last_tick_age_seconds: float,
        last_heartbeat_age_seconds: float,
        data_gaps_count: int,
    ) -> None:
        """Update data freshness health.

        Args:
            last_tick_age_seconds: Age of last market tick
            last_heartbeat_age_seconds: Age of last system heartbeat
            data_gaps_count: Number of data gaps detected recently
        """
        score = 100.0

        # Tick staleness
        if last_tick_age_seconds > 60:
            score -= 40
        elif last_tick_age_seconds > 30:
            score -= 20
        elif last_tick_age_seconds > 10:
            score -= 10

        # Heartbeat staleness
        if last_heartbeat_age_seconds > 120:
            score -= 30
        elif last_heartbeat_age_seconds > 60:
            score -= 15

        # Data gaps
        if data_gaps_count > 10:
            score -= 25
        elif data_gaps_count > 5:
            score -= 15
        elif data_gaps_count > 0:
            score -= 5

        score = max(0, min(100, score))
        status = _score_to_status(score)

        details = (
            f"Tick age: {last_tick_age_seconds:.0f}s, HB: {last_heartbeat_age_seconds:.0f}s, "
            f"gaps: {data_gaps_count}"
        )

        self._components["data"] = HealthComponent(
            name="data",
            score=score,
            weight=self._weights["data"],
            status=status,
            details=details,
        )

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall health score.

        Returns:
            Overall score 0-100
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for component in self._components.values():
            # Penalize stale data
            if component.is_stale():
                effective_score = component.score * 0.5  # 50% penalty for stale
            else:
                effective_score = component.score

            weighted_sum += effective_score * component.weight
            total_weight += component.weight

        if total_weight == 0:
            return 50.0  # Unknown

        return weighted_sum / total_weight

    def get_recommendations(self) -> list[str]:
        """Get actionable recommendations based on health.

        Returns:
            List of recommendations
        """
        recommendations = []

        for name, component in self._components.items():
            if component.status == HealthStatus.CRITICAL:
                recommendations.append(f"CRITICAL: {name} requires immediate attention - {component.details}")
            elif component.status == HealthStatus.WARNING:
                recommendations.append(f"WARNING: {name} needs investigation - {component.details}")
            elif component.is_stale():
                recommendations.append(f"STALE: {name} data is outdated, check monitoring")

        # Sort by severity
        recommendations.sort(key=lambda x: (
            0 if x.startswith("CRITICAL") else 1 if x.startswith("WARNING") else 2
        ))

        return recommendations

    def get_report(self) -> HealthReport:
        """Get complete health report.

        Returns:
            HealthReport with all data
        """
        overall_score = self.calculate_overall_score()

        return HealthReport(
            overall_score=overall_score,
            status=_score_to_status(overall_score),
            components=self._components.copy(),
            timestamp=datetime.utcnow(),
            recommendations=self.get_recommendations(),
        )

    def get_summary(self) -> dict[str, Any]:
        """Get health summary as dict for API responses.

        Returns:
            Summary dict
        """
        report = self.get_report()

        return {
            "overall_score": round(report.overall_score, 1),
            "status": report.status.value,
            "timestamp": report.timestamp.isoformat(),
            "components": {
                name: {
                    "score": round(comp.score, 1),
                    "status": comp.status.value,
                    "weight": comp.weight,
                    "details": comp.details,
                    "stale": comp.is_stale(),
                }
                for name, comp in report.components.items()
            },
            "recommendations": report.recommendations,
            "can_trade": report.status not in (HealthStatus.CRITICAL, HealthStatus.WARNING),
        }


# Global instance
health_score = HealthScoreCalculator()
