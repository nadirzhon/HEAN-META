"""Claude Brain Client - In-process AI market analysis.

Subscribes to CONTEXT_UPDATE events, periodically runs Claude analysis,
and stores thought history for the API.
"""

import asyncio
import json
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any

from hean.brain.models import BrainAnalysis, BrainThought, Force, TradingSignal
from hean.brain.snapshot import MarketSnapshotFormatter
from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class ClaudeBrainClient:
    """In-process brain that analyzes market state using Claude API."""

    def __init__(
        self,
        bus: EventBus,
        api_key: str = "",
        analysis_interval: int = 30,
    ) -> None:
        self._bus = bus
        self._api_key = api_key
        self._analysis_interval = analysis_interval

        # Anthropic client (optional)
        self._anthropic_client = None
        if self._api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.AsyncAnthropic(api_key=self._api_key)
                logger.info("Brain: Anthropic client initialized")
            except ImportError:
                logger.warning("Brain: anthropic package not installed, using rule-based fallback")

        # State
        self._latest_physics: dict[str, dict[str, Any]] = {}
        self._latest_participants: dict[str, dict[str, Any]] = {}
        self._latest_anomalies: list[dict[str, Any]] = []
        self._latest_temporal: dict[str, Any] = {}

        # History
        self._thoughts: deque[BrainThought] = deque(maxlen=200)
        self._analyses: deque[BrainAnalysis] = deque(maxlen=50)

        self._running = False
        self._analysis_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the brain client."""
        self._running = True
        self._bus.subscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info(
            f"Brain started (interval={self._analysis_interval}s, "
            f"claude={'enabled' if self._anthropic_client else 'rule-based'})"
        )

    async def stop(self) -> None:
        """Stop the brain client."""
        self._running = False
        self._bus.unsubscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("Brain stopped")

    async def _handle_context_update(self, event: Event) -> None:
        """Handle CONTEXT_UPDATE events from physics/participants."""
        data = event.data
        context_type = data.get("context_type", "")

        if context_type == "physics":
            physics = data.get("physics", {})
            symbol = physics.get("symbol", "BTCUSDT")
            self._latest_physics[symbol] = physics
        elif context_type == "participant_breakdown":
            symbol = data.get("symbol", "BTCUSDT")
            self._latest_participants[symbol] = data.get("breakdown", {})

    async def _analysis_loop(self) -> None:
        """Periodic analysis loop."""
        while self._running:
            try:
                await asyncio.sleep(self._analysis_interval)
                if not self._running:
                    break

                # Pick primary symbol for analysis
                symbol = "BTCUSDT"
                physics = self._latest_physics.get(symbol)
                participants = self._latest_participants.get(symbol)

                if not physics:
                    continue

                if self._anthropic_client:
                    analysis = await self._claude_analysis(physics, participants)
                else:
                    analysis = self._rule_based_analysis(physics, participants)

                if analysis:
                    self._analyses.append(analysis)
                    for thought in analysis.thoughts:
                        self._thoughts.append(thought)

                    # Publish BRAIN_ANALYSIS event for strategies to consume
                    await self._publish_brain_analysis(symbol, analysis)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Brain analysis error: {e}")
                await asyncio.sleep(5)

    async def _claude_analysis(
        self,
        physics: dict[str, Any],
        participants: dict[str, Any] | None,
    ) -> BrainAnalysis | None:
        """Run Claude API analysis."""
        prompt = MarketSnapshotFormatter.format(
            physics_state=physics,
            participants=participants,
            anomalies=self._latest_anomalies[:5] if self._latest_anomalies else None,
            temporal=self._latest_temporal or None,
        )

        try:
            message = await self._anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text

            # Try to parse JSON from response
            try:
                # Find JSON in response
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(response_text[start:end])
                    return self._parse_claude_response(parsed)
            except (json.JSONDecodeError, ValueError):
                pass

            # Fallback: create thought from raw text
            ts = datetime.utcnow().isoformat()
            thought = BrainThought(
                id=str(uuid.uuid4())[:8],
                timestamp=ts,
                stage="decision",
                content=response_text[:200],
                confidence=0.5,
            )
            return BrainAnalysis(
                timestamp=ts,
                thoughts=[thought],
                summary=response_text[:200],
            )

        except Exception as e:
            logger.warning(f"Claude API error: {e}")
            return self._rule_based_analysis(physics, participants)

    def _parse_claude_response(self, data: dict[str, Any]) -> BrainAnalysis:
        """Parse Claude's JSON response into BrainAnalysis."""
        ts = datetime.utcnow().isoformat()

        thoughts = []
        for t in data.get("thoughts", []):
            thoughts.append(BrainThought(
                id=str(uuid.uuid4())[:8],
                timestamp=ts,
                stage=t.get("stage", "decision"),
                content=t.get("content", ""),
                confidence=t.get("confidence"),
            ))

        signal = None
        sig_data = data.get("signal")
        if sig_data:
            signal = TradingSignal(
                symbol=sig_data.get("symbol", "BTCUSDT"),
                action=sig_data.get("action", "HOLD"),
                confidence=sig_data.get("confidence", 0.5),
                reason=sig_data.get("reason", ""),
            )

        return BrainAnalysis(
            timestamp=ts,
            thoughts=thoughts,
            signal=signal,
            summary=data.get("summary", ""),
        )

    def _rule_based_analysis(
        self,
        physics: dict[str, Any],
        participants: dict[str, Any] | None,
    ) -> BrainAnalysis:
        """Rule-based fallback analysis when Claude API is not available."""
        ts = datetime.utcnow().isoformat()
        thoughts: list[BrainThought] = []

        temperature = physics.get("temperature", 0)
        entropy = physics.get("entropy", 0)
        phase = physics.get("phase", "unknown")
        should_trade = physics.get("should_trade", False)
        trade_reason = physics.get("trade_reason", "")

        # Anomaly detection thought
        if temperature > 800:
            thoughts.append(BrainThought(
                id=str(uuid.uuid4())[:8], timestamp=ts, stage="anomaly",
                content=f"High temperature {temperature:.0f} - extreme volatility detected",
                confidence=0.9,
            ))
        elif temperature > 400:
            thoughts.append(BrainThought(
                id=str(uuid.uuid4())[:8], timestamp=ts, stage="anomaly",
                content=f"Elevated temperature {temperature:.0f} - increased activity",
                confidence=0.6,
            ))

        # Physics thought
        thoughts.append(BrainThought(
            id=str(uuid.uuid4())[:8], timestamp=ts, stage="physics",
            content=f"Phase: {phase.upper()}, T={temperature:.0f}, S={entropy:.2f}",
            confidence=0.85,
        ))

        # Participant X-Ray thought
        if participants:
            dominant = participants.get("dominant_player", "unknown")
            meta = participants.get("meta_signal", "neutral")
            thoughts.append(BrainThought(
                id=str(uuid.uuid4())[:8], timestamp=ts, stage="xray",
                content=f"Dominant: {dominant}. {meta}",
                confidence=0.7,
            ))

        # Decision thought
        action = "HOLD"
        confidence = 0.5
        reason = trade_reason or "No clear signal"

        if phase == "water" and temperature > 400 and entropy < 3.0 and should_trade:
            action = "BUY"
            confidence = 0.7
            reason = "WATER phase with rising temperature, moderate entropy"
        elif phase == "vapor" and temperature > 800:
            action = "HOLD"
            confidence = 0.8
            reason = "VAPOR phase, too volatile for new entries"
        elif phase == "ice" and entropy < 2.0:
            action = "HOLD"
            confidence = 0.6
            reason = "ICE phase, low volatility, wait for breakout"

        thoughts.append(BrainThought(
            id=str(uuid.uuid4())[:8], timestamp=ts, stage="decision",
            content=f"{action}: {reason}",
            confidence=confidence,
        ))

        signal = TradingSignal(
            symbol=physics.get("symbol", "BTCUSDT"),
            action=action,
            confidence=confidence,
            reason=reason,
        )

        return BrainAnalysis(
            timestamp=ts,
            thoughts=thoughts,
            signal=signal,
            summary=f"{phase.upper()} phase, T={temperature:.0f}. {reason}",
            market_regime=phase,
        )

    def get_thoughts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent thoughts for API."""
        thoughts = list(self._thoughts)[-limit:]
        return [t.model_dump() for t in thoughts]

    def get_latest_analysis(self) -> dict[str, Any] | None:
        """Get the latest analysis."""
        if self._analyses:
            return self._analyses[-1].to_dict()
        return None

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get analysis history."""
        analyses = list(self._analyses)[-limit:]
        return [a.to_dict() for a in analyses]

    async def _publish_brain_analysis(self, symbol: str, analysis: BrainAnalysis) -> None:
        """Publish brain analysis event to EventBus for strategies to consume."""
        # Extract key metrics from analysis
        sentiment = "neutral"
        confidence = 0.5
        forces = []

        if analysis.signal:
            # Map action to sentiment
            if analysis.signal.action in ("BUY", "LONG"):
                sentiment = "bullish"
            elif analysis.signal.action in ("SELL", "SHORT"):
                sentiment = "bearish"
            confidence = analysis.signal.confidence

        # Convert forces to dict
        if analysis.forces:
            forces = [
                {
                    "name": f.name,
                    "direction": f.direction,
                    "magnitude": f.magnitude,
                }
                for f in analysis.forces
            ]

        # Publish event
        await self._bus.publish(
            Event(
                event_type=EventType.BRAIN_ANALYSIS,
                data={
                    "symbol": symbol,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "forces": forces,
                    "market_regime": analysis.market_regime,
                    "summary": analysis.summary,
                    "signal_action": analysis.signal.action if analysis.signal else "HOLD",
                    "timestamp": analysis.timestamp,
                },
            )
        )

        logger.debug(
            f"Brain analysis published: {symbol} sentiment={sentiment} "
            f"confidence={confidence:.2f} regime={analysis.market_regime}"
        )
