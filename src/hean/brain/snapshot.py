"""Market snapshot formatter for Claude Brain prompts."""

from typing import Any


class MarketSnapshotFormatter:
    """Formats physics + participant + anomaly data into a Claude-ready prompt."""

    @staticmethod
    def format(
        physics_state: dict[str, Any] | None = None,
        participants: dict[str, Any] | None = None,
        anomalies: list[dict[str, Any]] | None = None,
        temporal: dict[str, Any] | None = None,
    ) -> str:
        parts = ["Analyze this market snapshot and provide trading insights:\n"]

        if physics_state:
            parts.append("## Physics State")
            parts.append(f"- Temperature: {physics_state.get('temperature', 0):.1f} ({physics_state.get('temperature_regime', 'N/A')})")
            parts.append(f"- Entropy: {physics_state.get('entropy', 0):.2f} ({physics_state.get('entropy_state', 'N/A')})")
            parts.append(f"- Phase: {physics_state.get('phase', 'unknown')}")
            parts.append(f"- Szilard Profit: ${physics_state.get('szilard_profit', 0):.2f}")
            parts.append(f"- Should Trade: {physics_state.get('should_trade', False)}")
            parts.append(f"- Trade Reason: {physics_state.get('trade_reason', 'N/A')}")
            parts.append("")

        if participants:
            parts.append("## Participant Breakdown")
            parts.append(f"- Dominant: {participants.get('dominant_player', 'unknown')}")
            parts.append(f"- MM Activity: {participants.get('mm_activity', 0):.1%}")
            parts.append(f"- Institutional Flow: ${participants.get('institutional_flow', 0):,.0f}")
            parts.append(f"- Retail Sentiment: {participants.get('retail_sentiment', 0.5):.1%}")
            parts.append(f"- Whale Activity: {participants.get('whale_activity', 0):.1%}")
            parts.append(f"- Meta Signal: {participants.get('meta_signal', 'N/A')}")
            parts.append("")

        if anomalies:
            parts.append("## Active Anomalies")
            for a in anomalies[:5]:
                parts.append(f"- [{a.get('type', 'unknown')}] {a.get('description', '')} (severity={a.get('severity', 0):.2f})")
            parts.append("")

        if temporal:
            levels = temporal.get("levels", {})
            if levels:
                parts.append("## Temporal Stack")
                for level_id in sorted(levels.keys(), reverse=True):
                    level = levels[level_id]
                    parts.append(f"- L{level_id} {level.get('name', '')}: {level.get('summary', '')}")
                parts.append("")

        parts.append("Respond with JSON: {\"thoughts\": [{\"stage\": \"anomaly|physics|xray|decision\", \"content\": \"...\", \"confidence\": 0.0-1.0}], \"signal\": {\"symbol\": \"BTCUSDT\", \"action\": \"BUY|SELL|HOLD\", \"confidence\": 0.0-1.0, \"reason\": \"...\"}, \"summary\": \"...\"}")

        return "\n".join(parts)
