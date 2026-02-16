from hean.core.microservices_bridge import (
    normalize_brain_analysis,
    normalize_physics_update,
    risk_payload_to_signal,
)


def test_normalize_physics_update_maps_core_fields() -> None:
    payload = {
        "symbol": "BTCUSDT",
        "temperature": 512.3,
        "entropy": 2.4,
        "phase": "WATER",
        "timestamp": 1234567890,
    }

    physics = normalize_physics_update(payload)

    assert physics["symbol"] == "BTCUSDT"
    assert physics["temperature"] == 512.3
    assert physics["temperature_regime"] == "WARM"
    assert physics["entropy"] == 2.4
    assert physics["entropy_state"] == "NORMAL"
    assert physics["phase"] == "water"
    assert physics["should_trade"] is True


def test_normalize_brain_analysis_uses_bias_when_sentiment_missing() -> None:
    payload = {
        "bias": "BULLISH",
        "confidence": 0.82,
        "summary": "market supports upside",
        "risk_level": "LOW",
    }

    analysis = normalize_brain_analysis(payload)

    assert analysis["sentiment"] == "bullish"
    assert analysis["confidence"] == 0.82
    assert analysis["summary"] == "market supports upside"
    assert analysis["risk_level"] == "LOW"


def test_risk_payload_to_signal_builds_bounded_buy_signal() -> None:
    payload = {
        "symbol": "ETHUSDT",
        "signal": "BUY",
        "confidence": 0.91,
        "price": 2500.0,
        "risk_approved": True,
        "risk_reason": "APPROVED",
    }

    signal = risk_payload_to_signal(payload)

    assert signal is not None
    assert signal.symbol == "ETHUSDT"
    assert signal.side == "buy"
    assert signal.entry_price == 2500.0
    assert signal.stop_loss is not None and signal.stop_loss < 2500.0
    assert signal.take_profit is not None and signal.take_profit > 2500.0
    assert signal.metadata["external_risk_approved"] is True


def test_risk_payload_to_signal_rejects_invalid_payloads() -> None:
    assert risk_payload_to_signal({"symbol": "BTCUSDT", "signal": "HOLD", "price": 1}) is None
    assert risk_payload_to_signal({"symbol": "BTCUSDT", "signal": "BUY"}) is None
