# brain -- LLM-Powered Market Analysis

Periodic AI market analysis using LLM APIs. Analyzes market state, identifies forces, and produces trading signals that feed into the Oracle hybrid signal fusion.

## Architecture

The `ClaudeBrainClient` runs a periodic analysis loop (default every 60 seconds). It collects market state via `MarketSnapshotFormatter`, sends it to an LLM for analysis, and publishes `BRAIN_ANALYSIS` events with structured results. The client supports two providers with fallback: OpenRouter (Qwen3-Max-Thinking, preferred for cost efficiency) and Anthropic (Claude, fallback). Analysis results are stored as `BrainAnalysis` objects containing thoughts (staged reasoning), identified market forces, trading signals, and a summary. The brain also provides self-awareness context through `SelfAwarenessContext`, allowing the AI to analyze its own system's performance and code.

## Key Classes

- `ClaudeBrainClient` (`claude_client.py`) -- Main brain client. Subscribes to CONTEXT_UPDATE events for market state. Runs periodic analysis via OpenRouter (Qwen3-Max-Thinking) or Anthropic (Claude) APIs. Publishes BRAIN_ANALYSIS events. Stores thought history for API access.
- `BrainAnalysis` (`models.py`) -- Pydantic model for analysis results: timestamp, list of BrainThought objects, list of Force objects, optional TradingSignal, summary text, market regime.
- `BrainThought` (`models.py`) -- Single thought in the analysis process with id, timestamp, stage (anomaly/physics/xray/decision), content, and optional confidence.
- `Force` (`models.py`) -- Market force identified by the brain: name, direction (bullish/bearish/neutral), magnitude.
- `TradingSignal` (`models.py`) -- Trading signal from brain: symbol, action (BUY/SELL/HOLD/NEUTRAL), confidence, reason.
- `MarketSnapshotFormatter` (`snapshot.py`) -- Formats current market state into a structured prompt for the LLM.
- `SelfAwarenessContext` (`self_awareness_context.py`) -- Aggregates system performance metrics, strategy performance, and backtest results into a comprehensive snapshot for AI self-analysis.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| CONTEXT_UPDATE | Subscribes | Market state updates for analysis context |
| BRAIN_ANALYSIS | Publishes | Structured analysis result with forces and signal |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| BRAIN_ENABLED | true | Enable brain analysis |
| BRAIN_ANALYSIS_INTERVAL | 60 | Seconds between analyses |
| ANTHROPIC_API_KEY | "" | Anthropic API key (fallback provider) |
| OPENROUTER_API_KEY | "" | OpenRouter API key (preferred provider) |
