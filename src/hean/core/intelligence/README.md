# core/intelligence -- Oracle Hybrid Signal Fusion and ML Engines

AI/ML signal fusion layer that combines multiple predictive sources (TCN, FinBERT, Ollama, Brain) into unified trading signals via weighted ensemble, with additional engines for causal inference, correlation analysis, and volatility prediction.

## Architecture

The `OracleIntegration` is the main entry point, subscribing to TICK events for TCN predictions, CONTEXT_UPDATE events for sentiment signals, and BRAIN_ANALYSIS events for AI analysis. It feeds data into the `OracleEngine`, which combines algorithmic fingerprinting (via optional C++ module) with the `TCPriceReversalPredictor` (PyTorch TCN). The `DynamicOracleWeighting` adapts source weights based on market regime, volatility, and recent accuracy. Signals are only published when combined confidence exceeds 0.6, and stale sources (>10 min) are excluded. Additional engines provide causal inference (Granger causality + transfer entropy), graph-based lead-lag detection, multimodal tensor fusion, and meta-learning.

## Key Classes

- `OracleIntegration` (`oracle_integration.py`) -- Fuses 4 signal sources: TCN (40%), FinBERT (20%), Ollama (20%), Brain (20%). Subscribes to TICK, POSITION_OPENED/CLOSED, and ORDER_BOOK_UPDATE events. Publishes ORACLE_PREDICTION events. Only emits when combined confidence > 0.6.
- `OracleEngine` (`oracle_engine.py`) -- Combines TCN price reversal predictions with optional C++ algorithmic fingerprinting. Caches predictions per symbol with multi-horizon price forecasts (500ms, 1s, 5s).
- `TCPriceReversalPredictor` (`tcn_predictor.py`) -- PyTorch Temporal Convolutional Network that processes the last 10,000 micro-ticks to predict probability of immediate price reversal. Triggers exit or position flip when probability > 85%.
- `DynamicOracleWeighting` (`dynamic_oracle.py`) -- Adapts source weights in real-time based on market phase (trend-following models get more weight in markup/markdown), volatility (predictive models get less weight in chaos), and recent accuracy tracking.
- `CorrelationEngine` (`correlation_engine.py`) -- Real-time Pearson correlation between assets. Identifies price gaps for pair trading: long the laggard, short the leader when correlation is high but prices diverge.
- `VolatilitySpikePredictor` (`volatility_predictor.py`) -- ONNX-based TFT model that predicts volatility spikes 1 second ahead. Triggers circuit breaker to clear maker orders when probability > 85%.
- `CausalInferenceEngine` (`causal_inference_engine.py`) -- Granger causality + transfer entropy analysis. Predicts Bybit moves by analyzing "pre-echoes" in global cross-asset orderflow. Produces `CausalRelationship` data with lag period and confidence.
- `GraphEngineWrapper` (`graph_engine.py`) -- Wrapper for C++ graph engine with real-time adjacency matrix and lead-lag detection. Falls back to Python `CorrelationEngine` when C++ module unavailable.
- `MultimodalSwarm` (`multimodal_swarm.py`) -- Processes price, sentiment, on-chain whale movements, and macro data as a unified tensor for holistic signal generation.
- `MetaLearningEngine` (`meta_learning_engine.py`) -- Recursive intelligence core that treats trading logic parameters as mutable weights, simulates failure scenarios, and can auto-patch code (when enabled).
- `MarketGenome` (`market_genome.py`) -- Market genome analysis for structural pattern recognition.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| TICK | Subscribes | Price data for TCN and correlation analysis |
| ORDER_BOOK_UPDATE | Subscribes | Orderbook for fingerprinting |
| CONTEXT_UPDATE | Subscribes | FinBERT/Ollama sentiment |
| BRAIN_ANALYSIS | Subscribes | Claude brain analysis signals |
| POSITION_OPENED | Subscribes | Track active positions for exit signals |
| POSITION_CLOSED | Subscribes | Remove closed positions |
| ORACLE_PREDICTION | Publishes | Fused oracle signal with direction and confidence |
| CAUSAL_SIGNAL | Publishes | Causal inference lead-lag signal |
| MARKET_GENOME_UPDATE | Publishes | Market structure pattern detection |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| TCN_MODEL_PATH | "" | Path to trained TCN weights (.pt) |
| ORACLE_DYNAMIC_WEIGHTING | false | Enable dynamic weight adaptation |
| OLLAMA_ENABLED | false | Enable Ollama LLM sentiment |
| OLLAMA_URL | http://localhost:11434 | Ollama server URL |
| OLLAMA_MODEL | llama3.2:3b | Ollama model name |
| OLLAMA_SENTIMENT_INTERVAL | 300 | Seconds between Ollama analyses |
