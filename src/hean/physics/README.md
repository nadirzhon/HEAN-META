# physics -- Market Thermodynamics Engine

Applies thermodynamic analogies to market data: temperature measures kinetic energy, entropy measures volume distribution disorder, and phase detection identifies market regimes. The Szilard engine calculates maximum extractable profit from information theory.

## Architecture

The `PhysicsEngine` subscribes to TICK events and orchestrates all physics components. For each symbol, it maintains rolling price/volume buffers and calculates temperature, entropy, phase, and Szilard profit. It publishes `PHYSICS_UPDATE` events consumed by strategies (via SSD lens in BaseStrategy), the ContextAggregator, and the StrategyAllocator. The SSD (Singular Spectral Determinism) system adds entropy flow rate tracking and resonance detection. When entropy flow is strongly positive (diverging), the SSD mode is "silent" and strategies skip the tick. When price/volume/entropy vectors align (resonance), the mode is "laplace" and strategies can act with boosted confidence.

## Key Classes

- `PhysicsEngine` (`engine.py`) -- Main orchestrator. Subscribes to TICK events, calculates T/S/phase per symbol, publishes PHYSICS_UPDATE. Manages `PhysicsState` per symbol including SSD fields (entropy_flow, ssd_mode, resonance_strength).
- `MarketTemperature` (`temperature.py`) -- Calculates temperature as kinetic energy per sample: T = KE / N, where KE = sum((deltaP * V)^2). Classifies into HOT (>800), WARM (400-800), COLD (<400) regimes.
- `MarketEntropy` (`entropy.py`) -- Calculates Shannon entropy of normalized volume distribution: S = -sum(p_i * log(p_i)). Low entropy = compressed (breakout imminent), high entropy = equilibrium. Tracks SSD entropy flow rate (dH/dt) and EMA-smoothed flow.
- `PhaseDetector` (`phase_detector.py`) -- Detects market phase transitions using temperature and entropy: ICE (low T, low S -- consolidation), WATER (medium T, medium S -- trending), VAPOR (high T, high S -- chaos). ICE-to-WATER transitions offer maximum edge. Includes SSD resonance detection (cosine similarity of price/volume/entropy vectors) and mode classification (SILENT/NORMAL/LAPLACE).
- `SzilardEngine` (`szilard.py`) -- Calculates maximum extractable profit using Szilard's formula: MAX_PROFIT = T * I, where I = log2(1/p) is information in bits and p is event probability.
- `MarketAnomalyDetector` (`anomaly_detector.py`) -- Detects unusual market conditions: OI spikes, funding divergence, whale inflows, volume anomalies, price dislocations, liquidation cascades.
- `ParticipantClassifier` (`participant_classifier.py`) -- Classifies market participants (institutional vs retail) based on order characteristics.
- `TemporalStack` (`temporal_stack.py`) -- Multi-timeframe temporal analysis of physics states.
- `CrossMarketAnalyzer` (`cross_market.py`) -- Cross-market physics comparison.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| TICK | Subscribes | Price/volume data for physics calculations |
| PHYSICS_UPDATE | Publishes | Full physics state: temperature, entropy, phase, SSD mode, Szilard profit |

## Configuration

Physics uses no dedicated config flags. The engine is always active when TradingSystem starts. Window size defaults to 100 ticks.
