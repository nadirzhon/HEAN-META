# symbiont_x -- Genetic Strategy Evolution

Experimental self-evolving trading organism that uses genetic algorithms to discover and optimize trading strategies. Not currently wired into the main trading flow.

## Architecture

`HEANSymbiontX` is the main orchestrator that coordinates a biological-metaphor architecture: a nervous system (WebSocket connectors, health sensors), a genome lab (genetic algorithm evolution), an immune system (circuit breakers, risk constitution, reflexes), a decision ledger (recording and replaying trades), a regime brain (feature extraction and regime classification), a capital allocator (genome-specific allocation), an adversarial twin (stress testing and survival scoring), and an execution kernel. The `EvolutionEngine` manages populations of `StrategyGenome` objects, applying selection, mutation, and crossover operators. Genomes are evaluated via backtesting (through the `BacktestEngine`), and the fittest strategies survive. The system has its own test suite but communicates with the main HEAN system through `SymbiontXBridge` and the `GenomeDirector` in the archon package.

## Sub-packages

### genome_lab/
- `EvolutionEngine` -- Manages populations (default 50), applies tournament selection, elite preservation (5), mutation, and crossover. Tracks generation history and best genome.
- `MutationEngine` -- Applies random mutations to strategy genomes with configurable rate.
- `CrossoverEngine` -- Crosses two parent genomes to produce offspring.
- `StrategyGenome` / `create_random_genome()` -- Genome type definitions and random initialization.

### immune_system/
- `CircuitBreakerSystem` -- Protects evolved strategies from catastrophic failure.
- `RiskConstitution` -- Hard-coded risk rules that evolved strategies cannot violate.
- `ReflexSystem` -- Fast, instinctive responses to danger (e.g., rapid drawdown).

### decision_ledger/
- `DecisionLedger` -- Records every trading decision for post-mortem analysis.
- `DecisionReplay` -- Replays historical decisions for evaluation.
- `DecisionAnalysis` -- Analyzes decision patterns and outcomes.

### backtesting/
- `BacktestEngine` -- Backtests genome-parameterized strategies on historical data.
- `Indicators` -- Technical indicators for backtesting.

### capital_allocator/
- `CapitalAllocator` -- Allocates capital across evolved genomes based on fitness.
- `Portfolio` -- Portfolio tracking for evolved strategies.
- `PortfolioRebalancer` -- Rebalances capital between genomes.

### adversarial_twin/
- `StressTestSuite` -- Runs stress tests against evolved strategies.
- `SurvivalScoreCalculator` -- Calculates survival scores for genomes.
- `TestWorlds` -- Simulated market environments for stress testing.

### nervous_system/
- `BybitWSConnector` -- WebSocket connector for market data (independent of main system).
- `HealthSensorArray` -- Monitors health of evolved strategies.
- `EventEnvelope` -- Event wrapper for internal communication.

### regime_brain/
- `RegimeClassifier` -- Classifies market regime for evolved strategies.
- `FeatureExtractor` -- Extracts market features for regime classification.

### Other
- `HEANSymbiontX` (`symbiont.py`) -- Main orchestrator class.
- `SymbiontXBridge` (`bridge.py`) -- Bridge between Symbiont X and main HEAN trading system.
- `KPISystem` (`kpi_system.py`) -- Key performance indicator tracking for evolved strategies.
- `ExecutionKernel` (`execution_kernel/executor.py`) -- Execution layer for evolved strategies.

## Events

Symbiont X operates mostly independently. It interacts with the main system through `SymbiontXBridge` and receives evolution directives from the ARCHON `GenomeDirector`.

## Configuration

Symbiont X is configured via its own config dict passed to `HEANSymbiontX.__init__()`. It is not integrated into the main `HEANSettings`.
