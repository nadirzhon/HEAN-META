"""
HEAN SYMBIONT X - Главный класс

Живой организм для автономной торговли
"""

import logging
import time
from pathlib import Path

from .adversarial_twin.stress_tests import StressTestSuite
from .adversarial_twin.survival_score import SurvivalScoreCalculator
from .capital_allocator.allocator import CapitalAllocator
from .capital_allocator.portfolio import Portfolio
from .capital_allocator.rebalancer import PortfolioRebalancer
from .decision_ledger.ledger import DecisionLedger
from .execution_kernel.executor import ExecutionKernel
from .genome_lab.evolution_engine import EvolutionEngine
from .immune_system.circuit_breakers import CircuitBreakerSystem
from .immune_system.constitution import RiskConstitution
from .immune_system.reflexes import ReflexSystem
from .kpi_system import KPISystem
from .nervous_system.health_sensors import HealthSensorArray

# Import all components
from .nervous_system.ws_connectors import BybitWSConnector
from .regime_brain.classifier import RegimeClassifier
from .regime_brain.features import FeatureExtractor

logger = logging.getLogger(__name__)


class HEANSymbiontX:
    """
    HEAN SYMBIONT X

    Самоэволюционирующий организм для автономной торговли на крипто-рынках
    """

    def __init__(
        self,
        config: dict,
        storage_path: str | None = None,
    ):
        self.config = config
        self.storage_path = Path(storage_path) if storage_path else Path("./symbiont_data")

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # === NERVOUS SYSTEM ===
        self.ws_connector = BybitWSConnector(
            symbols=config.get('symbols', ['BTCUSDT']),
            api_key=config.get('bybit_api_key'),
            api_secret=config.get('bybit_api_secret')
        )
        self.health_sensors = HealthSensorArray()

        # === REGIME BRAIN ===
        self.feature_extractors: dict[str, FeatureExtractor] = {}
        self.regime_classifiers: dict[str, RegimeClassifier] = {}

        for symbol in config.get('symbols', ['BTCUSDT']):
            self.feature_extractors[symbol] = FeatureExtractor(symbol)
            self.regime_classifiers[symbol] = RegimeClassifier(symbol)

        # === GENOME LAB ===
        self.evolution_engine = EvolutionEngine(
            population_size=config.get('population_size', 50),
            elite_size=config.get('elite_size', 5),
            mutation_rate=config.get('mutation_rate', 0.1),
            crossover_rate=config.get('crossover_rate', 0.3),
        )

        # === ADVERSARIAL TWIN ===
        self.stress_test_suite = StressTestSuite()
        self.survival_calculator = SurvivalScoreCalculator()

        # === CAPITAL ALLOCATOR ===
        self.portfolio = Portfolio(
            portfolio_id="main",
            name="SYMBIONT_X_Portfolio",
            total_capital=config.get('initial_capital', 10000),
        )

        self.capital_allocator = CapitalAllocator(
            allocation_method=config.get('allocation_method', 'survival_weighted'),
        )

        self.rebalancer = PortfolioRebalancer(
            allocator=self.capital_allocator,
            rebalance_interval_hours=config.get('rebalance_interval_hours', 24),
        )

        # === IMMUNE SYSTEM ===
        self.constitution = RiskConstitution(
            constitution_config=config.get('risk_constitution')
        )
        self.constitution.make_immutable()  # Lock the constitution

        self.reflex_system = ReflexSystem()
        self.reflex_system.register_default_triggers()

        self.circuit_breakers = CircuitBreakerSystem()
        self.circuit_breakers.register_default_breakers()

        # === DECISION LEDGER ===
        self.decision_ledger = DecisionLedger(
            storage_path=str(self.storage_path / "decisions.jsonl"),
            auto_persist=True,
        )

        # === EXECUTION KERNEL ===
        self.execution_kernel = ExecutionKernel(
            exchange_connector=self.ws_connector  # Will be proper exchange client
        )

        # === KPI SYSTEM ===
        self.kpi_system = KPISystem()

        # === STATE ===
        self.is_running = False
        self.start_time_ns: int | None = None
        self.manual_interventions = 0

        print("🧬 HEAN SYMBIONT X initialized")
        print(f"📁 Storage: {self.storage_path}")

    async def start(self):
        """Запускает SYMBIONT X"""

        if self.is_running:
            print("⚠️  SYMBIONT X already running")
            return

        print("🚀 Starting HEAN SYMBIONT X...")
        self.is_running = True
        self.start_time_ns = time.time_ns()

        # Initialize population
        print("🧬 Initializing strategy population...")
        self.evolution_engine.initialize_population(base_name="HEAN_Strategy")

        # Connect nervous system
        print("🧠 Connecting to market data...")
        await self.ws_connector.connect()

        # Register event handlers
        self.ws_connector.on_trade(self._on_trade_event)
        self.ws_connector.on_orderbook(self._on_orderbook_event)

        # Start main loop
        print("♥️  SYMBIONT X is ALIVE")
        self.update_kpis()

        # Display vital signs
        print(self.kpi_system.get_dashboard_view())

    async def _on_trade_event(self, event):
        """Обработка трейда"""
        # Process through health sensors
        self.health_sensors.process_event(event)

        # Extract features
        symbol = event.symbol
        if symbol in self.feature_extractors:
            self.feature_extractors[symbol].process_event(event)

    async def _on_orderbook_event(self, event):
        """Обработка обновления стакана"""
        self.health_sensors.process_event(event)

        symbol = event.symbol
        if symbol in self.feature_extractors:
            self.feature_extractors[symbol].process_event(event)

    async def evolve_generation(self):
        """Эволюционирует одно поколение стратегий"""

        print(f"\n🧬 Evolving generation {self.evolution_engine.generation_number + 1}...")

        # CRITICAL WARNING: Test worlds and stress tests are not yet implemented
        # Evolution will proceed with simulated/zero fitness scores
        # This means strategies are not actually validated before deployment
        logger.warning(
            "Evolution proceeding WITHOUT real testing - test worlds not implemented. "
            "Strategies will have zero survival scores and will not be promoted."
        )
        print("⚠️  WARNING: Test execution not implemented - strategies not validated")

        # Evolve
        self.evolution_engine.evolve_generation()

        print(f"✅ Generation {self.evolution_engine.generation_number} complete")
        print("⚠️  Note: Generation evolved without fitness testing")

        # Update KPIs
        self.update_kpis()

    def update_kpis(self):
        """Обновляет все KPI"""

        # 1. Survival Score
        portfolio_stats = self.portfolio.get_statistics()
        survival_score = 75.0  # Placeholder - would calculate from portfolio

        self.kpi_system.update_survival_score(
            score=survival_score,
            portfolio_stats=portfolio_stats
        )

        # 2. Execution Edge
        exec_stats = self.execution_kernel.get_statistics()
        self.kpi_system.update_execution_edge(
            avg_slippage_bps=3.5,  # Placeholder
            avg_latency_ms=exec_stats.get('avg_latency_ms', 0)
        )

        # 3. Immunity Saves
        reflex_stats = self.reflex_system.get_statistics()
        breaker_stats = self.circuit_breakers.get_statistics()

        self.kpi_system.update_immunity_saves(
            total_reflexes=reflex_stats['total_reflexes_triggered'],
            total_breakers=breaker_stats['total_trips']
        )

        # 4. Alpha Production
        self.kpi_system.update_alpha_production(
            total_pnl=portfolio_stats.get('total_pnl', 0),
            roi_pct=portfolio_stats.get('roi_pct', 0)
        )

        # 5. Truth Mode
        health_status = self.health_sensors.get_health_status()
        health_score = (health_status.overall_health_score / 5.0) * 100

        # Get regime confidence (average across symbols)
        regime_confidences = []
        for _symbol, classifier in self.regime_classifiers.items():
            history = classifier.get_regime_history(last_n=1)
            if history:
                regime_confidences.append(history[0]['confidence'] * 100)

        avg_regime_confidence = sum(regime_confidences) / len(regime_confidences) if regime_confidences else 0

        self.kpi_system.update_truth_mode(
            data_health_score=health_score,
            regime_confidence=avg_regime_confidence
        )

        # 6. Autonomy Level
        ledger_stats = self.decision_ledger.get_statistics()

        # Calculate decisions per hour
        if self.start_time_ns:
            runtime_hours = (time.time_ns() - self.start_time_ns) / (3600 * 1e9)
            decisions_per_hour = ledger_stats['total_decisions'] / runtime_hours if runtime_hours > 0 else 0
        else:
            decisions_per_hour = 0

        self.kpi_system.update_autonomy_level(
            strategies_active=portfolio_stats.get('active_strategies', 0),
            decisions_per_hour=int(decisions_per_hour),
            manual_interventions=self.manual_interventions
        )

    def get_vital_signs(self) -> str:
        """Возвращает vital signs dashboard"""
        return self.kpi_system.get_dashboard_view()

    def get_system_status(self) -> dict:
        """Возвращает полный статус системы"""

        return {
            'is_running': self.is_running,
            'uptime_seconds': (time.time_ns() - self.start_time_ns) / 1e9 if self.start_time_ns else 0,
            'kpis': self.kpi_system.to_dict(),
            'portfolio': self.portfolio.get_statistics(),
            'evolution': self.evolution_engine.get_statistics(),
            'immune_system': {
                'constitution': self.constitution.get_statistics(),
                'reflexes': self.reflex_system.get_statistics(),
                'circuit_breakers': self.circuit_breakers.get_statistics(),
            },
            'execution': self.execution_kernel.get_statistics(),
            'decision_ledger': self.decision_ledger.get_statistics(),
        }

    async def stop(self):
        """Останавливает SYMBIONT X"""

        if not self.is_running:
            return

        print("\n🛑 Stopping HEAN SYMBIONT X...")

        self.is_running = False

        # Disconnect nervous system
        # NOTE: WebSocket disconnect method not yet implemented
        # await self.ws_connector.disconnect()
        logger.warning("WebSocket disconnect not implemented - connection may remain open")

        # Persist decision ledger
        self.decision_ledger.persist_to_disk()

        # Save population
        population_path = self.storage_path / "population.json"
        self.evolution_engine.save_population(str(population_path))

        print("💤 SYMBIONT X stopped")

    def __str__(self) -> str:
        return f"HEAN SYMBIONT X (Generation {self.evolution_engine.generation_number})"
