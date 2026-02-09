# HEAN SYMBIONT X - АРХИТЕКТУРА ЖИВОГО ОРГАНИЗМА

**Версия:** 1.0
**Дата:** 28 января 2026
**Статус:** 🧬 Production Architecture

---

## 🎯 ФИЛОСОФИЯ

**HEAN SYMBIONT X** - это не бот. Это **симбиотический организм** для рынка Bybit, который:

- **Видит** рынок через нервную систему (Market Nervous System)
- **Думает** о режимах (Market Regime Brain)
- **Эволюционирует** стратегии (Alpha Genome Lab)
- **Тестирует** через злого экзаменатора (Adversarial Twin)
- **Распределяет** капитал умно (Capital Allocator)
- **Исполняет** молниеносно (Execution Microkernel)
- **Защищает** от смерти (Immune System)
- **Помнит** каждое решение (Decision Ledger)

---

## 🧬 АНАТОМИЯ ОРГАНИЗМА

```
┌─────────────────────────────────────────────────────────────────┐
│                    HEAN SYMBIONT X ORGANISM                     │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  IMMUNE SYSTEM   │ ← Защита от смерти
                    │  (Kill Switch)   │
                    └────────┬─────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         MARKET NERVOUS SYSTEM (нервы рынка)              │  │
│  │  WS: trades, orderbook, candles, funding, OI            │  │
│  │  Health: lag, gaps, drift, spread spikes                │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │ Events                                   │
│                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      MARKET REGIME BRAIN (мозг режимов)                  │  │
│  │  Classifier: TREND/RANGE/HIGH-VOL/LOW-VOL/THIN/SHOCK     │  │
│  │  Output: regime_state + confidence                       │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │ Regime                                   │
│          ┌──────────┴──────────┐                              │
│          ▼                     ▼                              │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ ALPHA GENOME LAB │  │   CAPITAL ALLOCATOR              │  │
│  │ (мутации)        │  │   (распределение капитала)        │  │
│  │                  │  │                                   │  │
│  │ • Genome         │  │ • Survival score                  │  │
│  │ • Mutations      │  │ • Correlation matrix              │  │
│  │ • Crossover      │  │ • Regime-aware allocation        │  │
│  │ • Regime-split   │  │ • Dynamic limits                 │  │
│  └────────┬─────────┘  └──────────┬────────────────────────┘  │
│           │                       │                           │
│           ▼                       ▼                           │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │    ADVERSARIAL DIGITAL TWIN (злой экзаменатор)          │  │
│  │  • Replay World (история)                               │  │
│  │  • Paper World (testnet)                                │  │
│  │  • Micro-Real World (мини-лоты)                         │  │
│  │  • Adversarial tests (latency, slippage, thin, shocks)  │  │
│  └────────────────────┬────────────────────────────────────┘  │
│                       │ Approved Strategies                   │
│                       ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │       EXECUTION MICROKERNEL (Rust) - скорость           │  │
│  │  • Smart limit orders + re-placement                    │  │
│  │  • Partial fill tracking                                │  │
│  │  • Slippage control                                     │  │
│  │  • Cancel/Replace on price move                         │  │
│  └────────────────────┬────────────────────────────────────┘  │
│                       │ Fills + Positions                     │
│                       ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │       DECISION LEDGER (память организма)                │  │
│  │  Каждая сделка: snapshot + reason + genome + outcome    │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

                     ┌────────────────────┐
                     │  UI + TELEGRAM     │
                     │  Control Center    │
                     └────────────────────┘
```

---

## 📦 МОДУЛИ И ИХ РЕАЛИЗАЦИЯ

### 1️⃣ MARKET NERVOUS SYSTEM

**Файлы:**
- `src/hean/nervous_system/ws_connectors.py` - WebSocket подключения
- `src/hean/nervous_system/event_envelope.py` - Унифицированные события
- `src/hean/nervous_system/health_sensors.py` - Сенсоры здоровья данных

**Функционал:**
```python
class MarketNervousSystem:
    """Нервная система рынка - видит всё в реальном времени"""

    def __init__(self):
        self.ws_streams = {
            'trades': TradeStream(),
            'orderbook': OrderbookStream(),
            'candles': CandleStream(),
            'funding': FundingStream(),
            'positions': PositionStream(),
        }
        self.health_sensors = HealthSensorArray()

    async def start(self):
        """Запуск всех нервов"""
        await asyncio.gather(*[
            stream.connect() for stream in self.ws_streams.values()
        ])

    def get_health_status(self) -> HealthStatus:
        """Проверка здоровья нервной системы"""
        return HealthStatus(
            data_lag_ms=self.health_sensors.check_lag(),
            gaps_detected=self.health_sensors.check_gaps(),
            spread_normal=self.health_sensors.check_spread(),
            quality_score=self.health_sensors.overall_quality()
        )
```

**Выход:** Поток `EventEnvelope` объектов для всей системы

---

### 2️⃣ MARKET REGIME BRAIN

**Файлы:**
- `src/hean/regime_brain/classifier.py` - Классификатор режимов
- `src/hean/regime_brain/features.py` - Фичи для классификации
- `src/hean/regime_brain/models.py` - ML модели

**Режимы:**
```python
class MarketRegime(Enum):
    TREND_UP = "trend_up"           # Восходящий тренд
    TREND_DOWN = "trend_down"       # Нисходящий тренд
    RANGE_TIGHT = "range_tight"     # Узкий рендж
    RANGE_WIDE = "range_wide"       # Широкий рендж
    HIGH_VOL = "high_vol"           # Высокая волатильность
    LOW_VOL = "low_vol"             # Низкая волатильность
    THIN_LIQUIDITY = "thin"         # Тонкая ликвидность
    NEWS_SHOCK = "shock"            # Новостной шок

class RegimeBrain:
    """Мозг режимов - классифицирует состояние рынка"""

    def classify_regime(self, market_snapshot) -> RegimeState:
        """
        Классифицирует текущий режим рынка

        Returns:
            RegimeState(
                primary_regime=MarketRegime.TREND_UP,
                confidence=0.87,
                probabilities={
                    MarketRegime.TREND_UP: 0.87,
                    MarketRegime.HIGH_VOL: 0.45,
                    MarketRegime.RANGE_WIDE: 0.12,
                },
                strength=0.92  # Насколько сильно выражен режим
            )
        """
        features = self.extract_features(market_snapshot)
        return self.model.predict(features)
```

**Фичи для классификации:**
- Volatility (ATR, std, parkinson)
- Trend strength (ADX, slope)
- Range detection (BB width, Donchian)
- Volume profile
- Spread dynamics
- Order flow imbalance

---

### 3️⃣ ALPHA GENOME LAB

**Файлы:**
- `src/hean/genome_lab/genome.py` - Определение генома
- `src/hean/genome_lab/mutations.py` - Мутации
- `src/hean/genome_lab/crossover.py` - Скрещивание
- `src/hean/genome_lab/evolution.py` - Эволюционный движок

**Геном стратегии:**
```python
@dataclass
class StrategyGenome:
    """Геном стратегии - ДНК альфы"""

    # ENTRY GENES (гены входа)
    entry_signals: List[Signal]          # RSI, MA cross, OFI, etc
    entry_filters: List[Filter]          # Режим, волатильность, время
    entry_regime_fit: List[MarketRegime] # В каких режимах работает

    # EXIT GENES (гены выхода)
    take_profit: TakeProfitGene          # Fixed, trailing, ATR-based
    stop_loss: StopLossGene              # Fixed, ATR, time-based
    trailing_stop: TrailingStopGene      # Trailing, chandelier
    time_exit: TimeExitGene              # Max hold time

    # RISK GENES (гены риска)
    position_size: PositionSizeGene      # Fixed, Kelly, risk-based
    max_leverage: float                   # 1.0 - 3.0
    max_exposure: float                   # % portfolio

    # EXECUTION GENES (гены исполнения)
    order_type: OrderTypeGene            # LIMIT, MARKET, POST_ONLY
    aggression: float                     # 0.0 (passive) - 1.0 (aggressive)
    repost_strategy: RepostGene          # Как перезаказывать

    # APPLICATION GENES (гены применения)
    symbols: List[str]                    # BTC, ETH, SOL, etc
    regimes: List[MarketRegime]          # Где применять

    # METADATA
    generation: int                       # Поколение
    parents: Tuple[str, str]             # ID родителей
    birth_date: datetime
    mutations_history: List[Mutation]

class AlphaGenomeLab:
    """Лаборатория генома - фабрика альфы"""

    def mutate_strategy(self, genome: StrategyGenome) -> StrategyGenome:
        """Создаёт мутанта из родителя"""
        mutation_type = random.choice([
            'parameter_tweak',      # Изменение параметров (±10-30%)
            'signal_add_remove',    # Добавить/убрать сигнал
            'filter_change',        # Изменить фильтр
            'execution_mutation',   # Изменить исполнение (часто даёт деньги!)
            'regime_split',         # Разделить на версии по режимам
            'exit_mutation',        # Изменить выходы
        ])

        mutant = self.apply_mutation(genome, mutation_type)
        mutant.generation = genome.generation + 1
        mutant.parents = (genome.id, None)
        mutant.mutations_history.append(
            Mutation(type=mutation_type, timestamp=now())
        )

        return mutant

    def crossover(self, parent_a: StrategyGenome, parent_b: StrategyGenome) -> StrategyGenome:
        """Скрещивает два генома - создаёт гибрид"""
        child = StrategyGenome()

        # Mix genes from parents
        child.entry_signals = random.choice([parent_a, parent_b]).entry_signals
        child.exit_genes = random.choice([parent_a, parent_b]).take_profit
        child.execution_genes = parent_a.execution_genes  # From strongest parent

        child.generation = max(parent_a.generation, parent_b.generation) + 1
        child.parents = (parent_a.id, parent_b.id)

        return child

    async def night_evolution(self, n_mutations: int = 50):
        """Ночная эволюция - генерит мутантов пока вы спите"""

        # Get current champions
        champions = self.get_top_strategies(n=10)

        # Generate mutations
        mutants = []
        for _ in range(n_mutations):
            parent = random.choice(champions)
            mutant = self.mutate_strategy(parent)
            mutants.append(mutant)

        # Test all mutants
        results = await self.adversarial_twin.test_batch(mutants)

        # Select survivors
        survivors = [m for m, r in zip(mutants, results) if r.passed]

        # Promote new champion if found
        if survivors:
            best = max(survivors, key=lambda m: results[m].survival_score)
            if best.survival_score > current_champion.survival_score:
                self.promote_to_champion(best)
                self.notify("🏆 New CHAMPION born!")
```

---

### 4️⃣ ADVERSARIAL DIGITAL TWIN

**Файлы:**
- `src/hean/adversarial/replay_world.py` - Исторический реплей
- `src/hean/adversarial/paper_world.py` - Paper trading
- `src/hean/adversarial/micro_real.py` - Микро-реал
- `src/hean/adversarial/stress_tests.py` - Злые тесты

**Три мира тестирования:**
```python
class AdversarialDigitalTwin:
    """Злой экзаменатор - проверяет живучесть"""

    async def test_strategy(self, genome: StrategyGenome) -> TestResult:
        """Полный цикл тестирования стратегии"""

        # World 1: Replay (история)
        replay_result = await self.replay_world.test(
            genome=genome,
            period_days=90,
            adversarial=True  # Добавляем стресс-факторы
        )

        if not replay_result.passed:
            return TestResult(passed=False, reason="Failed replay")

        # World 2: Paper (testnet)
        paper_result = await self.paper_world.test(
            genome=genome,
            duration_days=7
        )

        if not paper_result.passed:
            return TestResult(passed=False, reason="Failed paper")

        # World 3: Micro-Real (микро-лоты)
        micro_result = await self.micro_real.test(
            genome=genome,
            max_position_usd=10,  # Только $10
            duration_days=3
        )

        return TestResult(
            passed=micro_result.passed,
            survival_score=micro_result.survival_score,
            sharpe=micro_result.sharpe,
            max_dd=micro_result.max_dd,
            execution_edge=micro_result.execution_edge
        )

    def apply_adversarial_conditions(self, market_data):
        """Добавляет стресс-факторы для проверки робастности"""

        # 1. Latency injection (задержка)
        market_data = self.inject_latency(
            market_data,
            mean_ms=50,
            std_ms=100
        )

        # 2. Slippage amplification
        market_data = self.amplify_slippage(
            market_data,
            multiplier=2.0
        )

        # 3. Thin liquidity periods
        market_data = self.create_thin_periods(
            market_data,
            frequency=0.2  # 20% времени
        )

        # 4. News shocks simulation
        market_data = self.inject_shocks(
            market_data,
            n_shocks=5,
            magnitude=3.0  # 3 sigma moves
        )

        return market_data
```

**Критерии прохождения:**
- `survival_score > 0.7` (живучесть)
- `sharpe > 1.5` (риск-доходность)
- `max_drawdown < 15%` (просадка)
- `execution_edge > 0` (исполнение даёт деньги)
- `no_critical_errors` (без критических ошибок)

---

### 5️⃣ CAPITAL ALLOCATOR

**Файлы:**
- `src/hean/capital_allocator/allocator.py` - Аллокатор
- `src/hean/capital_allocator/survival_score.py` - Скоринг
- `src/hean/capital_allocator/correlation.py` - Корреляции

**Распределение капитала:**
```python
class CapitalAllocator:
    """Распределитель капитала - мозг фонда"""

    def allocate(self, strategies: List[Strategy], total_capital: float) -> Dict[Strategy, float]:
        """
        Распределяет капитал между стратегиями

        Критерии:
        1. Survival score (главный)
        2. Корреляция (диверсификация)
        3. Режим рынка (fit с текущим режимом)
        4. Track record (история)
        5. Комиссии/слиппедж (эффективность исполнения)
        """

        # 1. Calculate survival scores
        scores = {s: self.calculate_survival_score(s) for s in strategies}

        # 2. Build correlation matrix
        corr_matrix = self.calculate_correlation_matrix(strategies)

        # 3. Get current market regime
        regime = self.regime_brain.get_current_regime()

        # 4. Optimize allocation
        allocation = self.optimize_allocation(
            strategies=strategies,
            scores=scores,
            correlations=corr_matrix,
            regime=regime,
            total_capital=total_capital,
            constraints=[
                MaxPerStrategy(0.3),      # Макс 30% на стратегию
                MaxPerSymbol(0.25),        # Макс 25% на символ
                MaxPerRegime(0.4),         # Макс 40% на режим
                MinDiversification(5),     # Мин 5 стратегий
                ReserveBuffer(0.2),        # 20% резерв
            ]
        )

        return allocation

    def calculate_survival_score(self, strategy: Strategy) -> float:
        """
        Survival Score - главная метрика живучести

        Формула:
        survival_score = (
            0.3 * sharpe_component +
            0.2 * drawdown_component +
            0.2 * consistency_component +
            0.15 * execution_component +
            0.15 * regime_adaptation_component
        )
        """

        sharpe = normalize(strategy.sharpe_ratio, min=0, max=3)
        drawdown = 1 - normalize(strategy.max_drawdown, min=0, max=0.25)
        consistency = strategy.win_rate * strategy.profit_factor
        execution = strategy.execution_edge / strategy.theoretical_edge
        regime_adaptation = strategy.regime_fit_score

        return (
            0.3 * sharpe +
            0.2 * drawdown +
            0.2 * consistency +
            0.15 * execution +
            0.15 * regime_adaptation
        )
```

---

### 6️⃣ EXECUTION MICROKERNEL (Rust)

**Файлы:**
- `rust_services/execution_kernel/src/lib.rs` - Ядро
- `rust_services/execution_kernel/src/order_manager.rs` - Управление ордерами
- `rust_services/execution_kernel/src/fill_tracker.rs` - Трекинг исполнения

**Почему Rust:**
- Минимальная задержка (<1ms)
- Memory safety (нет segfault в production)
- Zero-cost abstractions
- Concurrent execution

```rust
// rust_services/execution_kernel/src/order_manager.rs

pub struct ExecutionMicrokernel {
    bybit_client: BybitClient,
    order_tracker: OrderTracker,
    fill_tracker: FillTracker,
    slippage_monitor: SlippageMonitor,
}

impl ExecutionMicrokernel {
    /// Умная постановка лимитного ордера
    pub async fn place_smart_limit(
        &mut self,
        signal: Signal,
        genome: &StrategyGenome,
    ) -> Result<OrderId> {
        // 1. Calculate optimal price
        let optimal_price = self.calculate_optimal_price(
            signal.side,
            genome.execution.aggression,
        );

        // 2. Place order
        let order_id = self.bybit_client.place_limit_order(
            symbol: &signal.symbol,
            side: signal.side,
            qty: signal.quantity,
            price: optimal_price,
            post_only: genome.execution.post_only,
        ).await?;

        // 3. Start monitoring
        self.order_tracker.track(order_id, signal);

        // 4. Auto re-placement if price moves away
        tokio::spawn(async move {
            self.monitor_and_repost(order_id, genome.execution.repost_strategy).await;
        });

        Ok(order_id)
    }

    /// Мониторинг и перезаказ
    async fn monitor_and_repost(
        &mut self,
        order_id: OrderId,
        repost_strategy: RepostStrategy,
    ) {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;

            let order_status = self.bybit_client.query_order(order_id).await;

            match order_status {
                OrderStatus::Filled => break,
                OrderStatus::Cancelled => break,
                OrderStatus::PartiallyFilled(filled_qty) => {
                    // Отслеживаем частичное исполнение
                    self.fill_tracker.record_partial_fill(order_id, filled_qty);
                }
                OrderStatus::Open => {
                    // Проверяем - не ушла ли цена
                    if self.should_repost(order_id, repost_strategy) {
                        self.cancel_and_repost(order_id).await?;
                    }
                }
            }
        }
    }

    /// Контроль проскальзывания
    fn check_slippage(&self, expected_price: f64, filled_price: f64) -> SlippageResult {
        let slippage_bps = ((filled_price - expected_price) / expected_price).abs() * 10000.0;

        if slippage_bps > self.config.max_slippage_bps {
            // Превышен лимит проскальзывания
            SlippageResult::Exceeded {
                actual_bps: slippage_bps,
                limit_bps: self.config.max_slippage_bps,
                action: SlippageAction::EnterSafeMode,
            }
        } else {
            SlippageResult::Ok(slippage_bps)
        }
    }
}
```

---

### 7️⃣ RISK CONSTITUTION + IMMUNE SYSTEM

**Файлы:**
- `src/hean/immune_system/constitution.py` - Конституция
- `src/hean/immune_system/reflexes.py` - Рефлексы
- `src/hean/immune_system/safe_mode.py` - Безопасный режим

**Конституция риска (неизменяемые законы):**
```python
class RiskConstitution:
    """Конституция риска - законы физики организма"""

    # ARTICLE I: Capital Preservation
    MAX_DAILY_LOSS_PCT = 3.0          # 3% максимум потерь в день
    MAX_DRAWDOWN_PCT = 15.0           # 15% максимальная просадка
    MAX_LEVERAGE = 3.0                # 3x максимальное плечо
    MAX_POSITION_SIZE_PCT = 10.0      # 10% портфеля на позицию

    # ARTICLE II: Trading Limits
    MAX_TRADES_PER_HOUR = 100         # Анти-overtrading
    MAX_COMMISSION_PCT = 0.5          # 0.5% комиссий от capital
    MAX_SLIPPAGE_BPS = 50             # 0.5% проскальзывание

    # ARTICLE III: Exposure Limits
    MAX_CORRELATION = 0.7             # Корреляция позиций
    MIN_LIQUIDITY_BUFFER_PCT = 20.0   # 20% резерв ликвидности
    MAX_SINGLE_SYMBOL_PCT = 25.0      # 25% на один символ

    # ARTICLE IV: Emergency Conditions
    SAFE_MODE_TRIGGERS = [
        "daily_loss > MAX_DAILY_LOSS",
        "drawdown > MAX_DRAWDOWN * 0.8",  # 80% от лимита
        "data_quality < 0.5",
        "exchange_lag > 1000ms",
        "consecutive_losses > 10",
    ]

    # ARTICLE V: Kill Switch
    KILL_SWITCH_TRIGGERS = [
        "drawdown >= MAX_DRAWDOWN",
        "exchange_down > 60s",
        "critical_error",
        "manual_trigger",
    ]

class ImmuneSystem:
    """Иммунная система - защита от смерти"""

    def __init__(self, constitution: RiskConstitution):
        self.constitution = constitution
        self.mode = TradingMode.NORMAL
        self.reflexes = {
            'safe_mode': SafeModeReflex(),
            'kill_switch': KillSwitchReflex(),
            'profit_lock': ProfitLockReflex(),
            'emergency_exit': EmergencyExitReflex(),
        }

    def check_health(self, state: SystemState) -> HealthCheck:
        """Проверка здоровья организма"""

        violations = []

        # Check daily loss
        if state.daily_loss_pct > self.constitution.MAX_DAILY_LOSS_PCT:
            violations.append(Violation(
                type="DAILY_LOSS_EXCEEDED",
                severity="CRITICAL",
                value=state.daily_loss_pct,
                limit=self.constitution.MAX_DAILY_LOSS_PCT
            ))

        # Check drawdown
        if state.drawdown_pct > self.constitution.MAX_DRAWDOWN_PCT:
            violations.append(Violation(
                type="DRAWDOWN_EXCEEDED",
                severity="CRITICAL",
                value=state.drawdown_pct,
                limit=self.constitution.MAX_DRAWDOWN_PCT
            ))

        # Check data quality
        if state.data_quality < 0.5:
            violations.append(Violation(
                type="DATA_DEGRADATION",
                severity="WARNING",
                value=state.data_quality,
                limit=0.5
            ))

        # Trigger reflexes if needed
        for violation in violations:
            if violation.severity == "CRITICAL":
                self.trigger_reflex(violation)

        return HealthCheck(
            healthy=len([v for v in violations if v.severity == "CRITICAL"]) == 0,
            violations=violations,
            mode=self.mode
        )

    def trigger_reflex(self, violation: Violation):
        """Триггер рефлексов при нарушениях"""

        if violation.type in ["DAILY_LOSS_EXCEEDED", "DRAWDOWN_EXCEEDED"]:
            # KILL SWITCH
            self.reflexes['kill_switch'].activate()
            self.mode = TradingMode.KILLED
            logger.critical(f"🔴 KILL SWITCH ACTIVATED: {violation.type}")

        elif violation.type == "DATA_DEGRADATION":
            # SAFE MODE
            self.reflexes['safe_mode'].activate()
            self.mode = TradingMode.SAFE
            logger.warning(f"⚠️ SAFE MODE ACTIVATED: {violation.type}")
```

**Safe Mode - режим обороны:**
```python
class SafeModeReflex:
    """Рефлекс безопасного режима"""

    def activate(self):
        """Переход в безопасный режим"""

        # 1. Reduce position sizes (50% от нормы)
        self.reduce_positions(multiplier=0.5)

        # 2. Reduce leverage (max 1.5x)
        self.reduce_leverage(max_leverage=1.5)

        # 3. Cancel all pending orders
        self.cancel_all_pending()

        # 4. Close high-risk positions
        self.close_high_risk_positions(
            criteria=["high_leverage", "high_correlation", "large_size"]
        )

        # 5. Switch to defensive strategies only
        self.enable_defensive_strategies_only()

        logger.warning("SAFE MODE: System in defensive mode")
```

---

### 8️⃣ DECISION LEDGER

**Файлы:**
- `src/hean/decision_ledger/ledger.py` - Основной ledger
- `src/hean/decision_ledger/replay.py` - Replay решений

**Память решений:**
```python
@dataclass
class DecisionRecord:
    """Запись о решении - каждое действие запоминается"""

    # Identifiers
    decision_id: str
    timestamp_ns: int
    strategy_id: str
    genome_version: str

    # Input State (снэпшот)
    market_snapshot: MarketSnapshot
    portfolio_snapshot: PortfolioSnapshot
    regime_state: RegimeState

    # Decision
    decision_type: DecisionType  # ENTRY, EXIT, REBALANCE, KILL
    signal: Signal
    reason: str  # Почему это решение было принято

    # Risk Checks
    constitution_checks: Dict[str, bool]
    survival_score: float

    # Execution
    order_id: Optional[str]
    expected_price: float
    expected_slippage: float

    # Outcome (заполняется после исполнения)
    actual_price: Optional[float]
    actual_slippage: Optional[float]
    realized_pnl: Optional[float]
    commission: Optional[float]
    execution_edge: Optional[float]  # Реальный PnL vs теоретический

    # Metadata
    regime_at_entry: MarketRegime
    regime_at_exit: Optional[MarketRegime]
    hold_duration_sec: Optional[int]

class DecisionLedger:
    """Ledger - память всех решений"""

    def __init__(self, db):
        self.db = db

    def record_decision(self, record: DecisionRecord):
        """Записывает решение в ledger"""
        self.db.insert('decisions', record)

    def explain_decision(self, decision_id: str) -> str:
        """Объясняет почему было принято это решение"""

        record = self.db.query(decision_id)

        explanation = f"""
        Decision: {record.decision_type} @ {record.timestamp}
        Strategy: {record.strategy_id} (genome v{record.genome_version})

        Market Context:
        - Regime: {record.regime_state.primary_regime} (confidence: {record.regime_state.confidence:.2f})
        - Price: {record.market_snapshot.price}
        - Volatility: {record.market_snapshot.volatility:.2%}

        Signal:
        - Type: {record.signal.type}
        - Strength: {record.signal.strength:.2f}
        - Reason: {record.reason}

        Risk Checks:
        {self.format_constitution_checks(record.constitution_checks)}

        Outcome:
        - PnL: ${record.realized_pnl:.2f}
        - Commission: ${record.commission:.2f}
        - Slippage: {record.actual_slippage:.4f} bps
        - Execution Edge: {record.execution_edge:.2%}
        """

        return explanation

    def replay_decision(self, decision_id: str):
        """Повторяет (replay) решение для анализа"""

        record = self.db.query(decision_id)

        # Воссоздаём состояние в момент решения
        state = SystemState(
            market=record.market_snapshot,
            portfolio=record.portfolio_snapshot,
            regime=record.regime_state,
        )

        # Запускаем стратегию с тем же геномом
        genome = self.genome_lab.load_genome(record.genome_version)
        strategy = Strategy(genome=genome)

        # Получаем решение
        decision = strategy.decide(state)

        # Сравниваем с оригинальным решением
        return ComparisonResult(
            original=record.decision,
            replayed=decision,
            match=record.decision == decision
        )
```

---

## 🎮 ПУЛЬТ УПРАВЛЕНИЯ (UI)

### Главный экран - "Пульт организма"

```
┌─────────────────────────────────────────────────────────────┐
│                 HEAN SYMBIONT X - CONTROL CENTER            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ EQUITY       │  │ PnL TODAY    │  │ MAX DD       │      │
│  │ $50,247      │  │ +$1,234      │  │ 8.4%         │      │
│  │ ────█████──  │  │ ▲ +2.5%      │  │ ████░░░░░    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ MARKET REGIME                                        │  │
│  │ 🟢 TREND_UP (confidence: 87%)                        │  │
│  │ Secondary: HIGH_VOL (45%)                           │  │
│  │ Strength: ████████░░ 92%                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ HEALTH STATUS                                        │  │
│  │ DATA:      🟢 Excellent  (lag: 12ms, quality: 98%)  │  │
│  │ EXCHANGE:  🟢 Stable     (uptime: 99.9%)             │  │
│  │ ENGINE:    🟢 Running    (strategies: 12 active)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    START    │ │    PAUSE    │ │   RESUME    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  SAFE MODE  │ │ LOCK PROFIT │ │ KILL SWITCH │ (HOLD)   │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ACTIVE STRATEGIES: 12                                │  │
│  │ CAPITAL DEPLOYED: $45,000 (90%)                      │  │
│  │ SURVIVORS RATE: 72% (last generation)               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Экран "Alpha Genome Lab"

```
┌─────────────────────────────────────────────────────────────┐
│                  ALPHA GENOME LAB - EVOLUTION               │
├─────────────────────────────────────────────────────────────┤
│  GENERATION: 47    ALIVE: 32    DEAD: 18    CHAMPION: #2   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🏆 CHAMPION: impulse_v12_gen47                            │
│     Survival Score: 0.89  Sharpe: 2.4  MaxDD: 6.2%        │
│     Regimes: TREND_UP, HIGH_VOL                           │
│     Age: 12 days   Trades: 347   Win Rate: 64%           │
│     [VIEW GENOME] [EXPLAIN] [MUTATE]                      │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│  TOP CANDIDATES:                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ #1 funding_arb_v8    Score: 0.85  [PROMOTE] [KILL] │  │
│  │ #2 basis_hybrid_v3   Score: 0.82  [PROMOTE] [KILL] │  │
│  │ #3 grid_mutant_v15   Score: 0.79  [PROMOTE] [KILL] │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  RECENT DEATHS:                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ ☠️ impulse_v11    Reason: Failed adversarial test  │  │
│  │ ☠️ scalp_v23      Reason: High commission rate     │  │
│  │ ☠️ momentum_v9    Reason: Regime mismatch          │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────┐ ┌──────────────────┐                 │
│  │ MUTATE BATCH    │ │ NIGHT EVOLUTION  │                 │
│  │ (generate 50)   │ │ (schedule)       │                 │
│  └─────────────────┘ └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 6 КЛЮЧЕВЫХ KPI (всегда видны)

```python
class SymbiontKPIs:
    """6 KPI организма - всегда на экране"""

    def calculate_kpis(self) -> KPISnapshot:
        return KPISnapshot(
            # 1. Survival Score (главный)
            survival_score=self.calculate_survival_score(),

            # 2. Execution Edge
            execution_edge=self.calculate_execution_edge(),

            # 3. Immunity Saves
            immunity_saves=self.count_immunity_saves(),

            # 4. Alpha Production
            alpha_production=self.calculate_alpha_production(),

            # 5. Truth Mode
            truth_mode_score=self.calculate_truth_score(),

            # 6. Autonomy Level
            autonomy_level=self.calculate_autonomy_level(),
        )

    def calculate_execution_edge(self) -> float:
        """
        Сколько денег дало исполнение vs "идеальная идея"

        execution_edge = (real_pnl - theoretical_pnl) / theoretical_pnl

        Positive edge = исполнение заработало больше чем сигнал
        """
        pass

    def count_immunity_saves(self) -> int:
        """
        Сколько раз иммунная система спасла депозит

        Считает срабатывания:
        - Safe mode activations
        - Circuit breakers
        - Kill switch (avoided losses)
        - Emergency exits
        """
        pass

    def calculate_alpha_production(self) -> AlphaProduction:
        """
        Производство альфы

        Returns:
            AlphaProduction(
                candidates_born=127,
                candidates_died=89,
                champions_promoted=3,
                avg_survival_days=8.4,
                evolution_rate=0.42  # Скорость эволюции
            )
        """
        pass

    def calculate_autonomy_level(self) -> float:
        """
        Уровень автономности

        autonomy = (auto_decisions / total_decisions) * decision_quality

        1.0 = полностью автономен и качественно
        0.0 = требует постоянного вмешательства
        """
        pass
```

---

## 🚀 ПЛАН РЕАЛИЗАЦИИ

### Phase 1: Нервная система (Week 1-2)
- ✅ WebSocket connectors (Bybit streams)
- ✅ Event envelope (унифицированные события)
- ✅ Health sensors (качество данных)

### Phase 2: Мозг режимов (Week 3-4)
- ✅ Feature extraction
- ✅ Regime classifier (ML model)
- ✅ Real-time classification

### Phase 3: Геном лаборатория (Week 5-6)
- ✅ Genome definition
- ✅ Mutation engine
- ✅ Crossover engine
- ✅ Evolution scheduler

### Phase 4: Злой экзаменатор (Week 7-8)
- ✅ Replay world
- ✅ Paper world
- ✅ Adversarial tests
- ✅ Micro-real world

### Phase 5: Execution kernel (Week 9-10)
- ✅ Rust microkernel
- ✅ Smart order placement
- ✅ Partial fill tracking
- ✅ Slippage control

### Phase 6: Иммунная система (Week 11-12)
- ✅ Risk constitution
- ✅ Reflexes (safe mode, kill switch)
- ✅ Health monitoring
- ✅ Emergency protocols

### Phase 7: UI & Control (Week 13-14)
- ✅ Control center UI
- ✅ Genome lab UI
- ✅ Telegram bot
- ✅ Mobile app (basic)

---

## 🎯 ИТОГОВАЯ АРХИТЕКТУРА

**HEAN SYMBIONT X** - это не бот. Это **живой организм**, который:

1. **Видит** рынок глазами (Market Nervous System)
2. **Понимает** режимы мозгом (Regime Brain)
3. **Эволюционирует** стратегии (Genome Lab)
4. **Тестирует** через боль (Adversarial Twin)
5. **Распределяет** капитал умно (Allocator)
6. **Исполняет** молниеносно (Microkernel - Rust)
7. **Защищается** рефлексами (Immune System)
8. **Помнит** каждый шаг (Decision Ledger)

**Это организм, который живёт и зарабатывает деньги пока вы спите.** 💰

---

_Архитектура готова к реализации. Начинаем код._
