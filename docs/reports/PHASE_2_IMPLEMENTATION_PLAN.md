## Phase 2 Implementation Plan: Remaining 3 Improvements

**Status:** Detailed blueprint for improvements 4-6
**Estimated Implementation Time:** 8-12 hours
**Complexity:** Medium-High

---

## Improvement #4: Execution Cost Optimization

### Objective
Minimize execution costs through intelligent order type selection, TWAP for large orders, and slippage estimation.

### Components to Build

#### 4.1 Smart Order Type Selector

**File:** `/src/hean/execution/smart_order_selector.py`

**Key Logic:**
```python
class SmartOrderSelector:
    """Selects optimal order type based on edge, urgency, and costs."""

    def select_order_type(
        self,
        signal: Signal,
        current_bid: float,
        current_ask: float,
        urgency: float = 0.5,
    ) -> Dict[str, Any]:
        # Calculate costs
        maker_rebate_bps = 0.1  # Bybit: -0.01%
        taker_fee_bps = 5.5     # Bybit: 0.055%
        est_slippage_bps = self._estimate_slippage(signal.symbol, urgency)

        # Limit order net edge
        if signal.side == "buy":
            limit_price = current_bid * 1.0001  # Slightly inside spread
        else:
            limit_price = current_ask * 0.9999

        limit_net_edge_bps = (
            self._calculate_edge(signal, limit_price)
            + maker_rebate_bps  # Rebate is positive
        )

        # Market order net edge
        market_price = current_ask if signal.side == "buy" else current_bid
        market_net_edge_bps = (
            self._calculate_edge(signal, market_price)
            - taker_fee_bps
            - est_slippage_bps
        )

        # Decision: Use limit if edge is positive and urgency is low
        if urgency < 0.7 and limit_net_edge_bps > 2.0:
            return {
                "order_type": "limit",
                "price": limit_price,
                "expected_edge_bps": limit_net_edge_bps,
                "ttl_ms": 150 if urgency > 0.5 else 300,
            }
        elif market_net_edge_bps > 0.5:
            return {
                "order_type": "market",
                "price": None,
                "expected_edge_bps": market_net_edge_bps,
            }
        else:
            return {
                "order_type": "skip",
                "reason": "insufficient_edge",
            }

    def _estimate_slippage(self, symbol: str, urgency: float) -> float:
        # Historical slippage data + orderbook depth analysis
        # Higher urgency = more slippage
        pass
```

#### 4.2 TWAP Executor

**File:** `/src/hean/execution/twap_executor.py`

**Key Logic:**
```python
class TWAPExecutor:
    """Time-Weighted Average Price execution for large orders."""

    async def execute_twap(
        self,
        order_request: OrderRequest,
        duration_sec: int = 300,
        num_slices: int = 10,
        randomize: bool = True,
    ) -> List[Order]:
        """
        Split order into time-weighted slices.

        Args:
            order_request: Large order to split
            duration_sec: Total execution duration
            num_slices: Number of child orders
            randomize: Add timing jitter to avoid detection

        Returns:
            List of executed child orders
        """
        slice_size = order_request.size / num_slices
        interval_sec = duration_sec / num_slices

        child_orders = []

        for i in range(num_slices):
            # Randomize timing ±20%
            if randomize:
                jitter = interval_sec * 0.2 * (2 * random.random() - 1)
                sleep_time = interval_sec + jitter
            else:
                sleep_time = interval_sec

            await asyncio.sleep(sleep_time)

            # Create child order
            child_request = OrderRequest(
                signal_id=f"{order_request.signal_id}_twap_{i}",
                strategy_id=order_request.strategy_id,
                symbol=order_request.symbol,
                side=order_request.side,
                size=slice_size,
                price=None,  # Market order for simplicity
                order_type="market",
                metadata={
                    "twap_parent": order_request.signal_id,
                    "twap_slice": i,
                    "twap_total_slices": num_slices,
                },
            )

            # Execute slice
            order = await self._bybit_http.place_order(child_request)
            child_orders.append(order)

            logger.info(
                f"TWAP slice {i+1}/{num_slices} executed: "
                f"{slice_size} {order.symbol} @ {order.avg_fill_price:.2f}"
            )

        # Calculate VWAP
        total_notional = sum(o.size * o.avg_fill_price for o in child_orders)
        total_size = sum(o.size for o in child_orders)
        vwap = total_notional / total_size if total_size > 0 else 0.0

        logger.info(
            f"TWAP complete: {total_size} {order_request.symbol} @ VWAP {vwap:.2f}"
        )

        return child_orders

    def should_use_twap(self, order_request: OrderRequest, current_price: float) -> bool:
        """Determine if order is large enough to warrant TWAP.

        Args:
            order_request: Order to evaluate
            current_price: Current market price

        Returns:
            True if order should use TWAP
        """
        notional_usd = order_request.size * current_price
        threshold_usd = 500  # $500+ orders use TWAP

        return notional_usd >= threshold_usd
```

#### 4.3 Slippage Estimator

**File:** `/src/hean/execution/slippage_estimator.py`

**Key Logic:**
```python
class SlippageEstimator:
    """Estimates slippage based on orderbook depth and historical data."""

    def __init__(self):
        self._historical_slippage: Dict[str, deque] = {}

    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        size: float,
        orderbook: Dict[str, List[Tuple[float, float]]],
    ) -> float:
        """
        Estimate slippage in basis points.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            orderbook: Current orderbook {'bids': [...], 'asks': [...]}

        Returns:
            Estimated slippage in bps
        """
        # Method 1: Orderbook depth analysis
        levels = orderbook['asks'] if side == 'buy' else orderbook['bids']

        remaining_size = size
        total_cost = 0.0
        mid_price = self._calculate_mid_price(orderbook)

        for price, level_size in levels:
            filled_size = min(remaining_size, level_size)
            total_cost += filled_size * price
            remaining_size -= filled_size

            if remaining_size <= 0:
                break

        if remaining_size > 0:
            # Order too large for visible orderbook
            return 50.0  # 50 bps slippage estimate for large orders

        avg_price = total_cost / size
        slippage_pct = abs(avg_price - mid_price) / mid_price
        slippage_bps = slippage_pct * 10000

        # Method 2: Historical slippage adjustment
        hist_avg = self._get_historical_avg_slippage(symbol, side)
        if hist_avg > 0:
            # Blend orderbook estimate (70%) + historical (30%)
            slippage_bps = slippage_bps * 0.7 + hist_avg * 0.3

        return slippage_bps

    def record_actual_slippage(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
    ) -> None:
        """Record actual slippage for learning."""
        slippage_bps = abs(actual_price - expected_price) / expected_price * 10000

        if symbol not in self._historical_slippage:
            self._historical_slippage[symbol] = deque(maxlen=100)

        self._historical_slippage[symbol].append({
            "side": side,
            "slippage_bps": slippage_bps,
            "timestamp": datetime.utcnow(),
        })
```

### Integration Points

**In `/src/hean/execution/router_bybit_only.py`:**

```python
# Add after imports
from hean.execution.smart_order_selector import SmartOrderSelector
from hean.execution.twap_executor import TWAPExecutor
from hean.execution.slippage_estimator import SlippageEstimator

# In __init__
self._order_selector = SmartOrderSelector()
self._twap_executor = TWAPExecutor(self._bybit_http)
self._slippage_estimator = SlippageEstimator()

# In _handle_order_request
async def _handle_order_request(self, event: Event) -> None:
    order_request = event.data["order_request"]

    # Check if TWAP needed
    current_price = self._current_prices.get(order_request.symbol)
    if self._twap_executor.should_use_twap(order_request, current_price):
        logger.info(f"Using TWAP for large order: {order_request.symbol}")
        await self._twap_executor.execute_twap(order_request)
        return

    # Smart order type selection
    current_bid = self._current_bids.get(order_request.symbol)
    current_ask = self._current_asks.get(order_request.symbol)

    decision = self._order_selector.select_order_type(
        signal=order_request,
        current_bid=current_bid,
        current_ask=current_ask,
        urgency=order_request.metadata.get("urgency", 0.5),
    )

    if decision["order_type"] == "skip":
        logger.info(f"Order skipped: {decision['reason']}")
        return
    elif decision["order_type"] == "limit":
        order_request.order_type = "limit"
        order_request.price = decision["price"]
    # else: market order

    # Continue with existing routing logic
    await self._route_to_bybit(order_request)
```

---

## Improvement #5: Deeper Physics Integration

### Objective
Use market thermodynamics data (temperature, entropy, phase) to directly influence position sizing and signal confidence filtering.

### Components to Build

#### 5.1 Physics-Aware Position Sizer

**File:** `/src/hean/risk/physics_position_sizer.py`

**Key Logic:**
```python
class PhysicsAwarePositionSizer:
    """Position sizer that adapts to market thermodynamics."""

    def calculate_position_size(
        self,
        signal: Signal,
        equity: float,
        market_phase: str,
        temperature: float,
        entropy: float,
        volatility: float,
    ) -> float:
        """
        Calculate position size with physics adjustments.

        Args:
            signal: Trading signal
            equity: Current equity
            market_phase: Phase from physics (accumulation/markup/dist/markdown)
            temperature: Market temperature (0-1)
            entropy: Market entropy (0-1)
            volatility: Current volatility

        Returns:
            Position size in units
        """
        # Base size: 1% risk
        base_risk_pct = 0.01
        base_size_usd = equity * base_risk_pct

        # Factor 1: Phase alignment multiplier
        phase_mult = self._calculate_phase_multiplier(signal, market_phase)

        # Factor 2: Temperature adjustment
        temp_mult = self._calculate_temperature_multiplier(temperature)

        # Factor 3: Entropy penalty
        entropy_mult = self._calculate_entropy_multiplier(entropy)

        # Factor 4: Volatility scaling
        vol_mult = self._calculate_volatility_multiplier(volatility)

        # Combined multiplier
        total_mult = phase_mult * temp_mult * entropy_mult * vol_mult

        # Clamp to [0.25, 2.0]
        total_mult = np.clip(total_mult, 0.25, 2.0)

        adjusted_size_usd = base_size_usd * total_mult

        # Convert to position size
        position_size = adjusted_size_usd / signal.entry_price

        logger.debug(
            f"Physics sizing: base=${base_size_usd:.2f} "
            f"phase_mult={phase_mult:.2f} temp_mult={temp_mult:.2f} "
            f"entropy_mult={entropy_mult:.2f} vol_mult={vol_mult:.2f} "
            f"→ final=${adjusted_size_usd:.2f}"
        )

        return position_size

    def _calculate_phase_multiplier(self, signal: Signal, phase: str) -> float:
        """Calculate multiplier based on phase alignment.

        Buy signals:
        - Accumulation: 1.5x (buying at bottom)
        - Markup: 1.3x (trend following)
        - Distribution: 0.6x (late to party)
        - Markdown: 0.4x (counter-trend)

        Sell signals: inverse
        """
        if signal.side == "buy":
            phase_mults = {
                "accumulation": 1.5,
                "markup": 1.3,
                "distribution": 0.6,
                "markdown": 0.4,
            }
        else:  # sell
            phase_mults = {
                "accumulation": 0.4,
                "markup": 0.6,
                "distribution": 1.5,
                "markdown": 1.3,
            }

        return phase_mults.get(phase, 1.0)

    def _calculate_temperature_multiplier(self, temperature: float) -> float:
        """Higher temp = more chaos = smaller positions."""
        if temperature < 0.3:  # Cold, stable
            return 1.3
        elif temperature < 0.5:  # Moderate
            return 1.0
        elif temperature < 0.7:  # Warm
            return 0.8
        else:  # Hot, chaotic
            return 0.5

    def _calculate_entropy_multiplier(self, entropy: float) -> float:
        """Higher entropy = more disorder = smaller positions."""
        # Linear penalty from 1.0 at entropy=0 to 0.5 at entropy=1
        return 1.0 - (entropy * 0.5)

    def _calculate_volatility_multiplier(self, volatility: float) -> float:
        """Higher volatility = smaller positions (inverse relationship)."""
        # Reference: 2% vol → 1.0x, 4% vol → 0.5x
        reference_vol = 0.02
        if volatility <= reference_vol:
            return 1.0
        else:
            return reference_vol / volatility
```

#### 5.2 Physics-Based Signal Filter

**File:** `/src/hean/strategies/physics_signal_filter.py`

**Key Logic:**
```python
class PhysicsSignalFilter:
    """Filters signals based on physics state confidence."""

    def should_take_signal(
        self,
        signal: Signal,
        physics_state: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Determine if signal should be taken based on physics.

        Args:
            signal: Trading signal to evaluate
            physics_state: Physics state dict (temp, entropy, phase, etc)

        Returns:
            (should_take, reason)
        """
        phase = physics_state.get("phase", "unknown")
        phase_confidence = physics_state.get("phase_confidence", 0.0)
        temperature = physics_state.get("temperature", 0.5)
        entropy = physics_state.get("entropy", 0.5)

        # Rule 1: Require minimum phase confidence
        if phase_confidence < 0.5:
            return False, "low_phase_confidence"

        # Rule 2: Block signals in high chaos (high temp + high entropy)
        if temperature > 0.75 and entropy > 0.75:
            return False, "extreme_chaos"

        # Rule 3: Phase alignment check
        if not self._is_phase_compatible(signal, phase):
            return False, f"phase_mismatch_{phase}"

        # Rule 4: Temperature-entropy stability window
        if temperature > 0.85 or entropy > 0.90:
            return False, "unstable_market"

        return True, "physics_aligned"

    def _is_phase_compatible(self, signal: Signal, phase: str) -> bool:
        """Check if signal direction is compatible with phase."""
        # Buy signals
        if signal.side == "buy":
            # Good: accumulation, markup
            # Bad: distribution, markdown
            return phase in ["accumulation", "markup", "unknown"]
        else:  # sell
            return phase in ["distribution", "markdown", "unknown"]
```

### Integration Points

**In signal chain (main.py or risk_governor.py):**

```python
from hean.risk.physics_position_sizer import PhysicsAwarePositionSizer
from hean.strategies.physics_signal_filter import PhysicsSignalFilter

# Initialize
self._physics_sizer = PhysicsAwarePositionSizer()
self._physics_filter = PhysicsSignalFilter()

# In signal handling
async def _handle_signal(self, event: Event) -> None:
    signal = event.data["signal"]

    # Get physics state
    physics_state = self._physics_engine.get_state(signal.symbol)

    # Filter by physics
    should_take, reason = self._physics_filter.should_take_signal(
        signal, physics_state
    )

    if not should_take:
        logger.info(f"Signal blocked by physics: {reason}")
        return

    # Calculate physics-aware position size
    position_size = self._physics_sizer.calculate_position_size(
        signal=signal,
        equity=self._accounting.get_equity(),
        market_phase=physics_state["phase"],
        temperature=physics_state["temperature"],
        entropy=physics_state["entropy"],
        volatility=self._regime_detector.get_volatility(signal.symbol),
    )

    # Continue with order request...
```

---

## Improvement #6: Symbiont X Integration

### Objective
Wire genetic algorithm parameter optimization into main trading flow for continuous improvement of strategy parameters.

### Components to Build

#### 6.1 Symbiont X Bridge

**File:** `/src/hean/symbiont_x/bridge.py`

**Key Logic:**
```python
class SymbiontXBridge:
    """Bridges genetic algorithm with live trading system."""

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._optimization_tasks: Dict[str, asyncio.Task] = {}

    async def start_background_optimization(
        self,
        strategy_id: str,
        param_ranges: Dict[str, Tuple[float, float]],
        generations: int = 50,
        population_size: int = 20,
    ) -> None:
        """
        Start background optimization for a strategy.

        Args:
            strategy_id: Strategy to optimize
            param_ranges: Parameter ranges {'param_name': (min, max)}
            generations: Number of evolutionary generations
            population_size: Population size per generation
        """
        logger.info(
            f"Starting background optimization for {strategy_id} "
            f"(gen={generations}, pop={population_size})"
        )

        # Create optimization task
        task = asyncio.create_task(
            self._optimize_strategy_loop(
                strategy_id, param_ranges, generations, population_size
            )
        )
        self._optimization_tasks[strategy_id] = task

    async def _optimize_strategy_loop(
        self,
        strategy_id: str,
        param_ranges: Dict[str, Tuple[float, float]],
        generations: int,
        population_size: int,
    ) -> None:
        """Run genetic algorithm optimization loop."""
        from hean.symbiont_x.genome_lab import GenomeLab

        genome_lab = GenomeLab()

        # Initialize population
        population = genome_lab.create_initial_population(
            param_ranges, population_size
        )

        for gen in range(generations):
            # Evaluate fitness (backtest each genome)
            fitness_scores = await self._evaluate_population(
                strategy_id, population
            )

            # Select best performers
            best_genome = population[np.argmax(fitness_scores)]

            logger.info(
                f"Gen {gen+1}/{generations}: Best fitness={max(fitness_scores):.3f} "
                f"Params={best_genome}"
            )

            # Check if improvement threshold met
            if max(fitness_scores) > 1.5:  # 50% improvement
                logger.info(
                    f"✨ Found significant improvement for {strategy_id}! "
                    f"Applying optimized params..."
                )
                await self._apply_optimized_params(strategy_id, best_genome)
                break

            # Evolve population
            population = genome_lab.evolve_population(
                population, fitness_scores
            )

            await asyncio.sleep(60)  # Rate limit

    async def _evaluate_population(
        self,
        strategy_id: str,
        population: List[Dict[str, float]],
    ) -> List[float]:
        """Evaluate fitness of each genome via shadow backtesting."""
        from hean.backtest.event_sim import EventSimulator

        fitness_scores = []

        for genome in population:
            # Create shadow strategy with these params
            shadow_strategy = self._create_shadow_strategy(strategy_id, genome)

            # Run quick backtest (last 7 days of data)
            simulator = EventSimulator(duration_days=7)
            results = await simulator.run_backtest([shadow_strategy])

            # Fitness = Sharpe ratio
            fitness = results.get("sharpe_ratio", 0.0)
            fitness_scores.append(fitness)

        return fitness_scores

    async def _apply_optimized_params(
        self,
        strategy_id: str,
        params: Dict[str, float],
    ) -> None:
        """Apply optimized parameters to live strategy."""
        # Publish STRATEGY_PARAMS_UPDATED event
        await self._bus.publish(Event(
            event_type=EventType.STRATEGY_PARAMS_UPDATED,
            data={
                "strategy_id": strategy_id,
                "params": params,
                "source": "symbiont_x_optimization",
            }
        ))

        logger.info(
            f"✅ Applied optimized params to {strategy_id}: {params}"
        )
```

### Integration Points

**In main.py:**

```python
from hean.symbiont_x.bridge import SymbiontXBridge

# In __init__
self._symbiont_bridge: SymbiontXBridge | None = None

# In run()
self._symbiont_bridge = SymbiontXBridge(bus=self._bus)

# Start optimization for key strategies
await self._symbiont_bridge.start_background_optimization(
    strategy_id="impulse_engine",
    param_ranges={
        "max_spread_bps": (8.0, 20.0),
        "max_volatility_spike": (0.01, 0.05),
        "vol_expansion_ratio": (1.02, 1.15),
    },
    generations=50,
)
```

**In strategies (e.g., impulse_engine.py):**

```python
# Subscribe to param updates
self._bus.subscribe(EventType.STRATEGY_PARAMS_UPDATED, self._handle_param_update)

async def _handle_param_update(self, event: Event) -> None:
    """Handle optimized parameter updates from Symbiont X."""
    if event.data.get("strategy_id") == self.strategy_id:
        params = event.data.get("params", {})

        # Apply new params
        if "max_spread_bps" in params:
            self._max_spread_bps = params["max_spread_bps"]
        if "max_volatility_spike" in params:
            self._max_volatility_spike = params["max_volatility_spike"]
        # ...

        logger.info(f"Applied optimized params: {params}")
```

---

## Implementation Priority

**Recommended order:**
1. **Improvement #5** (Physics Integration) - Highest ROI, cleanest integration
2. **Improvement #4** (Execution Optimization) - Direct cost savings
3. **Improvement #6** (Symbiont X) - Long-term continuous improvement

---

## Testing Strategy

### For Each Improvement

1. **Unit tests:** Test core logic in isolation
2. **Integration tests:** Test with mock EventBus
3. **Backtest validation:** Compare vs baseline
4. **Testnet validation:** Run for 24-48 hours on Bybit testnet
5. **Performance metrics:** Measure overhead and ROI

---

## Expected Outcomes

**Improvement #4 (Execution):**
- 3-8 bps saved per trade (maker rebates + reduced slippage)
- 15-25% cost reduction on large orders (TWAP)

**Improvement #5 (Physics):**
- 20-30% improvement in win rate (phase-aligned entries)
- 10-15% reduction in drawdown (entropy-based sizing)

**Improvement #6 (Symbiont X):**
- 5-10% quarterly performance improvement (continuous optimization)
- Automatic adaptation to changing market conditions

**Combined Impact:**
- 30-50% overall performance improvement
- 50-70% reduction in avoidable losses
- Robust to regime changes

---

## Timeline Estimate

- **Improvement #4:** 3-4 hours
- **Improvement #5:** 2-3 hours
- **Improvement #6:** 3-5 hours
- **Testing & Integration:** 2-3 hours

**Total: 10-15 hours**

---

## Next Actions

1. Review this plan and prioritize improvements
2. Set up development branch for Phase 2
3. Implement improvements in priority order
4. Create comprehensive test suite
5. Run backtest comparison (with vs without)
6. Deploy to testnet for validation
7. Merge to main after validation passes

---

**Status:** Ready for implementation
**Dependencies:** Phase 1 components must be integrated and tested first
**Risk:** Low (all improvements are additive, can be individually disabled)
