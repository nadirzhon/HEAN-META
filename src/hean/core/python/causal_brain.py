"""
CausalBrain.py

The Python "Brain" of the HEAN Absolute+ system.
Continuously learns from market patterns and sends Logic Mutation signals
to the C++ MetamorphicCore every 100ms.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import zmq


class MutationType(Enum):
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    CORRELATION = "correlation"


@dataclass
class LogicMutation:
    """Mutation signal sent to C++ MetamorphicCore"""
    mutation_id: int
    timestamp_ns: int
    mutation_type: str
    adaptation_rate: float
    mutation_params: list[float]
    is_priority: bool = False


@dataclass
class MarketState:
    """Current market state snapshot"""
    timestamp: float
    symbols: list[str]
    prices: dict[str, float]
    volumes: dict[str, float]
    spreads: dict[str, float]
    volatilities: dict[str, float]
    correlations: dict[tuple[str, str], float]


class CausalBrain:
    """
    The learning engine that observes market patterns and evolves trading logic.
    """

    def __init__(self, zmq_endpoint: str = "ipc:///tmp/hean_metamorphic"):
        self.zmq_endpoint = zmq_endpoint
        self.zmq_context = None
        self.zmq_socket = None

        # Learning state
        self.market_history: deque = deque(maxlen=10000)
        self.performance_history: deque = deque(maxlen=1000)
        self.mutation_counter = 0

        # Causal model parameters
        self.edge_confidence_threshold = 0.7
        self.risk_multiplier = 1.0
        self.execution_speed = 1.0
        self.spread_threshold = 0.0001

        # Learning rates
        self.learning_rate = 0.001
        self.adaptation_rate = 0.0

        # Performance tracking
        self.total_pnl = 0.0
        self.win_rate = 0.5
        self.avg_trade_duration = 0.0

        # Threading
        self.running = False
        self.mutation_thread: threading.Thread | None = None

        # System Intelligence Quotient (SIQ)
        self.siq_score = 0.0
        self.learning_velocity = 0.0
        self.pattern_recognition_rate = 0.0

    def initialize(self) -> bool:
        """Initialize ZeroMQ connection to C++ MetamorphicCore"""
        try:
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind(self.zmq_endpoint)
            return True
        except Exception as e:
            print(f"CausalBrain: Failed to initialize ZMQ: {e}")
            return False

    def update_market_state(self, state: MarketState):
        """Update internal market state and learn from patterns"""
        self.market_history.append(state)

        # Analyze patterns
        if len(self.market_history) > 100:
            self._analyze_market_patterns()
            self._compute_siq()

    def _analyze_market_patterns(self):
        """Analyze recent market patterns to identify causal relationships"""
        if len(self.market_history) < 100:
            return

        recent_states = list(self.market_history)[-100:]

        # Compute correlation changes
        correlation_changes = self._detect_correlation_shifts(recent_states)

        # Detect volatility regimes
        volatility_regime = self._detect_volatility_regime(recent_states)

        # Analyze execution quality
        execution_quality = self._analyze_execution_quality(recent_states)

        # Update learning parameters based on observations
        if correlation_changes > 0.15:
            # Market structure is changing rapidly
            self.adaptation_rate = min(1.0, self.adaptation_rate + 0.1)
            self.learning_rate *= 1.05
        elif correlation_changes < -0.1:
            # Market stabilizing
            self.adaptation_rate = max(0.0, self.adaptation_rate - 0.05)
            self.learning_rate *= 0.98

        if volatility_regime == "high":
            # Increase risk threshold in high volatility
            self.edge_confidence_threshold = min(0.9, self.edge_confidence_threshold + 0.01)
        elif volatility_regime == "low":
            # Lower threshold in stable markets
            self.edge_confidence_threshold = max(0.5, self.edge_confidence_threshold - 0.005)

        if execution_quality < 0.7:
            # Poor execution, need faster adaptation
            self.execution_speed *= 1.1

    def _detect_correlation_shifts(self, states: list[MarketState]) -> float:
        """Detect how much correlations are shifting"""
        if len(states) < 20:
            return 0.0

        # Compare early vs recent correlations
        early_corrs = self._compute_avg_correlations(states[:20])
        recent_corrs = self._compute_avg_correlations(states[-20:])

        shift = 0.0
        count = 0
        for key in early_corrs:
            if key in recent_corrs:
                shift += abs(early_corrs[key] - recent_corrs[key])
                count += 1

        return shift / max(count, 1)

    def _compute_avg_correlations(self, states: list[MarketState]) -> dict[tuple[str, str], float]:
        """Compute average correlations over a window"""
        if not states:
            return {}

        all_corrs = {}
        for state in states:
            for key, value in state.correlations.items():
                if key not in all_corrs:
                    all_corrs[key] = []
                all_corrs[key].append(value)

        avg_corrs = {}
        for key, values in all_corrs.items():
            avg_corrs[key] = np.mean(values) if values else 0.0

        return avg_corrs

    def _detect_volatility_regime(self, states: list[MarketState]) -> str:
        """Detect current volatility regime"""
        if not states:
            return "medium"

        volatilities = []
        for state in states:
            if state.volatilities:
                volatilities.extend(state.volatilities.values())

        if not volatilities:
            return "medium"

        avg_vol = np.mean(volatilities)
        std_vol = np.std(volatilities)

        if avg_vol > std_vol * 2:
            return "high"
        elif avg_vol < std_vol * 0.5:
            return "low"
        else:
            return "medium"

    def _analyze_execution_quality(self, states: list[MarketState]) -> float:
        """Analyze how well executions are performing"""
        if not states:
            return 0.5

        # Simple heuristic: lower spreads indicate better execution
        spreads = []
        for state in states:
            if state.spreads:
                spreads.extend(state.spreads.values())

        if not spreads:
            return 0.5

        avg_spread = np.mean(spreads)
        # Normalize to 0-1 quality score
        quality = 1.0 / (1.0 + avg_spread * 10000)
        return quality

    def _compute_siq(self):
        """Compute System Intelligence Quotient"""
        # SIQ = f(learning_velocity, pattern_recognition, adaptation_speed, performance)

        # Learning velocity: how fast we're adapting
        if len(self.market_history) > 100:
            recent_adaptations = self.adaptation_rate
            self.learning_velocity = recent_adaptations * 100

        # Pattern recognition rate: how many new patterns detected
        if len(self.market_history) > 200:
            pattern_changes = self._detect_correlation_shifts(
                list(self.market_history)[-200:-100]
            ) - self._detect_correlation_shifts(
                list(self.market_history)[-100:]
            )
            self.pattern_recognition_rate = abs(pattern_changes) * 10

        # Combine factors
        performance_factor = min(1.0, max(0.0, self.win_rate))
        adaptation_factor = min(1.0, self.adaptation_rate)

        self.siq_score = (
            0.3 * self.learning_velocity +
            0.3 * self.pattern_recognition_rate +
            0.2 * performance_factor +
            0.2 * adaptation_factor
        )

        # Normalize to 0-100 scale
        self.siq_score = min(100.0, max(0.0, self.siq_score * 25))

    def generate_mutation(self) -> LogicMutation:
        """Generate a logic mutation signal based on current learning state"""
        self.mutation_counter += 1

        timestamp_ns = int(time.time() * 1e9)

        # Determine mutation type based on what needs adaptation
        if self.adaptation_rate > 0.5:
            mutation_type = MutationType.CORRELATION
            params = [
                self.edge_confidence_threshold,
                self.risk_multiplier,
                self.execution_speed
            ]
        elif abs(self.risk_multiplier - 1.0) > 0.2:
            mutation_type = MutationType.RISK
            params = [self.risk_multiplier]
        elif self.execution_speed > 1.5 or self.execution_speed < 0.7:
            mutation_type = MutationType.EXECUTION
            params = [self.spread_threshold, self.execution_speed]
        else:
            mutation_type = MutationType.STRATEGY
            params = [self.edge_confidence_threshold, self.execution_speed]

        mutation = LogicMutation(
            mutation_id=self.mutation_counter,
            timestamp_ns=timestamp_ns,
            mutation_type=mutation_type.value,
            adaptation_rate=self.adaptation_rate,
            mutation_params=params,
            is_priority=self.adaptation_rate > 0.7
        )

        return mutation

    def send_mutation(self, mutation: LogicMutation):
        """Send mutation signal to C++ MetamorphicCore via ZeroMQ"""
        if not self.zmq_socket:
            return

        # Format: "mutation_id|timestamp|type|rate|param1,param2,..."
        params_str = ",".join([str(p) for p in mutation.mutation_params])
        message = f"{mutation.mutation_id}|{mutation.timestamp_ns}|{mutation.mutation_type}|{mutation.adaptation_rate}|{params_str}"

        try:
            self.zmq_socket.send_string(message)
        except Exception as e:
            print(f"CausalBrain: Failed to send mutation: {e}")

    def _mutation_loop(self):
        """Main loop that sends mutations every 100ms"""
        while self.running:
            mutation = self.generate_mutation()
            self.send_mutation(mutation)

            # Sleep for 100ms
            time.sleep(0.1)

    def start(self):
        """Start the mutation signal generator"""
        if self.running:
            return

        if not self.zmq_socket:
            if not self.initialize():
                return False

        self.running = True
        self.mutation_thread = threading.Thread(target=self._mutation_loop, daemon=True)
        self.mutation_thread.start()
        return True

    def stop(self):
        """Stop the mutation signal generator"""
        self.running = False
        if self.mutation_thread:
            self.mutation_thread.join(timeout=1.0)

        if self.zmq_socket:
            self.zmq_socket.close()
            self.zmq_socket = None

        if self.zmq_context:
            self.zmq_context.term()
            self.zmq_context = None

    def update_performance(self, pnl: float, win: bool, duration: float):
        """Update performance metrics"""
        self.total_pnl += pnl
        self.performance_history.append({"pnl": pnl, "win": win, "duration": duration})

        if len(self.performance_history) > 100:
            wins = sum(1 for p in list(self.performance_history)[-100:] if p["win"])
            self.win_rate = wins / 100.0

        if len(self.performance_history) > 50:
            durations = [p["duration"] for p in list(self.performance_history)[-50:]]
            self.avg_trade_duration = np.mean(durations) if durations else 0.0

    def get_siq(self) -> float:
        """Get current System Intelligence Quotient"""
        return self.siq_score

    def get_learning_stats(self) -> dict:
        """Get learning statistics"""
        return {
            "siq": self.siq_score,
            "learning_velocity": self.learning_velocity,
            "pattern_recognition_rate": self.pattern_recognition_rate,
            "adaptation_rate": self.adaptation_rate,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "edge_confidence_threshold": self.edge_confidence_threshold,
            "risk_multiplier": self.risk_multiplier,
            "execution_speed": self.execution_speed
        }
