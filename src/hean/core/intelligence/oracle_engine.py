"""
Oracle Engine: Integration of Algorithmic Fingerprinting and TCN-based Price Prediction
Combines fingerprinting signals with TCN reversal predictions to provide predictive alpha
"""


from hean.core.intelligence.tcn_predictor import TCPriceReversalPredictor
from hean.core.types import Tick
from hean.logging import get_logger

logger = get_logger(__name__)

# Try to import C++ fingerprinting engine
try:
    import graph_engine_py as cpp_engine
    CPP_ENGINE_AVAILABLE = True
except ImportError:
    logger.warning("C++ graph_engine_py not available. Fingerprinting will be disabled.")
    CPP_ENGINE_AVAILABLE = False


class OracleEngine:
    """
    Oracle Engine: Combines Algorithmic Fingerprinting and TCN predictions
    Provides Predictive Alpha signals to the trading swarm
    """

    def __init__(self, sequence_length: int = 10000):
        """
        Initialize Oracle Engine.

        Args:
            sequence_length: Number of micro-ticks for TCN (default: 10,000)
        """
        self.tcn_predictor = TCPriceReversalPredictor(sequence_length=sequence_length)
        self.fingerprinting_enabled = CPP_ENGINE_AVAILABLE

        if self.fingerprinting_enabled:
            try:
                cpp_engine.algo_fingerprinter_init()
                logger.info("Algorithmic Fingerprinting Engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize fingerprinting engine: {e}")
                self.fingerprinting_enabled = False

        # Cache for predictions
        self.last_predictions: dict[str, dict] = {}

        # Price predictions at different horizons
        self.price_predictions: dict[str, dict] = {}  # symbol -> {500ms: price, 1s: price, 5s: price}

    def update_tick(self, tick: Tick) -> None:
        """
        Update with new tick data.

        Args:
            tick: Market tick data
        """
        # Update TCN predictor
        volume = tick.volume if hasattr(tick, 'volume') else 0.0
        bid = tick.bid if tick.bid is not None else tick.price * 0.9999
        ask = tick.ask if tick.ask is not None else tick.price * 1.0001

        self.tcn_predictor.update_tick(
            price=tick.price,
            volume=volume,
            bid=bid,
            ask=ask,
            timestamp=tick.timestamp
        )

        # Get TCN prediction
        tcn_result = self.tcn_predictor.get_prediction_with_confidence()

        # Get fingerprinting alpha signal
        fingerprint_alpha = None
        if self.fingerprinting_enabled:
            try:
                fingerprint_result = cpp_engine.algo_fingerprinter_get_predictive_alpha(tick.symbol)
                if fingerprint_result.get('signal_available', False):
                    fingerprint_alpha = {
                        'alpha_signal': fingerprint_result['alpha_signal'],
                        'confidence': fingerprint_result['confidence'],
                        'bot_id': fingerprint_result.get('bot_id', 'UNKNOWN')
                    }
            except Exception as e:
                logger.debug(f"Fingerprinting error for {tick.symbol}: {e}")

        # Combine predictions
        prediction = {
            'symbol': tick.symbol,
            'timestamp': tick.timestamp,
            'current_price': tick.price,
            'tcn_reversal_prob': tcn_result['probability'],
            'tcn_should_trigger': tcn_result['should_trigger'],
            'tcn_confidence_high': tcn_result['confidence_high'],
            'tcn_confidence_low': tcn_result['confidence_low'],
            'fingerprint_alpha': fingerprint_alpha,
        }

        # Calculate price predictions at different horizons
        self._calculate_price_predictions(tick, prediction)

        self.last_predictions[tick.symbol] = prediction

    def _calculate_price_predictions(self, tick: Tick, prediction: dict) -> None:
        """
        Calculate price predictions at 500ms, 1s, and 5s horizons.

        Args:
            tick: Current tick
            prediction: Prediction dictionary to update
        """
        current_price = tick.price

        # Get momentum from recent price changes
        tcn_prob = prediction['tcn_reversal_prob']
        fingerprint_alpha = prediction.get('fingerprint_alpha')

        # Calculate expected price movement
        # If TCN predicts reversal (high prob), expect price to reverse
        # If fingerprinting shows bullish alpha, expect price to rise
        # Combine signals with confidence weighting

        reversal_strength = tcn_prob if prediction['tcn_should_trigger'] else 0.0
        alpha_strength = 0.0
        if fingerprint_alpha:
            alpha_strength = fingerprint_alpha['alpha_signal'] * fingerprint_alpha['confidence']

        # Expected return (negative if reversal, positive if continuation)
        expected_return_500ms = -0.0005 * reversal_strength + 0.0003 * alpha_strength  # ~0.05% max
        expected_return_1s = -0.001 * reversal_strength + 0.0006 * alpha_strength  # ~0.1% max
        expected_return_5s = -0.003 * reversal_strength + 0.002 * alpha_strength  # ~0.3% max

        # Calculate confidence (lower confidence for longer horizons)
        confidence_500ms = 0.7 * (tcn_prob + (1.0 if fingerprint_alpha else 0.5))
        confidence_1s = 0.5 * (tcn_prob + (1.0 if fingerprint_alpha else 0.5))
        confidence_5s = 0.3 * (tcn_prob + (1.0 if fingerprint_alpha else 0.5))

        # Clamp confidence to [0, 1]
        confidence_500ms = max(0.0, min(1.0, confidence_500ms))
        confidence_1s = max(0.0, min(1.0, confidence_1s))
        confidence_5s = max(0.0, min(1.0, confidence_5s))

        # Calculate predicted prices
        price_500ms = current_price * (1.0 + expected_return_500ms)
        price_1s = current_price * (1.0 + expected_return_1s)
        price_5s = current_price * (1.0 + expected_return_5s)

        # Store predictions
        self.price_predictions[tick.symbol] = {
            '500ms': {
                'price': price_500ms,
                'confidence': confidence_500ms,
                'return_pct': expected_return_500ms * 100
            },
            '1s': {
                'price': price_1s,
                'confidence': confidence_1s,
                'return_pct': expected_return_1s * 100
            },
            '5s': {
                'price': price_5s,
                'confidence': confidence_5s,
                'return_pct': expected_return_5s * 100
            },
            'current_price': current_price,
            'timestamp': tick.timestamp
        }

        prediction['price_predictions'] = self.price_predictions[tick.symbol]

    def get_predictive_alpha(self, symbol: str) -> dict | None:
        """
        Get Predictive Alpha signal for a symbol.
        Combines TCN reversal prediction with fingerprinting alpha.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with predictive alpha signal, or None if no signal
        """
        if symbol not in self.last_predictions:
            return None

        prediction = self.last_predictions[symbol]

        # Determine if we should trigger an exit or flip
        should_exit = False
        should_flip = False
        signal_strength = 0.0

        # If TCN predicts reversal > 85%, trigger exit
        if prediction['tcn_should_trigger']:
            should_exit = True
            signal_strength = prediction['tcn_reversal_prob']

        # If fingerprinting shows strong opposite alpha, consider flipping
        fingerprint_alpha = prediction.get('fingerprint_alpha')
        if fingerprint_alpha and abs(fingerprint_alpha['alpha_signal']) > 0.7:
            if should_exit:
                # Strong reversal + opposite alpha = flip position
                should_flip = True
                should_exit = False
                signal_strength = max(signal_strength, fingerprint_alpha['confidence'])
            else:
                signal_strength = fingerprint_alpha['confidence']

        if not (should_exit or should_flip) and signal_strength < 0.6:
            return None

        return {
            'symbol': symbol,
            'should_exit': should_exit,
            'should_flip': should_flip,
            'signal_strength': signal_strength,
            'tcn_reversal_prob': prediction['tcn_reversal_prob'],
            'fingerprint_alpha': fingerprint_alpha,
            'price_predictions': prediction.get('price_predictions'),
            'timestamp': prediction['timestamp']
        }

    def get_price_predictions(self, symbol: str) -> dict | None:
        """
        Get price predictions at 500ms, 1s, and 5s horizons.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with price predictions and confidence intervals, or None
        """
        return self.price_predictions.get(symbol)

    def update_order_book(self, symbol: str, order_id: str, price: float, size: float,
                         timestamp_ns: int, is_limit: bool) -> None:
        """
        Update order book information for fingerprinting.
        Only tracks large limit orders.

        Args:
            symbol: Trading symbol
            order_id: Order identifier
            price: Order price
            size: Order size
            timestamp_ns: Timestamp in nanoseconds
            is_limit: True if limit order, False if market order
        """
        if not self.fingerprinting_enabled:
            return

        try:
            cpp_engine.algo_fingerprinter_update_order(
                order_id=order_id,
                symbol=symbol,
                price=price,
                size=size,
                timestamp_ns=timestamp_ns,
                is_limit=is_limit
            )
        except Exception as e:
            logger.debug(f"Failed to update order for fingerprinting: {e}")

    def remove_order(self, order_id: str) -> None:
        """
        Remove order from fingerprinting tracking.

        Args:
            order_id: Order identifier
        """
        if not self.fingerprinting_enabled:
            return

        try:
            cpp_engine.algo_fingerprinter_remove_order(order_id)
        except Exception as e:
            logger.debug(f"Failed to remove order from fingerprinting: {e}")
