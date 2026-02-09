"""
Feature Extractor - Извлечение признаков для классификации режима

Принимает рыночные данные, возвращает RegimeFeatures
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Fallback: use built-in statistics

from collections import deque

from ..nervous_system.event_envelope import EventEnvelope, EventType
from .regime_types import RegimeFeatures


class FeatureExtractor:
    """
    Извлечение признаков из рыночных данных

    Накапливает историю и вычисляет фичи в реальном времени
    """

    def __init__(self, symbol: str, window_minutes: int = 15):
        self.symbol = symbol
        self.window_minutes = window_minutes

        # Price history
        self.price_history = deque(maxlen=1000)  # Last 1000 prices with timestamps

        # Trade history
        self.trade_history = deque(maxlen=5000)  # Last 5000 trades

        # Orderbook snapshots
        self.orderbook_history = deque(maxlen=100)  # Last 100 snapshots

        # Computed indicators
        self.ema_8 = None
        self.ema_21 = None
        self.ema_55 = None
        self.atr_14 = None

        # Statistics
        self.last_update_ns = 0

    def process_event(self, event: EventEnvelope):
        """Обрабатывает событие и обновляет историю"""
        if event.symbol != self.symbol:
            return

        self.last_update_ns = event.timestamp_ns

        if event.event_type == EventType.TRADE:
            self._process_trade(event)
        elif event.event_type == EventType.ORDERBOOK:
            self._process_orderbook(event)
        elif event.event_type == EventType.CANDLE:
            self._process_candle(event)

    def _process_trade(self, event: EventEnvelope):
        """Обрабатывает трейд"""
        trade = {
            'timestamp_ns': event.timestamp_ns,
            'price': event.data['price'],
            'size': event.data['size'],
            'side': event.data['side'],
        }
        self.trade_history.append(trade)

        # Update price history
        self.price_history.append({
            'timestamp_ns': event.timestamp_ns,
            'price': event.data['price'],
        })

    def _process_orderbook(self, event: EventEnvelope):
        """Обрабатывает обновление стакана"""
        snapshot = {
            'timestamp_ns': event.timestamp_ns,
            'bids': event.data.get('bids', []),
            'asks': event.data.get('asks', []),
        }
        self.orderbook_history.append(snapshot)

    def _process_candle(self, event: EventEnvelope):
        """Обрабатывает свечу"""
        price = event.data['close']
        self.price_history.append({
            'timestamp_ns': event.timestamp_ns,
            'price': price,
        })

    def extract_features(self) -> RegimeFeatures | None:
        """
        Извлекает фичи из накопленной истории

        Returns None если недостаточно данных
        """
        if len(self.price_history) < 100:
            return None  # Not enough data

        # Extract all features
        trend_features = self._extract_trend_features()
        range_features = self._extract_range_features()
        volatility_features = self._extract_volatility_features()
        liquidity_features = self._extract_liquidity_features()
        microstructure_features = self._extract_microstructure_features()
        shock_features = self._extract_shock_features()

        return RegimeFeatures(
            **trend_features,
            **range_features,
            **volatility_features,
            **liquidity_features,
            **microstructure_features,
            **shock_features,
        )

    def _extract_trend_features(self) -> dict[str, float]:
        """Извлекает трендовые фичи"""
        prices = self._get_recent_prices()
        if len(prices) < 100:
            return self._default_trend_features()

        current_price = prices[-1]

        # Price changes
        price_1m_ago = self._get_price_n_minutes_ago(1)
        price_5m_ago = self._get_price_n_minutes_ago(5)
        price_15m_ago = self._get_price_n_minutes_ago(15)

        price_change_1m = ((current_price - price_1m_ago) / price_1m_ago * 100) if price_1m_ago else 0
        price_change_5m = ((current_price - price_5m_ago) / price_5m_ago * 100) if price_5m_ago else 0
        price_change_15m = ((current_price - price_15m_ago) / price_15m_ago * 100) if price_15m_ago else 0

        # EMAs
        self._update_emas(prices)
        ema_alignment = self._calculate_ema_alignment()

        # Trend strength (ADX-like)
        trend_strength = self._calculate_trend_strength(prices)

        return {
            'price_change_1m': price_change_1m,
            'price_change_5m': price_change_5m,
            'price_change_15m': price_change_15m,
            'ema_alignment': ema_alignment,
            'trend_strength': trend_strength,
        }

    def _extract_range_features(self) -> dict[str, float]:
        """Извлекает range фичи"""
        prices = self._get_recent_prices()
        if len(prices) < 100:
            return self._default_range_features()

        # ATR
        atr = self._calculate_atr(prices)
        atr_percentile = self._calculate_atr_percentile(atr)

        # Range
        recent_high = max(prices[-100:])
        recent_low = min(prices[-100:])
        range_width = recent_high - recent_low
        range_width_bps = (range_width / recent_low * 10000) if recent_low else 0

        # Price position in range
        current_price = prices[-1]
        price_position = ((current_price - recent_low) / (recent_high - recent_low)) if range_width > 0 else 0.5

        return {
            'atr_percentile': atr_percentile,
            'price_position_in_range': price_position,
            'range_width_bps': range_width_bps,
        }

    def _extract_volatility_features(self) -> dict[str, float]:
        """Извлекает фичи волатильности"""
        prices = self._get_recent_prices()
        if len(prices) < 100:
            return self._default_volatility_features()

        # Returns
        returns = np.diff(prices) / prices[:-1]

        # Volatility (std of returns)
        vol_1m = np.std(returns[-60:]) if len(returns) >= 60 else 0
        vol_5m = np.std(returns[-300:]) if len(returns) >= 300 else 0
        vol_historical = np.std(returns) if len(returns) > 0 else 1e-8

        vol_ratio = vol_1m / vol_historical if vol_historical > 0 else 1.0

        return {
            'volatility_1m': vol_1m * 100,  # Convert to %
            'volatility_5m': vol_5m * 100,
            'volatility_ratio': vol_ratio,
        }

    def _extract_liquidity_features(self) -> dict[str, float]:
        """Извлекает фичи ликвидности"""
        if len(self.orderbook_history) == 0:
            return self._default_liquidity_features()

        latest_ob = self.orderbook_history[-1]
        bids = latest_ob['bids']
        asks = latest_ob['asks']

        if not bids or not asks:
            return self._default_liquidity_features()

        # Spread
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid * 10000) if best_bid > 0 else 0

        # Orderbook depth (sum of top 10 levels)
        bid_depth = sum(float(level[1]) for level in bids[:10])
        ask_depth = sum(float(level[1]) for level in asks[:10])
        total_depth = bid_depth + ask_depth

        # Volume ratio
        if len(self.trade_history) >= 100:
            recent_volume = sum(t['size'] for t in list(self.trade_history)[-60:])
            avg_volume = sum(t['size'] for t in list(self.trade_history)[-300:]) / 5 if len(self.trade_history) >= 300 else recent_volume
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0

        return {
            'spread_bps': spread_bps,
            'orderbook_depth': total_depth,
            'volume_ratio': volume_ratio,
        }

    def _extract_microstructure_features(self) -> dict[str, float]:
        """Извлекает микроструктурные фичи"""
        if len(self.trade_history) < 10:
            return self._default_microstructure_features()

        recent_trades = list(self.trade_history)[-100:]

        # Trade rate
        if len(recent_trades) >= 2:
            time_span_sec = (recent_trades[-1]['timestamp_ns'] - recent_trades[0]['timestamp_ns']) / 1e9
            trade_rate_hz = len(recent_trades) / time_span_sec if time_span_sec > 0 else 0
        else:
            trade_rate_hz = 0

        # Buy/sell imbalance
        buy_volume = sum(t['size'] for t in recent_trades if t['side'] == 'Buy')
        sell_volume = sum(t['size'] for t in recent_trades if t['side'] == 'Sell')
        total_volume = buy_volume + sell_volume
        buy_sell_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

        # Large trades (> 2x average size)
        avg_size = sum(t['size'] for t in recent_trades) / len(recent_trades)
        large_trades = [t for t in recent_trades if t['size'] > avg_size * 2]
        large_trade_frequency = len(large_trades) / len(recent_trades) if recent_trades else 0

        return {
            'trade_rate_hz': trade_rate_hz,
            'buy_sell_imbalance': buy_sell_imbalance,
            'large_trade_frequency': large_trade_frequency,
        }

    def _extract_shock_features(self) -> dict[str, float]:
        """Извлекает фичи шоков"""
        prices = self._get_recent_prices()
        if len(prices) < 100:
            return self._default_shock_features()

        # Price jump in sigmas
        returns = np.diff(prices) / prices[:-1]
        recent_return = returns[-1] if len(returns) > 0 else 0
        return_std = np.std(returns) if len(returns) > 1 else 1e-8
        price_jump_sigma = abs(recent_return / return_std) if return_std > 0 else 0

        # Volume spike
        if len(self.trade_history) >= 100:
            recent_trades = list(self.trade_history)[-20:]
            historical_trades = list(self.trade_history)[-300:-20]
            recent_volume = sum(t['size'] for t in recent_trades)
            avg_volume = sum(t['size'] for t in historical_trades) / len(historical_trades) * 20 if historical_trades else recent_volume
            volume_spike_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_spike_ratio = 1.0

        # Spread spike
        if len(self.orderbook_history) >= 20:
            recent_spreads = []
            for ob in list(self.orderbook_history)[-5:]:
                if ob['bids'] and ob['asks']:
                    spread = float(ob['asks'][0][0]) - float(ob['bids'][0][0])
                    recent_spreads.append(spread)

            historical_spreads = []
            for ob in list(self.orderbook_history)[-50:-5]:
                if ob['bids'] and ob['asks']:
                    spread = float(ob['asks'][0][0]) - float(ob['bids'][0][0])
                    historical_spreads.append(spread)

            current_spread = recent_spreads[-1] if recent_spreads else 0
            avg_spread = np.mean(historical_spreads) if historical_spreads else current_spread
            spread_spike_ratio = current_spread / avg_spread if avg_spread > 0 else 1.0
        else:
            spread_spike_ratio = 1.0

        return {
            'price_jump_sigma': price_jump_sigma,
            'volume_spike_ratio': volume_spike_ratio,
            'spread_spike_ratio': spread_spike_ratio,
        }

    # Helper methods

    def _get_recent_prices(self, n: int | None = None) -> list[float]:
        """Возвращает последние N цен"""
        prices = [p['price'] for p in self.price_history]
        return prices if n is None else prices[-n:]

    def _get_price_n_minutes_ago(self, minutes: int) -> float | None:
        """Возвращает цену N минут назад"""
        target_ns = self.last_update_ns - (minutes * 60 * 1_000_000_000)

        # Find closest price
        for p in reversed(self.price_history):
            if p['timestamp_ns'] <= target_ns:
                return p['price']

        return None

    def _update_emas(self, prices: list[float]):
        """Обновляет EMA индикаторы"""
        if self.ema_8 is None:
            self.ema_8 = prices[-1]
            self.ema_21 = prices[-1]
            self.ema_55 = prices[-1]

        # Simple EMA update
        alpha_8 = 2 / (8 + 1)
        alpha_21 = 2 / (21 + 1)
        alpha_55 = 2 / (55 + 1)

        self.ema_8 = alpha_8 * prices[-1] + (1 - alpha_8) * self.ema_8
        self.ema_21 = alpha_21 * prices[-1] + (1 - alpha_21) * self.ema_21
        self.ema_55 = alpha_55 * prices[-1] + (1 - alpha_55) * self.ema_55

    def _calculate_ema_alignment(self) -> float:
        """
        Вычисляет выравнивание EMA

        +1.0 = perfect uptrend (EMA8 > EMA21 > EMA55)
        -1.0 = perfect downtrend (EMA8 < EMA21 < EMA55)
        0.0 = mixed
        """
        if self.ema_8 is None:
            return 0.0

        # Check alignment
        uptrend = self.ema_8 > self.ema_21 > self.ema_55
        downtrend = self.ema_8 < self.ema_21 < self.ema_55

        if uptrend:
            # Measure strength by distance
            distance = (self.ema_8 - self.ema_55) / self.ema_55
            return min(distance * 100, 1.0)  # Normalize to [0, 1]
        elif downtrend:
            distance = (self.ema_55 - self.ema_8) / self.ema_55
            return max(-distance * 100, -1.0)  # Normalize to [-1, 0]
        else:
            return 0.0

    def _calculate_trend_strength(self, prices: list[float]) -> float:
        """Вычисляет силу тренда (ADX-like)"""
        if len(prices) < 14:
            return 0.0

        # Simple trend strength: correlation with linear regression
        x = np.arange(len(prices[-100:]))
        y = np.array(prices[-100:])

        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            return abs(correlation)  # 0-1, higher = stronger trend
        return 0.0

    def _calculate_atr(self, prices: list[float], period: int = 14) -> float:
        """Вычисляет ATR"""
        if len(prices) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i]
            low = prices[i]
            prev_close = prices[i - 1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Average of last 14 true ranges
        atr = np.mean(true_ranges[-period:]) if len(true_ranges) >= period else 0
        return atr

    def _calculate_atr_percentile(self, current_atr: float) -> float:
        """Вычисляет процентиль ATR (0-1)"""
        prices = self._get_recent_prices()
        if len(prices) < 100:
            return 0.5

        # Calculate historical ATRs
        historical_atrs = []
        for i in range(14, len(prices), 10):  # Every 10 bars
            window = prices[i - 14:i]
            atr = self._calculate_atr(window, period=14)
            historical_atrs.append(atr)

        if not historical_atrs:
            return 0.5

        # Calculate percentile
        percentile = sum(1 for atr in historical_atrs if atr < current_atr) / len(historical_atrs)
        return percentile

    # Default features when not enough data

    def _default_trend_features(self) -> dict[str, float]:
        return {
            'price_change_1m': 0.0,
            'price_change_5m': 0.0,
            'price_change_15m': 0.0,
            'ema_alignment': 0.0,
            'trend_strength': 0.0,
        }

    def _default_range_features(self) -> dict[str, float]:
        return {
            'atr_percentile': 0.5,
            'price_position_in_range': 0.5,
            'range_width_bps': 0.0,
        }

    def _default_volatility_features(self) -> dict[str, float]:
        return {
            'volatility_1m': 0.0,
            'volatility_5m': 0.0,
            'volatility_ratio': 1.0,
        }

    def _default_liquidity_features(self) -> dict[str, float]:
        return {
            'spread_bps': 0.0,
            'orderbook_depth': 0.0,
            'volume_ratio': 1.0,
        }

    def _default_microstructure_features(self) -> dict[str, float]:
        return {
            'trade_rate_hz': 0.0,
            'buy_sell_imbalance': 0.0,
            'large_trade_frequency': 0.0,
        }

    def _default_shock_features(self) -> dict[str, float]:
        return {
            'price_jump_sigma': 0.0,
            'volume_spike_ratio': 1.0,
            'spread_spike_ratio': 1.0,
        }
