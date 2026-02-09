"""
Regime Classifier - Классификатор рыночных режимов

Принимает RegimeFeatures, возвращает RegimeState с уверенностью
"""

from collections import deque

from .regime_types import MarketRegime, RegimeFeatures, RegimeState


class RegimeClassifier:
    """
    Классификатор рыночных режимов

    Использует rule-based подход + ML (будущее)
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

        # Regime history
        self.regime_history = deque(maxlen=1000)
        self.current_regime: MarketRegime | None = None
        self.regime_start_time_ns: int | None = None

        # Thresholds (можно подстроить под конкретный символ)
        self.thresholds = {
            # Trend
            'trend_min_strength': 0.6,
            'trend_min_ema_alignment': 0.3,
            'trend_min_price_change': 0.1,  # 0.1% per minute

            # Range
            'range_max_trend_strength': 0.4,
            'range_tight_width_bps': 20,  # < 20 bps = tight
            'range_wide_width_bps': 100,  # > 100 bps = wide

            # Volatility
            'high_vol_ratio': 2.0,  # 2x normal volatility
            'low_vol_ratio': 0.5,   # 0.5x normal volatility

            # Liquidity
            'thin_liquidity_spread_bps': 50,  # > 50 bps = thin
            'thin_liquidity_depth_threshold': 1000,  # Arbitrary

            # Shock
            'shock_price_jump_sigma': 3.0,  # 3 sigma move
            'shock_volume_spike': 5.0,      # 5x normal volume
            'shock_spread_spike': 3.0,      # 3x normal spread
        }

    def classify(self, features: RegimeFeatures, timestamp_ns: int) -> RegimeState:
        """
        Классифицирует текущий режим

        Returns RegimeState с режимом и уверенностью
        """

        # Rule-based classification
        regime_scores = self._calculate_regime_scores(features)

        # Primary regime (highest confidence)
        primary_regime = max(regime_scores.items(), key=lambda x: x[1])
        regime, confidence = primary_regime

        # Secondary regimes (confidence > 0.3)
        secondary_regimes = {
            r: score for r, score in regime_scores.items()
            if r != regime and score > 0.3
        }

        # Regime duration
        if regime != self.current_regime:
            self.regime_start_time_ns = timestamp_ns
            self.current_regime = regime

        regime_duration_ms = (timestamp_ns - self.regime_start_time_ns) / 1_000_000 if self.regime_start_time_ns else 0

        # Count regime changes in last hour
        one_hour_ago_ns = timestamp_ns - (3600 * 1_000_000_000)
        regime_changes = len([
            r for r in self.regime_history
            if r['timestamp_ns'] > one_hour_ago_ns and r['regime'] != regime
        ])

        # Create state
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            secondary_regimes=secondary_regimes,
            features=features.to_dict(),
            symbol=self.symbol,
            timestamp_ns=timestamp_ns,
            regime_duration_ms=regime_duration_ms,
            regime_changes_last_hour=regime_changes,
        )

        # Record to history
        self.regime_history.append({
            'timestamp_ns': timestamp_ns,
            'regime': regime,
            'confidence': confidence,
        })

        return state

    def _calculate_regime_scores(self, features: RegimeFeatures) -> dict[MarketRegime, float]:
        """
        Вычисляет scores для всех режимов

        Returns dict {regime: confidence}
        """

        scores = {}

        # TREND_UP
        scores[MarketRegime.TREND_UP] = self._score_trend_up(features)

        # TREND_DOWN
        scores[MarketRegime.TREND_DOWN] = self._score_trend_down(features)

        # RANGE_TIGHT
        scores[MarketRegime.RANGE_TIGHT] = self._score_range_tight(features)

        # RANGE_WIDE
        scores[MarketRegime.RANGE_WIDE] = self._score_range_wide(features)

        # HIGH_VOL
        scores[MarketRegime.HIGH_VOL] = self._score_high_vol(features)

        # LOW_VOL
        scores[MarketRegime.LOW_VOL] = self._score_low_vol(features)

        # THIN_LIQUIDITY
        scores[MarketRegime.THIN_LIQUIDITY] = self._score_thin_liquidity(features)

        # NEWS_SHOCK
        scores[MarketRegime.NEWS_SHOCK] = self._score_news_shock(features)

        # Normalize scores to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {r: s / max_score for r, s in scores.items()}

        return scores

    def _score_trend_up(self, f: RegimeFeatures) -> float:
        """Score для восходящего тренда"""
        score = 0.0

        # Price changes positive
        if f.price_change_1m > self.thresholds['trend_min_price_change']:
            score += 0.3
        if f.price_change_5m > self.thresholds['trend_min_price_change'] * 3:
            score += 0.2

        # EMA alignment positive
        if f.ema_alignment > self.thresholds['trend_min_ema_alignment']:
            score += 0.3 * (f.ema_alignment / 1.0)

        # Trend strength high
        if f.trend_strength > self.thresholds['trend_min_strength']:
            score += 0.2 * f.trend_strength

        return min(score, 1.0)

    def _score_trend_down(self, f: RegimeFeatures) -> float:
        """Score для нисходящего тренда"""
        score = 0.0

        # Price changes negative
        if f.price_change_1m < -self.thresholds['trend_min_price_change']:
            score += 0.3
        if f.price_change_5m < -self.thresholds['trend_min_price_change'] * 3:
            score += 0.2

        # EMA alignment negative
        if f.ema_alignment < -self.thresholds['trend_min_ema_alignment']:
            score += 0.3 * (abs(f.ema_alignment) / 1.0)

        # Trend strength high
        if f.trend_strength > self.thresholds['trend_min_strength']:
            score += 0.2 * f.trend_strength

        return min(score, 1.0)

    def _score_range_tight(self, f: RegimeFeatures) -> float:
        """Score для узкого диапазона"""
        score = 0.0

        # Low trend strength
        if f.trend_strength < self.thresholds['range_max_trend_strength']:
            score += 0.3

        # Tight range
        if f.range_width_bps < self.thresholds['range_tight_width_bps']:
            score += 0.4

        # Low volatility
        if f.volatility_ratio < 1.0:
            score += 0.3

        return min(score, 1.0)

    def _score_range_wide(self, f: RegimeFeatures) -> float:
        """Score для широкого диапазона"""
        score = 0.0

        # Low trend strength
        if f.trend_strength < self.thresholds['range_max_trend_strength']:
            score += 0.3

        # Wide range
        if f.range_width_bps > self.thresholds['range_wide_width_bps']:
            score += 0.5

        # Price oscillating
        if 0.3 < f.price_position_in_range < 0.7:
            score += 0.2

        return min(score, 1.0)

    def _score_high_vol(self, f: RegimeFeatures) -> float:
        """Score для высокой волатильности"""
        score = 0.0

        # High volatility ratio
        if f.volatility_ratio > self.thresholds['high_vol_ratio']:
            score += 0.5 * min((f.volatility_ratio / self.thresholds['high_vol_ratio']), 1.0)

        # High ATR percentile
        if f.atr_percentile > 0.8:
            score += 0.3

        # Large price moves
        if abs(f.price_change_1m) > 0.3:  # > 0.3% per minute
            score += 0.2

        return min(score, 1.0)

    def _score_low_vol(self, f: RegimeFeatures) -> float:
        """Score для низкой волатильности"""
        score = 0.0

        # Low volatility ratio
        if f.volatility_ratio < self.thresholds['low_vol_ratio']:
            score += 0.5

        # Low ATR percentile
        if f.atr_percentile < 0.3:
            score += 0.3

        # Small price moves
        if abs(f.price_change_1m) < 0.05:  # < 0.05% per minute
            score += 0.2

        return min(score, 1.0)

    def _score_thin_liquidity(self, f: RegimeFeatures) -> float:
        """Score для тонкой ликвидности"""
        score = 0.0

        # Wide spread
        if f.spread_bps > self.thresholds['thin_liquidity_spread_bps']:
            score += 0.5

        # Low orderbook depth
        if f.orderbook_depth < self.thresholds['thin_liquidity_depth_threshold']:
            score += 0.3

        # Low trade rate
        if f.trade_rate_hz < 1.0:  # < 1 trade per second
            score += 0.2

        return min(score, 1.0)

    def _score_news_shock(self, f: RegimeFeatures) -> float:
        """Score для новостного шока"""
        score = 0.0

        # Large price jump
        if f.price_jump_sigma > self.thresholds['shock_price_jump_sigma']:
            score += 0.4

        # Volume spike
        if f.volume_spike_ratio > self.thresholds['shock_volume_spike']:
            score += 0.3

        # Spread spike
        if f.spread_spike_ratio > self.thresholds['shock_spread_spike']:
            score += 0.3

        return min(score, 1.0)

    def get_regime_history(self, last_n: int = 100) -> list[dict]:
        """Возвращает историю режимов"""
        return list(self.regime_history)[-last_n:]

    def get_regime_statistics(self) -> dict:
        """Возвращает статистику по режимам"""
        if not self.regime_history:
            return {}

        # Count regimes
        regime_counts = {}
        for entry in self.regime_history:
            regime = entry['regime']
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

        # Total time in each regime
        total_entries = len(self.regime_history)
        regime_percentages = {
            regime: count / total_entries * 100
            for regime, count in regime_counts.items()
        }

        # Current regime duration
        current_duration_ms = 0
        if self.regime_start_time_ns and self.regime_history:
            latest_ts = self.regime_history[-1]['timestamp_ns']
            current_duration_ms = (latest_ts - self.regime_start_time_ns) / 1_000_000

        return {
            'regime_counts': regime_counts,
            'regime_percentages': regime_percentages,
            'current_regime': self.current_regime.value if self.current_regime else None,
            'current_regime_duration_ms': current_duration_ms,
            'total_samples': total_entries,
        }
