"""
Типы рыночных режимов

Определения всех возможных состояний рынка
"""

from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Рыночные режимы"""
    TREND_UP = "trend_up"           # Восходящий тренд
    TREND_DOWN = "trend_down"       # Нисходящий тренд
    RANGE_TIGHT = "range_tight"     # Узкий диапазон
    RANGE_WIDE = "range_wide"       # Широкий диапазон
    HIGH_VOL = "high_volatility"    # Высокая волатильность
    LOW_VOL = "low_volatility"      # Низкая волатильность
    THIN_LIQUIDITY = "thin_liquidity"  # Тонкая ликвидность
    NEWS_SHOCK = "news_shock"       # Новостной шок
    UNKNOWN = "unknown"             # Неизвестный режим


@dataclass
class RegimeState:
    """
    Состояние рыночного режима

    Содержит текущий режим, уверенность, фичи
    """

    # Primary regime
    regime: MarketRegime
    confidence: float  # 0.0 - 1.0

    # Secondary regimes (может быть комбинация)
    secondary_regimes: dict[MarketRegime, float]

    # Features used for classification
    features: dict[str, float]

    # Metadata
    symbol: str
    timestamp_ns: int

    # Stability
    regime_duration_ms: float  # How long in current regime
    regime_changes_last_hour: int

    def is_stable(self) -> bool:
        """Стабильный ли режим"""
        return (
            self.confidence > 0.7 and
            self.regime_duration_ms > 60000 and  # > 1 minute
            self.regime_changes_last_hour < 5
        )

    def is_trending(self) -> bool:
        """Есть ли тренд"""
        return self.regime in [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]

    def is_ranging(self) -> bool:
        """Находится ли в диапазоне"""
        return self.regime in [MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE]

    def is_volatile(self) -> bool:
        """Высокая ли волатильность"""
        return self.regime == MarketRegime.HIGH_VOL or self.confidence_for(MarketRegime.HIGH_VOL) > 0.5

    def is_dangerous(self) -> bool:
        """Опасный ли режим для торговли"""
        return self.regime in [
            MarketRegime.NEWS_SHOCK,
            MarketRegime.THIN_LIQUIDITY,
        ] or (
            self.regime == MarketRegime.HIGH_VOL and self.confidence > 0.8
        )

    def confidence_for(self, regime: MarketRegime) -> float:
        """Уверенность для конкретного режима"""
        if regime == self.regime:
            return self.confidence
        return self.secondary_regimes.get(regime, 0.0)

    def get_best_strategy_type(self) -> str:
        """Рекомендованный тип стратегии для текущего режима"""
        if self.regime == MarketRegime.TREND_UP:
            return "momentum_long"
        elif self.regime == MarketRegime.TREND_DOWN:
            return "momentum_short"
        elif self.regime in [MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE]:
            return "mean_reversion"
        elif self.regime == MarketRegime.HIGH_VOL:
            return "volatility_breakout"
        elif self.regime == MarketRegime.LOW_VOL:
            return "range_scalp"
        elif self.regime == MarketRegime.THIN_LIQUIDITY:
            return "passive_only"  # Только пассивные ордера
        elif self.regime == MarketRegime.NEWS_SHOCK:
            return "safe_mode"  # Не торговать
        else:
            return "adaptive"


@dataclass
class RegimeFeatures:
    """
    Фичи для классификации режима

    Извлекаются из рыночных данных
    """

    # Trend features
    price_change_1m: float      # % изменение за 1 минуту
    price_change_5m: float      # % изменение за 5 минут
    price_change_15m: float     # % изменение за 15 минут
    ema_alignment: float        # Выравнивание EMA (8, 21, 55)
    trend_strength: float       # Сила тренда (0-1)

    # Range features
    atr_percentile: float       # ATR в процентилях
    price_position_in_range: float  # Позиция цены в диапазоне (0-1)
    range_width_bps: float      # Ширина диапазона в bps

    # Volatility features
    volatility_1m: float        # Волатильность 1м (стандартное отклонение)
    volatility_5m: float        # Волатильность 5м
    volatility_ratio: float     # Ratio текущей / исторической

    # Liquidity features
    spread_bps: float           # Спред в базисных пунктах
    orderbook_depth: float      # Глубина стакана (top 10 levels)
    volume_ratio: float         # Текущий volume / средний

    # Microstructure features
    trade_rate_hz: float        # Частота трейдов
    buy_sell_imbalance: float   # Дисбаланс покупок/продаж
    large_trade_frequency: float  # Частота крупных трейдов

    # Shock detection features
    price_jump_sigma: float     # Размер прыжка цены в сигмах
    volume_spike_ratio: float   # Спайк объёма / нормальный
    spread_spike_ratio: float   # Спайк спреда / нормальный

    def to_dict(self) -> dict[str, float]:
        """Конвертация в словарь для ML"""
        return {
            'price_change_1m': self.price_change_1m,
            'price_change_5m': self.price_change_5m,
            'price_change_15m': self.price_change_15m,
            'ema_alignment': self.ema_alignment,
            'trend_strength': self.trend_strength,
            'atr_percentile': self.atr_percentile,
            'price_position_in_range': self.price_position_in_range,
            'range_width_bps': self.range_width_bps,
            'volatility_1m': self.volatility_1m,
            'volatility_5m': self.volatility_5m,
            'volatility_ratio': self.volatility_ratio,
            'spread_bps': self.spread_bps,
            'orderbook_depth': self.orderbook_depth,
            'volume_ratio': self.volume_ratio,
            'trade_rate_hz': self.trade_rate_hz,
            'buy_sell_imbalance': self.buy_sell_imbalance,
            'large_trade_frequency': self.large_trade_frequency,
            'price_jump_sigma': self.price_jump_sigma,
            'volume_spike_ratio': self.volume_spike_ratio,
            'spread_spike_ratio': self.spread_spike_ratio,
        }
