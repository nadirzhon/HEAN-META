"""
Decision Analyzer - Анализатор решений

Анализирует качество решений и находит паттерны
"""

import statistics

from .decision_types import Decision, DecisionType
from .ledger import DecisionLedger


class DecisionAnalyzer:
    """
    Анализатор решений

    Находит успешные/провальные паттерны
    """

    def __init__(self, ledger: DecisionLedger):
        self.ledger = ledger

    def analyze_strategy(self, strategy_id: str) -> dict:
        """Анализирует решения стратегии"""

        decisions = self.ledger.get_decisions(
            strategy_id=strategy_id,
            limit=10000
        )

        if not decisions:
            return {}

        # Overall metrics
        total_decisions = len(decisions)
        completed_decisions = [d for d in decisions if d.is_completed()]

        if not completed_decisions:
            return {
                'strategy_id': strategy_id,
                'total_decisions': total_decisions,
                'completed_decisions': 0,
            }

        # Success rate
        successful = [d for d in completed_decisions if d.is_successful()]
        success_rate = len(successful) / len(completed_decisions)

        # PnL analysis
        total_pnl = sum(d.pnl_impact for d in completed_decisions)
        winning_pnl = sum(d.pnl_impact for d in completed_decisions if d.pnl_impact > 0)
        losing_pnl = sum(d.pnl_impact for d in completed_decisions if d.pnl_impact < 0)

        profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else 0

        # Latency analysis
        latencies = [d.get_latency_ms() for d in completed_decisions]
        avg_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0

        # Confidence analysis
        avg_confidence = statistics.mean([d.confidence for d in decisions]) if decisions else 0

        # By decision type
        by_type = {}
        for dtype in DecisionType:
            type_decisions = [d for d in decisions if d.decision_type == dtype]
            if type_decisions:
                type_success = [d for d in type_decisions if d.is_successful()]
                by_type[dtype.value] = {
                    'count': len(type_decisions),
                    'success_rate': len(type_success) / len(type_decisions),
                }

        return {
            'strategy_id': strategy_id,
            'total_decisions': total_decisions,
            'completed_decisions': len(completed_decisions),
            'success_rate': success_rate,
            'total_pnl': total_pnl,
            'winning_pnl': winning_pnl,
            'losing_pnl': losing_pnl,
            'profit_factor': profit_factor,
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'avg_confidence': avg_confidence,
            'decisions_by_type': by_type,
        }

    def find_best_decisions(
        self,
        min_pnl: float = 0,
        min_confidence: float = 0.7,
        limit: int = 100
    ) -> list[Decision]:
        """Находит лучшие решения"""

        decisions = self.ledger.get_recent_decisions(n=10000)

        # Filter
        good_decisions = [
            d for d in decisions
            if d.is_successful() and
            d.pnl_impact >= min_pnl and
            d.confidence >= min_confidence
        ]

        # Sort by PnL
        good_decisions = sorted(
            good_decisions,
            key=lambda d: d.pnl_impact,
            reverse=True
        )

        return good_decisions[:limit]

    def find_worst_decisions(
        self,
        max_pnl: float = 0,
        limit: int = 100
    ) -> list[Decision]:
        """Находит худшие решения"""

        decisions = self.ledger.get_recent_decisions(n=10000)

        # Filter
        bad_decisions = [
            d for d in decisions
            if d.pnl_impact <= max_pnl
        ]

        # Sort by PnL (worst first)
        bad_decisions = sorted(
            bad_decisions,
            key=lambda d: d.pnl_impact
        )

        return bad_decisions[:limit]

    def find_decision_patterns(
        self,
        pattern_type: str = "winning"
    ) -> dict:
        """
        Находит паттерны в решениях

        Returns общие характеристики успешных/провальных решений
        """

        decisions = self.ledger.get_recent_decisions(n=10000)

        if pattern_type == "winning":
            target_decisions = [d for d in decisions if d.is_successful() and d.pnl_impact > 0]
        else:
            target_decisions = [d for d in decisions if d.pnl_impact < 0]

        if not target_decisions:
            return {}

        # Common regimes
        regimes = [d.market_regime for d in target_decisions if d.market_regime]
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Most common regime
        most_common_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else None

        # Average confidence
        avg_confidence = statistics.mean([d.confidence for d in target_decisions])

        # Common symbols
        symbols = [d.symbol for d in target_decisions if d.symbol]
        symbol_counts = {}
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return {
            'pattern_type': pattern_type,
            'sample_size': len(target_decisions),
            'most_common_regime': most_common_regime,
            'regime_distribution': regime_counts,
            'avg_confidence': avg_confidence,
            'symbol_distribution': symbol_counts,
        }

    def compare_strategies(
        self,
        strategy_ids: list[str]
    ) -> dict:
        """Сравнивает несколько стратегий"""

        comparisons = {}

        for strategy_id in strategy_ids:
            analysis = self.analyze_strategy(strategy_id)
            comparisons[strategy_id] = analysis

        # Rank by success rate
        ranked_by_success = sorted(
            comparisons.items(),
            key=lambda x: x[1].get('success_rate', 0),
            reverse=True
        )

        # Rank by PnL
        ranked_by_pnl = sorted(
            comparisons.items(),
            key=lambda x: x[1].get('total_pnl', 0),
            reverse=True
        )

        return {
            'comparisons': comparisons,
            'ranked_by_success_rate': [(sid, data['success_rate']) for sid, data in ranked_by_success],
            'ranked_by_pnl': [(sid, data['total_pnl']) for sid, data in ranked_by_pnl],
        }

    def get_decision_timeline(
        self,
        strategy_id: str | None = None,
        limit: int = 1000
    ) -> list[dict]:
        """
        Возвращает timeline решений

        Для визуализации
        """

        if strategy_id:
            decisions = self.ledger.get_decisions(strategy_id=strategy_id, limit=limit)
        else:
            decisions = self.ledger.get_recent_decisions(n=limit)

        timeline = []

        for decision in decisions:
            timeline.append({
                'timestamp_ns': decision.decided_at_ns,
                'decision_id': decision.decision_id,
                'decision_type': decision.decision_type.value,
                'outcome': decision.outcome.value,
                'pnl_impact': decision.pnl_impact,
                'confidence': decision.confidence,
                'market_regime': decision.market_regime,
            })

        return sorted(timeline, key=lambda x: x['timestamp_ns'])
