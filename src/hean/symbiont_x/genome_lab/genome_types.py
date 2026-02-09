"""
Strategy Genome - Геном торговой стратегии

Представление стратегии как набора генов
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GeneType(Enum):
    """Типы генов"""

    # Entry genes
    ENTRY_SIGNAL = "entry_signal"
    ENTRY_FILTER = "entry_filter"
    ENTRY_TIMING = "entry_timing"

    # Exit genes
    EXIT_SIGNAL = "exit_signal"
    EXIT_TIMING = "exit_timing"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

    # Position sizing
    POSITION_SIZE = "position_size"
    LEVERAGE = "leverage"
    RISK_PER_TRADE = "risk_per_trade"

    # Regime adaptation
    REGIME_FILTER = "regime_filter"
    REGIME_MULTIPLIER = "regime_multiplier"

    # Market selection
    MARKET_FILTER = "market_filter"
    TIMEFRAME = "timeframe"

    # Parameters
    INDICATOR_PARAM = "indicator_param"
    THRESHOLD = "threshold"


@dataclass
class Gene:
    """
    Ген - единица наследуемой информации

    Каждый ген кодирует один аспект стратегии
    """

    gene_id: str
    gene_type: GeneType
    name: str
    value: Any

    # Metadata
    mutable: bool = True  # Может ли мутировать
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None

    def mutate(self, mutation_rate: float = 0.1) -> 'Gene':
        """Мутация гена"""
        if not self.mutable:
            return self

        import random

        # Numerical mutation
        if isinstance(self.value, (int, float)):
            if self.min_value is not None and self.max_value is not None:
                # Add random noise
                range_size = self.max_value - self.min_value
                noise = random.gauss(0, range_size * mutation_rate)
                new_value = self.value + noise

                # Clamp to range
                new_value = max(self.min_value, min(self.max_value, new_value))

                return Gene(
                    gene_id=self.gene_id,
                    gene_type=self.gene_type,
                    name=self.name,
                    value=new_value,
                    mutable=self.mutable,
                    min_value=self.min_value,
                    max_value=self.max_value,
                    allowed_values=self.allowed_values,
                )

        # Categorical mutation
        elif self.allowed_values:
            if random.random() < mutation_rate:
                new_value = random.choice(self.allowed_values)
                return Gene(
                    gene_id=self.gene_id,
                    gene_type=self.gene_type,
                    name=self.name,
                    value=new_value,
                    mutable=self.mutable,
                    min_value=self.min_value,
                    max_value=self.max_value,
                    allowed_values=self.allowed_values,
                )

        return self

    def to_dict(self) -> dict:
        """Сериализация"""
        return {
            'gene_id': self.gene_id,
            'gene_type': self.gene_type.value,
            'name': self.name,
            'value': self.value,
            'mutable': self.mutable,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'allowed_values': self.allowed_values,
        }


@dataclass
class StrategyGenome:
    """
    Геном стратегии

    Полное представление торговой стратегии как набора генов
    """

    genome_id: str
    name: str
    genes: list[Gene]

    # Lineage
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    # Performance tracking
    fitness_score: float = 0.0
    trades_executed: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Metadata
    created_at_ns: int = 0
    last_updated_ns: int = 0
    alive: bool = True

    def get_gene(self, gene_type: GeneType, name: str | None = None) -> Gene | None:
        """Получить ген по типу (и опционально имени)"""
        for gene in self.genes:
            if gene.gene_type == gene_type:
                if name is None or gene.name == name:
                    return gene
        return None

    def get_genes_by_type(self, gene_type: GeneType) -> list[Gene]:
        """Получить все гены определённого типа"""
        return [g for g in self.genes if g.gene_type == gene_type]

    def set_gene(self, gene: Gene):
        """Установить или обновить ген"""
        for i, g in enumerate(self.genes):
            if g.gene_id == gene.gene_id:
                self.genes[i] = gene
                return

        # If not found, add
        self.genes.append(gene)

    def clone(self) -> 'StrategyGenome':
        """Клонирование генома"""
        import copy
        import time

        new_genome = copy.deepcopy(self)
        new_genome.genome_id = str(uuid.uuid4())
        new_genome.parent_ids = [self.genome_id]
        new_genome.generation = self.generation + 1
        new_genome.created_at_ns = time.time_ns()
        new_genome.fitness_score = 0.0
        new_genome.trades_executed = 0
        new_genome.alive = True

        return new_genome

    def mutate(self, mutation_rate: float = 0.1) -> 'StrategyGenome':
        """Мутация генома"""
        mutated = self.clone()
        mutated.genes = [g.mutate(mutation_rate) for g in mutated.genes]
        mutated.name = f"{self.name}_M{mutated.generation}"
        return mutated

    def get_genome_hash(self) -> str:
        """Хэш генома (для уникальности)"""
        genome_str = json.dumps(
            [g.to_dict() for g in self.genes],
            sort_keys=True
        )
        return hashlib.sha256(genome_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Сериализация"""
        return {
            'genome_id': self.genome_id,
            'name': self.name,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'genes': [g.to_dict() for g in self.genes],
            'fitness_score': self.fitness_score,
            'trades_executed': self.trades_executed,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'created_at_ns': self.created_at_ns,
            'last_updated_ns': self.last_updated_ns,
            'alive': self.alive,
            'genome_hash': self.get_genome_hash(),
        }

    def to_strategy_config(self) -> dict:
        """
        Конвертация генома в конфигурацию стратегии

        Готово для исполнения
        """
        config = {
            'strategy_id': self.genome_id,
            'name': self.name,
            'generation': self.generation,
        }

        # Entry configuration
        entry_signal = self.get_gene(GeneType.ENTRY_SIGNAL)
        if entry_signal:
            config['entry_signal'] = entry_signal.value

        entry_filter = self.get_gene(GeneType.ENTRY_FILTER)
        if entry_filter:
            config['entry_filter'] = entry_filter.value

        # Exit configuration
        exit_signal = self.get_gene(GeneType.EXIT_SIGNAL)
        if exit_signal:
            config['exit_signal'] = exit_signal.value

        stop_loss = self.get_gene(GeneType.STOP_LOSS)
        if stop_loss:
            config['stop_loss_pct'] = stop_loss.value

        take_profit = self.get_gene(GeneType.TAKE_PROFIT)
        if take_profit:
            config['take_profit_pct'] = take_profit.value

        # Position sizing
        position_size = self.get_gene(GeneType.POSITION_SIZE)
        if position_size:
            config['position_size'] = position_size.value

        leverage = self.get_gene(GeneType.LEVERAGE)
        if leverage:
            config['leverage'] = leverage.value

        risk_per_trade = self.get_gene(GeneType.RISK_PER_TRADE)
        if risk_per_trade:
            config['risk_per_trade_pct'] = risk_per_trade.value

        # Regime adaptation
        regime_filter = self.get_gene(GeneType.REGIME_FILTER)
        if regime_filter:
            config['allowed_regimes'] = regime_filter.value

        # Market selection
        market_filter = self.get_gene(GeneType.MARKET_FILTER)
        if market_filter:
            config['symbols'] = market_filter.value

        timeframe = self.get_gene(GeneType.TIMEFRAME)
        if timeframe:
            config['timeframe'] = timeframe.value

        # Indicator parameters
        indicator_params = self.get_genes_by_type(GeneType.INDICATOR_PARAM)
        config['indicator_params'] = {g.name: g.value for g in indicator_params}

        # Thresholds
        thresholds = self.get_genes_by_type(GeneType.THRESHOLD)
        config['thresholds'] = {g.name: g.value for g in thresholds}

        return config

    @staticmethod
    def from_dict(data: dict) -> 'StrategyGenome':
        """Десериализация"""
        genes = [
            Gene(
                gene_id=g['gene_id'],
                gene_type=GeneType(g['gene_type']),
                name=g['name'],
                value=g['value'],
                mutable=g.get('mutable', True),
                min_value=g.get('min_value'),
                max_value=g.get('max_value'),
                allowed_values=g.get('allowed_values'),
            )
            for g in data['genes']
        ]

        return StrategyGenome(
            genome_id=data['genome_id'],
            name=data['name'],
            genes=genes,
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            fitness_score=data.get('fitness_score', 0.0),
            trades_executed=data.get('trades_executed', 0),
            win_rate=data.get('win_rate', 0.0),
            sharpe_ratio=data.get('sharpe_ratio', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            created_at_ns=data.get('created_at_ns', 0),
            last_updated_ns=data.get('last_updated_ns', 0),
            alive=data.get('alive', True),
        )


def create_random_genome(name: str) -> StrategyGenome:
    """
    Создаёт случайный геном стратегии

    Полезно для начальной популяции
    """
    import random
    import time

    genes = []

    # Entry signal (random choice)
    entry_signals = ['momentum', 'mean_reversion', 'breakout', 'volume_spike']
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.ENTRY_SIGNAL,
        name='entry_signal',
        value=random.choice(entry_signals),
        allowed_values=entry_signals,
    ))

    # Exit signal
    exit_signals = ['profit_target', 'momentum_reverse', 'time_based', 'trailing_stop']
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.EXIT_SIGNAL,
        name='exit_signal',
        value=random.choice(exit_signals),
        allowed_values=exit_signals,
    ))

    # Stop loss (0.5% - 5%)
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.STOP_LOSS,
        name='stop_loss_pct',
        value=random.uniform(0.5, 5.0),
        min_value=0.5,
        max_value=5.0,
    ))

    # Take profit (1% - 10%)
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.TAKE_PROFIT,
        name='take_profit_pct',
        value=random.uniform(1.0, 10.0),
        min_value=1.0,
        max_value=10.0,
    ))

    # Position size (1% - 20% of capital)
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.POSITION_SIZE,
        name='position_size_pct',
        value=random.uniform(1.0, 20.0),
        min_value=1.0,
        max_value=20.0,
    ))

    # Leverage (1x - 5x)
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.LEVERAGE,
        name='leverage',
        value=random.uniform(1.0, 5.0),
        min_value=1.0,
        max_value=5.0,
    ))

    # Risk per trade (0.5% - 3%)
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.RISK_PER_TRADE,
        name='risk_per_trade_pct',
        value=random.uniform(0.5, 3.0),
        min_value=0.5,
        max_value=3.0,
    ))

    # Timeframe
    timeframes = ['1m', '5m', '15m', '1h']
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.TIMEFRAME,
        name='timeframe',
        value=random.choice(timeframes),
        allowed_values=timeframes,
    ))

    # Indicator parameters (example: EMA period)
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.INDICATOR_PARAM,
        name='ema_period',
        value=random.randint(5, 50),
        min_value=5,
        max_value=50,
    ))

    # Threshold (example: momentum threshold)
    genes.append(Gene(
        gene_id=str(uuid.uuid4()),
        gene_type=GeneType.THRESHOLD,
        name='momentum_threshold',
        value=random.uniform(0.01, 0.1),
        min_value=0.01,
        max_value=0.1,
    ))

    return StrategyGenome(
        genome_id=str(uuid.uuid4()),
        name=name,
        genes=genes,
        generation=0,
        parent_ids=[],
        created_at_ns=time.time_ns(),
        alive=True,
    )
