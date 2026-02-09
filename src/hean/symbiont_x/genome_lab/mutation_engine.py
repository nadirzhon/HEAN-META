"""
Mutation Engine - Двигатель мутаций

Создаёт вариации стратегий через мутации генов
"""

import random
import time

from .genome_types import Gene, GeneType, StrategyGenome


class MutationEngine:
    """
    Двигатель мутаций

    Применяет различные типы мутаций к геномам
    """

    def __init__(self, base_mutation_rate: float = 0.1):
        self.base_mutation_rate = base_mutation_rate

        # Mutation statistics
        self.mutations_applied = 0
        self.successful_mutations = 0

    def mutate(
        self,
        genome: StrategyGenome,
        mutation_rate: float | None = None,
        adaptive: bool = True
    ) -> StrategyGenome:
        """
        Мутирует геном

        Args:
            genome: Геном для мутации
            mutation_rate: Кастомная частота мутаций (или base)
            adaptive: Адаптивная мутация на основе fitness

        Returns:
            Новый мутировавший геном
        """

        if mutation_rate is None:
            mutation_rate = self.base_mutation_rate

        # Adaptive mutation: higher rate for low fitness
        if adaptive and genome.fitness_score > 0:
            # Low fitness → higher mutation rate
            # High fitness → lower mutation rate
            fitness_factor = max(0.1, 1.0 - genome.fitness_score)
            mutation_rate = mutation_rate * fitness_factor

        # Clone and mutate
        mutated = genome.clone()
        mutated.name = f"{genome.name}_M{mutated.generation}"

        # Apply mutations
        mutation_type = self._select_mutation_type()

        if mutation_type == 'point':
            mutated = self._point_mutation(mutated, mutation_rate)
        elif mutation_type == 'gene_swap':
            mutated = self._gene_swap_mutation(mutated)
        elif mutation_type == 'gene_duplicate':
            mutated = self._gene_duplication(mutated)
        elif mutation_type == 'gene_delete':
            mutated = self._gene_deletion(mutated)

        self.mutations_applied += 1
        mutated.last_updated_ns = time.time_ns()

        return mutated

    def _select_mutation_type(self) -> str:
        """Выбирает тип мутации"""
        mutation_types = [
            ('point', 0.7),          # 70% - point mutations
            ('gene_swap', 0.15),     # 15% - gene swap
            ('gene_duplicate', 0.10), # 10% - gene duplication
            ('gene_delete', 0.05),   # 5% - gene deletion
        ]

        r = random.random()
        cumulative = 0.0

        for mut_type, prob in mutation_types:
            cumulative += prob
            if r <= cumulative:
                return mut_type

        return 'point'

    def _point_mutation(self, genome: StrategyGenome, mutation_rate: float) -> StrategyGenome:
        """
        Point mutation - изменение значений генов

        Самый частый тип мутации
        """

        for i, gene in enumerate(genome.genes):
            if random.random() < mutation_rate and gene.mutable:
                mutated_gene = gene.mutate(mutation_rate)
                genome.genes[i] = mutated_gene

        return genome

    def _gene_swap_mutation(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Gene swap - замена гена на другой из того же типа

        Например, смена entry signal с 'momentum' на 'breakout'
        """

        # Find genes with allowed_values (categorical genes)
        categorical_genes = [
            (i, g) for i, g in enumerate(genome.genes)
            if g.allowed_values and len(g.allowed_values) > 1
        ]

        if categorical_genes:
            idx, gene = random.choice(categorical_genes)

            # Pick different value
            new_value = random.choice([v for v in gene.allowed_values if v != gene.value])

            new_gene = Gene(
                gene_id=gene.gene_id,
                gene_type=gene.gene_type,
                name=gene.name,
                value=new_value,
                mutable=gene.mutable,
                min_value=gene.min_value,
                max_value=gene.max_value,
                allowed_values=gene.allowed_values,
            )

            genome.genes[idx] = new_gene

        return genome

    def _gene_duplication(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Gene duplication - дублирование гена

        Может создавать более сложные стратегии
        """

        # Pick random gene to duplicate
        if genome.genes:
            gene_to_duplicate = random.choice(genome.genes)

            # Clone gene with new ID
            import uuid
            duplicated_gene = Gene(
                gene_id=str(uuid.uuid4()),
                gene_type=gene_to_duplicate.gene_type,
                name=f"{gene_to_duplicate.name}_dup",
                value=gene_to_duplicate.value,
                mutable=gene_to_duplicate.mutable,
                min_value=gene_to_duplicate.min_value,
                max_value=gene_to_duplicate.max_value,
                allowed_values=gene_to_duplicate.allowed_values,
            )

            genome.genes.append(duplicated_gene)

        return genome

    def _gene_deletion(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Gene deletion - удаление гена

        Упрощает стратегию (может улучшить робастность)
        """

        # Don't delete if too few genes
        if len(genome.genes) <= 5:
            return genome

        # Find non-critical genes (not entry/exit signals)
        deletable_genes = [
            (i, g) for i, g in enumerate(genome.genes)
            if g.gene_type not in [GeneType.ENTRY_SIGNAL, GeneType.EXIT_SIGNAL]
        ]

        if deletable_genes:
            idx, _ = random.choice(deletable_genes)
            del genome.genes[idx]

        return genome

    def batch_mutate(
        self,
        genomes: list[StrategyGenome],
        n_mutations: int = 1,
        mutation_rate: float | None = None
    ) -> list[StrategyGenome]:
        """
        Создаёт мутации для списка геномов

        Args:
            genomes: Список геномов
            n_mutations: Сколько мутаций создать для каждого
            mutation_rate: Частота мутаций

        Returns:
            Список новых мутировавших геномов
        """

        mutated_genomes = []

        for genome in genomes:
            for _ in range(n_mutations):
                mutated = self.mutate(genome, mutation_rate=mutation_rate)
                mutated_genomes.append(mutated)

        return mutated_genomes

    def adaptive_mutate(
        self,
        population: list[StrategyGenome],
        top_n: int = 5
    ) -> list[StrategyGenome]:
        """
        Адаптивная мутация популяции

        Мутирует топ-N лучших геномов с адаптивными коэффициентами
        """

        # Sort by fitness
        sorted_population = sorted(
            population,
            key=lambda g: g.fitness_score,
            reverse=True
        )

        # Take top N
        top_genomes = sorted_population[:top_n]

        # Mutate each with adaptive rate
        mutated = []
        for genome in top_genomes:
            # Better genomes get more conservative mutations
            fitness_rank = sorted_population.index(genome) + 1
            mutation_rate = self.base_mutation_rate * (fitness_rank / top_n)

            # Create 2-3 mutations per top genome
            n_mutations = 3 if fitness_rank <= 2 else 2

            for _ in range(n_mutations):
                mut = self.mutate(genome, mutation_rate=mutation_rate, adaptive=True)
                mutated.append(mut)

        return mutated

    def get_statistics(self) -> dict:
        """Возвращает статистику мутаций"""
        success_rate = (
            self.successful_mutations / self.mutations_applied
            if self.mutations_applied > 0 else 0
        )

        return {
            'mutations_applied': self.mutations_applied,
            'successful_mutations': self.successful_mutations,
            'success_rate': success_rate,
            'base_mutation_rate': self.base_mutation_rate,
        }
