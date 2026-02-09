"""
Crossover Engine - Двигатель скрещивания

Комбинирует два родительских генома в потомка
"""

import random
import time
import uuid

from .genome_types import Gene, StrategyGenome


class CrossoverEngine:
    """
    Двигатель скрещивания геномов

    Создаёт новые стратегии комбинируя лучшие черты родителей
    """

    def __init__(self):
        # Crossover statistics
        self.crossovers_performed = 0
        self.successful_crossovers = 0

    def crossover(
        self,
        parent1: StrategyGenome,
        parent2: StrategyGenome,
        method: str = 'uniform'
    ) -> StrategyGenome:
        """
        Скрещивает два генома

        Args:
            parent1: Первый родитель
            parent2: Второй родитель
            method: Метод скрещивания ('uniform', 'single_point', 'two_point', 'best_of_each')

        Returns:
            Новый геном-потомок
        """

        if method == 'uniform':
            child = self._uniform_crossover(parent1, parent2)
        elif method == 'single_point':
            child = self._single_point_crossover(parent1, parent2)
        elif method == 'two_point':
            child = self._two_point_crossover(parent1, parent2)
        elif method == 'best_of_each':
            child = self._best_of_each_crossover(parent1, parent2)
        else:
            child = self._uniform_crossover(parent1, parent2)

        # Set metadata
        child.genome_id = str(uuid.uuid4())
        child.parent_ids = [parent1.genome_id, parent2.genome_id]
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.name = f"Cross_G{child.generation}_{parent1.name[:5]}x{parent2.name[:5]}"
        child.created_at_ns = time.time_ns()
        child.fitness_score = 0.0
        child.alive = True

        self.crossovers_performed += 1

        return child

    def _uniform_crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """
        Uniform crossover - каждый ген выбирается случайно от одного из родителей

        50/50 шанс для каждого гена
        """

        # Create mapping of gene types to genes
        p1_genes = {(g.gene_type, g.name): g for g in parent1.genes}
        p2_genes = {(g.gene_type, g.name): g for g in parent2.genes}

        # Get all unique gene keys
        all_keys = set(p1_genes.keys()) | set(p2_genes.keys())

        child_genes = []

        for key in all_keys:
            # If both parents have this gene, randomly choose
            if key in p1_genes and key in p2_genes:
                chosen_gene = random.choice([p1_genes[key], p2_genes[key]])
            # If only one parent has it, take from that parent
            elif key in p1_genes:
                chosen_gene = p1_genes[key]
            else:
                chosen_gene = p2_genes[key]

            # Clone gene
            child_genes.append(self._clone_gene(chosen_gene))

        child = parent1.clone()
        child.genes = child_genes

        return child

    def _single_point_crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """
        Single-point crossover - разрез в одной точке

        Гены до точки от parent1, после - от parent2
        """

        # Choose crossover point
        min_len = min(len(parent1.genes), len(parent2.genes))
        if min_len <= 1:
            return self._uniform_crossover(parent1, parent2)

        crossover_point = random.randint(1, min_len - 1)

        # Combine genes
        child_genes = []

        # Take from parent1 up to crossover point
        child_genes.extend([self._clone_gene(g) for g in parent1.genes[:crossover_point]])

        # Take from parent2 after crossover point
        child_genes.extend([self._clone_gene(g) for g in parent2.genes[crossover_point:]])

        child = parent1.clone()
        child.genes = child_genes

        return child

    def _two_point_crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """
        Two-point crossover - два разреза

        P1 | P2 | P1
        """

        min_len = min(len(parent1.genes), len(parent2.genes))
        if min_len <= 2:
            return self._uniform_crossover(parent1, parent2)

        # Choose two crossover points
        point1 = random.randint(1, min_len - 2)
        point2 = random.randint(point1 + 1, min_len - 1)

        child_genes = []

        # First segment from parent1
        child_genes.extend([self._clone_gene(g) for g in parent1.genes[:point1]])

        # Middle segment from parent2
        child_genes.extend([self._clone_gene(g) for g in parent2.genes[point1:point2]])

        # Last segment from parent1
        child_genes.extend([self._clone_gene(g) for g in parent1.genes[point2:]])

        child = parent1.clone()
        child.genes = child_genes

        return child

    def _best_of_each_crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """
        Best-of-each crossover - выбирает лучшие гены от каждого родителя

        Основан на fitness родителей
        """

        # Create mapping
        p1_genes = {(g.gene_type, g.name): g for g in parent1.genes}
        p2_genes = {(g.gene_type, g.name): g for g in parent2.genes}

        all_keys = set(p1_genes.keys()) | set(p2_genes.keys())

        child_genes = []

        # Choose based on parent fitness
        for key in all_keys:
            if key in p1_genes and key in p2_genes:
                # Both have it - choose from better parent
                if parent1.fitness_score >= parent2.fitness_score:
                    chosen_gene = p1_genes[key]
                else:
                    chosen_gene = p2_genes[key]
            elif key in p1_genes:
                chosen_gene = p1_genes[key]
            else:
                chosen_gene = p2_genes[key]

            child_genes.append(self._clone_gene(chosen_gene))

        child = parent1.clone()
        child.genes = child_genes

        return child

    def _clone_gene(self, gene: Gene) -> Gene:
        """Клонирует ген с новым ID"""
        return Gene(
            gene_id=str(uuid.uuid4()),
            gene_type=gene.gene_type,
            name=gene.name,
            value=gene.value,
            mutable=gene.mutable,
            min_value=gene.min_value,
            max_value=gene.max_value,
            allowed_values=gene.allowed_values,
        )

    def batch_crossover(
        self,
        population: list[StrategyGenome],
        n_offspring: int,
        selection_method: str = 'fitness_weighted'
    ) -> list[StrategyGenome]:
        """
        Создаёт несколько потомков из популяции

        Args:
            population: Популяция геномов
            n_offspring: Количество потомков
            selection_method: Метод отбора родителей

        Returns:
            Список новых геномов
        """

        offspring = []

        for _ in range(n_offspring):
            # Select parents
            parent1, parent2 = self._select_parents(population, selection_method)

            # Crossover
            child = self.crossover(parent1, parent2, method='uniform')
            offspring.append(child)

        return offspring

    def _select_parents(
        self,
        population: list[StrategyGenome],
        method: str
    ) -> tuple[StrategyGenome, StrategyGenome]:
        """
        Выбирает двух родителей из популяции

        Args:
            population: Популяция
            method: Метод отбора ('random', 'fitness_weighted', 'tournament')

        Returns:
            (parent1, parent2)
        """

        if method == 'random':
            return random.sample(population, 2)

        elif method == 'fitness_weighted':
            # Weight by fitness
            total_fitness = sum(g.fitness_score for g in population)

            if total_fitness == 0:
                # If no fitness yet, random
                return random.sample(population, 2)

            # Weighted selection
            weights = [g.fitness_score / total_fitness for g in population]

            parent1 = random.choices(population, weights=weights, k=1)[0]

            # Select second parent (different from first)
            remaining = [g for g in population if g.genome_id != parent1.genome_id]
            if not remaining:
                return parent1, parent1  # Edge case

            remaining_weights = [
                g.fitness_score / sum(r.fitness_score for r in remaining)
                for g in remaining
            ]

            parent2 = random.choices(remaining, weights=remaining_weights, k=1)[0]

            return parent1, parent2

        elif method == 'tournament':
            # Tournament selection (pick best of N random)
            tournament_size = min(5, len(population))

            tournament1 = random.sample(population, tournament_size)
            parent1 = max(tournament1, key=lambda g: g.fitness_score)

            tournament2 = random.sample(population, tournament_size)
            parent2 = max(tournament2, key=lambda g: g.fitness_score)

            return parent1, parent2

        else:
            return random.sample(population, 2)

    def get_statistics(self) -> dict:
        """Возвращает статистику скрещиваний"""
        success_rate = (
            self.successful_crossovers / self.crossovers_performed
            if self.crossovers_performed > 0 else 0
        )

        return {
            'crossovers_performed': self.crossovers_performed,
            'successful_crossovers': self.successful_crossovers,
            'success_rate': success_rate,
        }
