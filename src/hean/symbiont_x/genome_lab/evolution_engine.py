"""
Evolution Engine - Двигатель эволюции

Координирует весь процесс эволюции: популяция, селекция, мутации, скрещивание
"""

import time
from collections import deque
from collections.abc import Callable

from .crossover import CrossoverEngine
from .genome_types import StrategyGenome, create_random_genome
from .mutation_engine import MutationEngine


class EvolutionEngine:
    """
    Двигатель эволюции стратегий

    Управляет популяцией, применяет естественный отбор
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.3,
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Population
        self.population: list[StrategyGenome] = []
        self.generation_number = 0

        # Engines
        self.mutation_engine = MutationEngine(base_mutation_rate=mutation_rate)
        self.crossover_engine = CrossoverEngine()

        # History
        self.generation_history = deque(maxlen=100)
        self.best_genome_ever: StrategyGenome | None = None

        # Statistics
        self.total_genomes_created = 0
        self.total_genomes_killed = 0

    def initialize_population(self, base_name: str = "Strategy") -> list[StrategyGenome]:
        """
        Инициализирует начальную популяцию

        Создаёт случайные геномы
        """

        self.population = []

        for i in range(self.population_size):
            genome = create_random_genome(f"{base_name}_{i}")
            self.population.append(genome)

        self.total_genomes_created += self.population_size
        self.generation_number = 0

        return self.population

    def evolve_generation(
        self,
        fitness_evaluator: Callable[[StrategyGenome], float] | None = None
    ) -> list[StrategyGenome]:
        """
        Эволюционирует одно поколение

        Args:
            fitness_evaluator: Функция для оценки fitness (если не уже установлен)

        Returns:
            Новая популяция
        """

        # Evaluate fitness if not already done
        if fitness_evaluator:
            for genome in self.population:
                if genome.fitness_score == 0:
                    genome.fitness_score = fitness_evaluator(genome)

        # Selection - выбираем выживших
        survivors = self._selection()

        # Elitism - сохраняем лучших без изменений
        elite = self._elitism(survivors)

        # Reproduction - создаём новых
        offspring = self._reproduction(survivors)

        # Combine into new population
        new_population = elite + offspring

        # Trim to population size
        new_population = new_population[:self.population_size]

        # Kill off old population that didn't make it
        old_ids = {g.genome_id for g in self.population}
        new_ids = {g.genome_id for g in new_population}
        killed_ids = old_ids - new_ids
        self.total_genomes_killed += len(killed_ids)

        # Update population
        self.population = new_population
        self.generation_number += 1

        # Record generation stats
        self._record_generation_stats()

        # Update best genome ever
        best_current = max(self.population, key=lambda g: g.fitness_score)
        if self.best_genome_ever is None or best_current.fitness_score > self.best_genome_ever.fitness_score:
            self.best_genome_ever = best_current

        return self.population

    def _selection(self) -> list[StrategyGenome]:
        """
        Селекция - выбирает выживших

        Использует fitness-proportionate selection
        """

        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)

        # Select top 50% + some random from bottom 50%
        n_top = int(self.population_size * 0.5)
        n_random = int(self.population_size * 0.2)

        survivors = sorted_pop[:n_top]

        # Add some random from bottom (diversity)
        if len(sorted_pop) > n_top:
            import random
            bottom_half = sorted_pop[n_top:]
            random_survivors = random.sample(
                bottom_half,
                min(n_random, len(bottom_half))
            )
            survivors.extend(random_survivors)

        return survivors

    def _elitism(self, survivors: list[StrategyGenome]) -> list[StrategyGenome]:
        """
        Элитизм - сохраняет лучших без изменений

        Returns elite копии (без мутаций)
        """

        sorted_survivors = sorted(survivors, key=lambda g: g.fitness_score, reverse=True)
        elite = sorted_survivors[:self.elite_size]

        # Clone elite
        elite_clones = []
        for genome in elite:
            clone = genome.clone()
            clone.genome_id = genome.genome_id  # Keep same ID (это elite)
            clone.name = f"{genome.name}_Elite"
            elite_clones.append(clone)

        return elite_clones

    def _reproduction(self, survivors: list[StrategyGenome]) -> list[StrategyGenome]:
        """
        Репродукция - создаёт новых потомков

        Комбинация мутаций и скрещивания
        """

        offspring = []

        # How many offspring to create
        n_offspring_needed = self.population_size - self.elite_size

        # Calculate split between mutation and crossover
        n_crossover = int(n_offspring_needed * self.crossover_rate)
        n_mutation = n_offspring_needed - n_crossover

        # Create offspring via crossover
        if n_crossover > 0:
            crossover_offspring = self.crossover_engine.batch_crossover(
                population=survivors,
                n_offspring=n_crossover,
                selection_method='fitness_weighted'
            )
            offspring.extend(crossover_offspring)

        # Create offspring via mutation
        if n_mutation > 0:
            mutation_offspring = self.mutation_engine.adaptive_mutate(
                population=survivors,
                top_n=min(10, len(survivors))
            )

            # Take first n_mutation
            offspring.extend(mutation_offspring[:n_mutation])

        self.total_genomes_created += len(offspring)

        return offspring

    def inject_genome(self, genome: StrategyGenome):
        """
        Инъекция нового генома в популяцию

        Полезно для добавления вручную созданных стратегий
        """

        # Replace worst performer
        if len(self.population) >= self.population_size:
            worst = min(self.population, key=lambda g: g.fitness_score)
            self.population.remove(worst)
            self.total_genomes_killed += 1

        self.population.append(genome)
        self.total_genomes_created += 1

    def kill_genome(self, genome_id: str):
        """Убивает конкретный геном"""
        self.population = [g for g in self.population if g.genome_id != genome_id]
        self.total_genomes_killed += 1

    def get_top_genomes(self, n: int = 10) -> list[StrategyGenome]:
        """Возвращает топ N геномов по fitness"""
        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        return sorted_pop[:n]

    def get_genome_by_id(self, genome_id: str) -> StrategyGenome | None:
        """Находит геном по ID"""
        for genome in self.population:
            if genome.genome_id == genome_id:
                return genome
        return None

    def _record_generation_stats(self):
        """Записывает статистику поколения"""
        if not self.population:
            return

        fitness_scores = [g.fitness_score for g in self.population]

        import statistics

        stats = {
            'generation': self.generation_number,
            'timestamp_ns': time.time_ns(),
            'population_size': len(self.population),
            'avg_fitness': statistics.mean(fitness_scores),
            'max_fitness': max(fitness_scores),
            'min_fitness': min(fitness_scores),
            'median_fitness': statistics.median(fitness_scores),
            'fitness_stdev': statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0,
            'best_genome_id': max(self.population, key=lambda g: g.fitness_score).genome_id,
        }

        self.generation_history.append(stats)

    def get_statistics(self) -> dict:
        """Возвращает статистику эволюции"""
        if not self.population:
            return {}

        current_fitness = [g.fitness_score for g in self.population]
        import statistics

        stats = {
            'generation_number': self.generation_number,
            'population_size': len(self.population),
            'total_genomes_created': self.total_genomes_created,
            'total_genomes_killed': self.total_genomes_killed,
            'current_avg_fitness': statistics.mean(current_fitness),
            'current_max_fitness': max(current_fitness),
            'best_genome_ever': self.best_genome_ever.to_dict() if self.best_genome_ever else None,
            'mutation_stats': self.mutation_engine.get_statistics(),
            'crossover_stats': self.crossover_engine.get_statistics(),
        }

        # Add generation history
        if self.generation_history:
            stats['generation_history'] = list(self.generation_history)

        return stats

    def get_diversity_score(self) -> float:
        """
        Вычисляет diversity score популяции

        0.0 = все одинаковые, 1.0 = все разные
        """

        if len(self.population) < 2:
            return 0.0

        # Count unique genome hashes
        unique_hashes = len({g.get_genome_hash() for g in self.population})

        diversity = unique_hashes / len(self.population)
        return diversity

    def save_population(self, filepath: str):
        """Сохраняет популяцию в файл"""
        import json

        data = {
            'generation_number': self.generation_number,
            'population': [g.to_dict() for g in self.population],
            'stats': self.get_statistics(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_population(self, filepath: str):
        """Загружает популяцию из файла"""
        import json

        from .genome_types import StrategyGenome

        with open(filepath) as f:
            data = json.load(f)

        self.generation_number = data['generation_number']
        self.population = [
            StrategyGenome.from_dict(g) for g in data['population']
        ]

    def reset(self):
        """Сброс эволюции"""
        self.population = []
        self.generation_number = 0
        self.generation_history.clear()
        self.best_genome_ever = None
        self.total_genomes_created = 0
        self.total_genomes_killed = 0
