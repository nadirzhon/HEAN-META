"""
Unit tests for Crossover Engine
"""

import pytest
from hean.symbiont_x.genome_lab import (
    create_random_genome,
    CrossoverEngine,
)


class TestCrossoverEngine:
    """Test CrossoverEngine class"""

    def test_crossover_engine_creation(self):
        """Test creating crossover engine"""
        engine = CrossoverEngine()
        assert engine is not None
        assert engine.crossovers_performed == 0

    def test_uniform_crossover(self):
        """Test uniform crossover (default)"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2, method='uniform')

        # Child should exist and have genes
        assert child is not None
        assert len(child.genes) > 0

        # Child should have both parents' IDs
        assert len(child.parent_ids) == 2
        assert parent1.genome_id in child.parent_ids
        assert parent2.genome_id in child.parent_ids

    def test_single_point_crossover(self):
        """Test single-point crossover"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2, method='single_point')

        assert child is not None
        assert len(child.genes) > 0

    def test_two_point_crossover(self):
        """Test two-point crossover"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2, method='two_point')

        assert child is not None
        assert len(child.genes) > 0

    def test_best_of_each_crossover(self):
        """Test best-of-each crossover"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")
        parent1.fitness_score = 0.8
        parent2.fitness_score = 0.6

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2, method='best_of_each')

        assert child is not None
        assert len(child.genes) > 0

    def test_crossover_increments_generation(self):
        """Test that crossover increments generation"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        parent1.generation = 5
        parent2.generation = 5

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2)

        assert child.generation == 6

    def test_crossover_with_different_generations(self):
        """Test crossover with parents from different generations"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        parent1.generation = 5
        parent2.generation = 7

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2)

        # Child generation should be max(parent generations) + 1
        assert child.generation == 8

    def test_crossover_child_has_new_id(self):
        """Test that child has a new unique ID"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2)

        assert child.genome_id != parent1.genome_id
        assert child.genome_id != parent2.genome_id

    def test_crossover_child_has_new_name(self):
        """Test that child has a descriptive name"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2)

        assert "Cross" in child.name

    def test_crossover_resets_fitness(self):
        """Test that child fitness is reset"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")
        parent1.fitness_score = 0.9
        parent2.fitness_score = 0.8

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2)

        assert child.fitness_score == 0.0

    def test_crossover_updates_statistics(self):
        """Test that crossover updates engine statistics"""
        engine = CrossoverEngine()
        assert engine.crossovers_performed == 0

        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")
        engine.crossover(parent1, parent2)

        assert engine.crossovers_performed == 1

    def test_multiple_crossovers_produce_different_children(self):
        """Test that multiple crossovers can produce different children"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()

        child1 = engine.crossover(parent1, parent2, method='uniform')
        child2 = engine.crossover(parent1, parent2, method='uniform')

        # Children should have different IDs
        assert child1.genome_id != child2.genome_id

    def test_batch_crossover(self):
        """Test batch crossover from population"""
        population = [create_random_genome(f"Strategy{i}") for i in range(10)]

        # Set fitness scores
        for i, genome in enumerate(population):
            genome.fitness_score = i * 0.1

        engine = CrossoverEngine()
        offspring = engine.batch_crossover(population, n_offspring=5)

        assert len(offspring) == 5
        for child in offspring:
            assert child.genome_id not in {g.genome_id for g in population}

    def test_batch_crossover_with_fitness_weighted_selection(self):
        """Test batch crossover with fitness-weighted parent selection"""
        population = [create_random_genome(f"Strategy{i}") for i in range(10)]

        # Set varying fitness scores
        for i, genome in enumerate(population):
            genome.fitness_score = (i + 1) * 0.1

        engine = CrossoverEngine()
        offspring = engine.batch_crossover(
            population,
            n_offspring=3,
            selection_method='fitness_weighted'
        )

        assert len(offspring) == 3

    def test_batch_crossover_with_tournament_selection(self):
        """Test batch crossover with tournament parent selection"""
        population = [create_random_genome(f"Strategy{i}") for i in range(10)]

        for i, genome in enumerate(population):
            genome.fitness_score = i * 0.1

        engine = CrossoverEngine()
        offspring = engine.batch_crossover(
            population,
            n_offspring=3,
            selection_method='tournament'
        )

        assert len(offspring) == 3

    def test_get_statistics(self):
        """Test getting crossover statistics"""
        engine = CrossoverEngine()

        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")
        engine.crossover(parent1, parent2)
        engine.crossover(parent1, parent2)

        stats = engine.get_statistics()

        assert 'crossovers_performed' in stats
        assert 'successful_crossovers' in stats
        assert 'success_rate' in stats
        assert stats['crossovers_performed'] == 2

    def test_crossover_child_is_alive(self):
        """Test that child genome is marked as alive"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()
        child = engine.crossover(parent1, parent2)

        assert child.alive is True

    def test_crossover_default_method(self):
        """Test that default crossover method works"""
        parent1 = create_random_genome("Parent1")
        parent2 = create_random_genome("Parent2")

        engine = CrossoverEngine()
        # No method specified - should default to 'uniform'
        child = engine.crossover(parent1, parent2)

        assert child is not None
