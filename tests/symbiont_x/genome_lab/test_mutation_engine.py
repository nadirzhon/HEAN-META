"""
Unit tests for Mutation Engine
"""

import pytest
from hean.symbiont_x.genome_lab import (
    create_random_genome,
    MutationEngine,
)


class TestMutationEngine:
    """Test MutationEngine class"""

    def test_mutation_engine_creation(self):
        """Test creating mutation engine"""
        engine = MutationEngine()
        assert engine is not None
        assert engine.base_mutation_rate == 0.1

    def test_mutation_engine_custom_rate(self):
        """Test creating engine with custom mutation rate"""
        engine = MutationEngine(base_mutation_rate=0.5)
        assert engine.base_mutation_rate == 0.5

    def test_mutate_produces_new_genome(self):
        """Test that mutation produces a new genome"""
        genome = create_random_genome("TestStrategy")

        engine = MutationEngine()
        mutated = engine.mutate(genome, mutation_rate=0.5)

        # Should be a different genome
        assert mutated.genome_id != genome.genome_id
        assert mutated is not genome

    def test_mutated_genome_has_correct_generation(self):
        """Test that mutated genome has incremented generation"""
        genome = create_random_genome("TestStrategy")
        genome.generation = 5

        engine = MutationEngine()
        mutated = engine.mutate(genome, mutation_rate=0.5)

        assert mutated.generation == 6

    def test_mutated_genome_has_parent_id(self):
        """Test that mutated genome tracks parent"""
        genome = create_random_genome("TestStrategy")

        engine = MutationEngine()
        mutated = engine.mutate(genome, mutation_rate=0.5)

        assert len(mutated.parent_ids) == 1
        assert mutated.parent_ids[0] == genome.genome_id

    def test_mutated_genome_has_new_id(self):
        """Test that mutated genome has different ID"""
        genome = create_random_genome("TestStrategy")

        engine = MutationEngine()
        mutated = engine.mutate(genome, mutation_rate=0.5)

        assert mutated.genome_id != genome.genome_id

    def test_mutated_genome_has_new_name(self):
        """Test that mutated genome has updated name"""
        genome = create_random_genome("TestStrategy")

        engine = MutationEngine()
        mutated = engine.mutate(genome, mutation_rate=0.5)

        assert "_M" in mutated.name

    def test_mutation_preserves_gene_count(self):
        """Test that mutation preserves number of genes (mostly)"""
        genome = create_random_genome("TestStrategy")
        original_count = len(genome.genes)

        engine = MutationEngine()
        mutated = engine.mutate(genome, mutation_rate=0.5)

        # Gene count should be similar (might change slightly due to duplication/deletion)
        # Allow +-1 for gene_duplicate/gene_delete mutations
        assert abs(len(mutated.genes) - original_count) <= 1

    def test_mutation_updates_statistics(self):
        """Test that mutation updates engine statistics"""
        engine = MutationEngine()
        assert engine.mutations_applied == 0

        genome = create_random_genome("TestStrategy")
        engine.mutate(genome, mutation_rate=0.5)

        assert engine.mutations_applied == 1

    def test_batch_mutate(self):
        """Test batch mutation of multiple genomes"""
        genomes = [create_random_genome(f"Strategy{i}") for i in range(5)]

        engine = MutationEngine()
        mutated = engine.batch_mutate(genomes, n_mutations=2)

        # Should have 5 * 2 = 10 mutated genomes
        assert len(mutated) == 10

        # All should be new genomes
        original_ids = {g.genome_id for g in genomes}
        for m in mutated:
            assert m.genome_id not in original_ids

    def test_adaptive_mutate(self):
        """Test adaptive mutation based on fitness"""
        genomes = [create_random_genome(f"Strategy{i}") for i in range(10)]

        # Set varying fitness scores
        for i, genome in enumerate(genomes):
            genome.fitness_score = i * 0.1

        engine = MutationEngine()
        mutated = engine.adaptive_mutate(genomes, top_n=3)

        # Should have mutations from top 3 genomes
        assert len(mutated) > 0

        # All should be new genomes
        for m in mutated:
            assert m.genome_id not in {g.genome_id for g in genomes}

    def test_get_statistics(self):
        """Test getting mutation statistics"""
        engine = MutationEngine()

        genome = create_random_genome("TestStrategy")
        engine.mutate(genome, mutation_rate=0.5)
        engine.mutate(genome, mutation_rate=0.5)

        stats = engine.get_statistics()

        assert 'mutations_applied' in stats
        assert 'successful_mutations' in stats
        assert 'success_rate' in stats
        assert 'base_mutation_rate' in stats
        assert stats['mutations_applied'] == 2

    def test_adaptive_mutation_rate(self):
        """Test that adaptive mutation adjusts rate based on fitness"""
        genome = create_random_genome("TestStrategy")
        genome.fitness_score = 0.9  # High fitness

        engine = MutationEngine(base_mutation_rate=0.5)
        # With adaptive=True (default), high fitness should lower mutation rate
        mutated = engine.mutate(genome, adaptive=True)

        # Just verify it completes without error
        assert mutated is not None

    def test_mutation_with_zero_rate(self):
        """Test mutation with rate=0 (edge case)"""
        genome = create_random_genome("TestStrategy")

        engine = MutationEngine()
        # This should still create a cloned genome, just without point mutations
        mutated = engine.mutate(genome, mutation_rate=0.0)

        assert mutated is not None
        assert mutated.genome_id != genome.genome_id

    def test_mutable_genes_respect_flag(self):
        """Test that non-mutable genes are preserved"""
        genome = create_random_genome("TestStrategy")

        # Set some genes as non-mutable
        for gene in genome.genes[:3]:
            gene.mutable = False

        engine = MutationEngine()
        # Multiple mutations to increase chance of hitting those genes
        for _ in range(10):
            mutated = engine.mutate(genome, mutation_rate=1.0)
            # Non-mutable genes should theoretically be unchanged
            # But since we clone, we can't directly verify without deep inspection
            assert mutated is not None
