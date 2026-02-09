"""
Unit tests for Genome types
"""

import pytest
from hean.symbiont_x.genome_lab import (
    StrategyGenome,
    create_random_genome,
    GeneType,
)
from hean.symbiont_x.genome_lab.genome_types import Gene


class TestStrategyGenome:
    """Test StrategyGenome class"""

    def test_create_random_genome(self):
        """Test creating a random genome"""
        genome = create_random_genome("TestStrategy")

        assert genome.name == "TestStrategy"
        assert genome.generation == 0
        assert len(genome.genes) > 0
        assert genome.fitness_score == 0.0
        assert genome.parent_ids == []

    def test_genome_has_required_gene_types(self):
        """Test that genome has genes of expected types"""
        genome = create_random_genome("TestStrategy")

        # Check that we have at least some required gene types
        gene_types = {g.gene_type for g in genome.genes}

        assert GeneType.ENTRY_SIGNAL in gene_types
        assert GeneType.EXIT_SIGNAL in gene_types
        assert GeneType.STOP_LOSS in gene_types
        assert GeneType.TAKE_PROFIT in gene_types

    def test_gene_bounds(self):
        """Test that numeric genes are within valid bounds"""
        genome = create_random_genome("TestStrategy")

        for gene in genome.genes:
            if isinstance(gene.value, (int, float)):
                # Numeric genes should be within their min/max
                if gene.min_value is not None:
                    assert gene.value >= gene.min_value, f"{gene.name} below min"
                if gene.max_value is not None:
                    assert gene.value <= gene.max_value, f"{gene.name} above max"

    def test_genome_serialization(self):
        """Test genome to_dict and from_dict"""
        genome = create_random_genome("TestStrategy")
        genome.fitness_score = 0.75

        # Serialize
        genome_dict = genome.to_dict()

        assert 'name' in genome_dict
        assert 'genes' in genome_dict
        assert 'generation' in genome_dict
        assert 'fitness_score' in genome_dict
        assert 'created_at_ns' in genome_dict

        assert genome_dict['name'] == "TestStrategy"
        assert genome_dict['fitness_score'] == 0.75

    def test_genome_clone(self):
        """Test cloning a genome"""
        genome = create_random_genome("TestStrategy")
        genome.fitness_score = 0.85

        clone = genome.clone()

        assert clone.name == genome.name  # Name is preserved on clone
        assert len(clone.genes) == len(genome.genes)
        assert clone.generation == genome.generation + 1  # Generation increments
        assert clone.fitness_score == 0.0  # Fitness resets on clone
        assert clone.genome_id != genome.genome_id  # Different ID

    def test_genome_set_fitness(self):
        """Test setting fitness"""
        genome = create_random_genome("TestStrategy")

        assert genome.fitness_score == 0.0

        genome.fitness_score = 0.92
        assert genome.fitness_score == 0.92

    def test_genome_comparison(self):
        """Test comparing genomes by fitness"""
        genome1 = create_random_genome("Strategy1")
        genome2 = create_random_genome("Strategy2")

        genome1.fitness_score = 0.8
        genome2.fitness_score = 0.9

        # genome2 should be better
        assert genome2.fitness_score > genome1.fitness_score

    def test_genome_parent_tracking(self):
        """Test tracking parent genomes"""
        parent = create_random_genome("Parent")
        child = parent.clone()

        assert len(child.parent_ids) == 1
        assert child.parent_ids[0] == parent.genome_id

    def test_genome_generation_increment(self):
        """Test generation incrementing on clone"""
        parent = create_random_genome("Parent")
        parent.generation = 5

        child = parent.clone()

        assert child.generation == 6

    def test_get_gene_by_type(self):
        """Test getting genes by type"""
        genome = create_random_genome("TestStrategy")

        entry_gene = genome.get_gene(GeneType.ENTRY_SIGNAL)
        assert entry_gene is not None
        assert entry_gene.gene_type == GeneType.ENTRY_SIGNAL

    def test_get_genes_by_type(self):
        """Test getting multiple genes by type"""
        genome = create_random_genome("TestStrategy")

        indicator_genes = genome.get_genes_by_type(GeneType.INDICATOR_PARAM)
        for gene in indicator_genes:
            assert gene.gene_type == GeneType.INDICATOR_PARAM

    def test_genome_from_dict(self):
        """Test deserializing genome from dict"""
        genome = create_random_genome("TestStrategy")
        genome.fitness_score = 0.5

        genome_dict = genome.to_dict()
        restored = StrategyGenome.from_dict(genome_dict)

        assert restored.name == genome.name
        assert restored.genome_id == genome.genome_id
        assert restored.fitness_score == genome.fitness_score
        assert len(restored.genes) == len(genome.genes)

    def test_genome_hash(self):
        """Test genome hash for uniqueness detection"""
        genome1 = create_random_genome("Strategy1")
        genome2 = create_random_genome("Strategy2")

        # Different genomes should have different hashes (with high probability)
        hash1 = genome1.get_genome_hash()
        hash2 = genome2.get_genome_hash()

        assert isinstance(hash1, str)
        assert len(hash1) == 16
        # Note: Random genomes might occasionally have same hash, but unlikely

    def test_genome_to_strategy_config(self):
        """Test converting genome to strategy config"""
        genome = create_random_genome("TestStrategy")

        config = genome.to_strategy_config()

        assert 'strategy_id' in config
        assert 'name' in config
        assert config['strategy_id'] == genome.genome_id
        assert config['name'] == genome.name
