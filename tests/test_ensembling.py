"""
Tests for the ensembling module.

This module contains comprehensive tests for all ensembling strategies
used in test-time scaling for semantic operations.
"""

import pytest

from lotus.sem_ops.ensembling import (
    Ensemble,
    EnsembleConfig,
    EnsembleStrategy,
    confidence_threshold,
    consensus,
    majority_vote,
    weighted_average,
)


class TestMajorityVote:
    """Tests for the majority_vote function."""

    def test_basic_boolean_majority(self):
        """Should return the most common boolean value."""
        samples = [True, True, False]
        assert majority_vote(samples) is True

    def test_all_same_value(self):
        """Should return the unanimous value."""
        samples = [True, True, True]
        assert majority_vote(samples) is True

    def test_string_values(self):
        """Should work with string values."""
        samples = ["cat", "dog", "cat", "bird", "cat"]
        assert majority_vote(samples) == "cat"

    def test_tie_returns_first_most_common(self):
        """In case of tie, should return deterministic result."""
        samples = [True, False]
        result = majority_vote(samples)
        assert result in [True, False]

    def test_single_element(self):
        """Should handle single element lists."""
        assert majority_vote([True]) is True
        assert majority_vote(["value"]) == "value"

    def test_empty_list_raises_error(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot compute majority vote"):
            majority_vote([])


class TestWeightedAverage:
    """Tests for the weighted_average function."""

    def test_uniform_weights(self):
        """With uniform weights, should behave like majority vote."""
        samples = [True, True, False]
        assert weighted_average(samples) is True

    def test_weighted_towards_true(self):
        """Higher weights on True should return True."""
        samples = [True, False, False]
        weights = [0.9, 0.1, 0.1]
        assert weighted_average(samples, weights) is True

    def test_weighted_towards_false(self):
        """Higher weights on False should return False."""
        samples = [True, True, False]
        weights = [0.1, 0.1, 0.9]
        assert weighted_average(samples, weights) is False

    def test_no_weights_uses_uniform(self):
        """If weights are None, should use uniform weights."""
        samples = [True, True, False, False, True]
        assert weighted_average(samples) is True

    def test_mismatched_lengths_raises_error(self):
        """Should raise ValueError if lengths don't match."""
        with pytest.raises(ValueError, match="same length"):
            weighted_average([True, False], [0.5])

    def test_empty_list_raises_error(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot compute"):
            weighted_average([])

    def test_zero_total_weight_raises_error(self):
        """Should raise ValueError if total weight is zero."""
        with pytest.raises(ValueError, match="cannot be zero"):
            weighted_average([True, False], [0.0, 0.0])


class TestConsensus:
    """Tests for the consensus function."""

    def test_unanimous_true(self):
        """Should return True when all samples are True."""
        samples = [True, True, True]
        assert consensus(samples) is True

    def test_unanimous_false(self):
        """Should return False when all samples are False."""
        samples = [False, False, False]
        assert consensus(samples) is False

    def test_no_consensus_returns_default(self):
        """Should return default when samples disagree."""
        samples = [True, True, False]
        assert consensus(samples, default=None) is None

    def test_custom_default_value(self):
        """Should return custom default when specified."""
        samples = [True, False]
        assert consensus(samples, default="inconclusive") == "inconclusive"

    def test_string_consensus(self):
        """Should work with string values."""
        samples = ["yes", "yes", "yes"]
        assert consensus(samples) == "yes"

    def test_single_element_consensus(self):
        """Single element should always have consensus."""
        assert consensus([True]) is True
        assert consensus(["value"]) == "value"

    def test_empty_list_raises_error(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot compute consensus"):
            consensus([])


class TestConfidenceThreshold:
    """Tests for the confidence_threshold function."""

    def test_high_confidence(self):
        """Should return high confidence for unanimous samples."""
        samples = [True, True, True]
        result, confidence = confidence_threshold(samples)
        assert result is True
        assert confidence == 1.0

    def test_moderate_confidence(self):
        """Should calculate correct confidence for mixed samples."""
        samples = [True, True, False]
        result, confidence = confidence_threshold(samples)
        assert result is True
        assert abs(confidence - 0.667) < 0.01

    def test_low_confidence(self):
        """Should handle low confidence scenarios."""
        samples = [True, False]
        result, confidence = confidence_threshold(samples)
        assert confidence == 0.5

    def test_empty_list_raises_error(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot compute confidence"):
            confidence_threshold([])


class TestEnsembleConfig:
    """Tests for EnsembleConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = EnsembleConfig()
        assert config.n_samples == 3
        assert config.strategy == EnsembleStrategy.MAJORITY_VOTE
        assert config.temperature == 1.0
        assert config.confidence_threshold == 0.6

    def test_custom_values(self):
        """Should accept custom values."""
        config = EnsembleConfig(
            n_samples=5,
            strategy=EnsembleStrategy.CONSENSUS,
            temperature=0.8,
            confidence_threshold=0.75
        )
        assert config.n_samples == 5
        assert config.strategy == EnsembleStrategy.CONSENSUS


class TestEnsemble:
    """Tests for the Ensemble class."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        ensemble = Ensemble()
        assert ensemble.config.strategy == EnsembleStrategy.MAJORITY_VOTE

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = EnsembleConfig(strategy=EnsembleStrategy.CONSENSUS)
        ensemble = Ensemble(config)
        assert ensemble.config.strategy == EnsembleStrategy.CONSENSUS

    def test_aggregate_majority_vote(self):
        """Should aggregate using majority vote."""
        ensemble = Ensemble(EnsembleConfig(strategy=EnsembleStrategy.MAJORITY_VOTE))
        result = ensemble.aggregate([True, True, False])
        assert result is True

    def test_aggregate_weighted_average(self):
        """Should aggregate using weighted average for booleans."""
        ensemble = Ensemble(EnsembleConfig(strategy=EnsembleStrategy.WEIGHTED_AVERAGE))
        result = ensemble.aggregate([True, False, False], weights=[0.9, 0.1, 0.1])
        assert result is True

    def test_aggregate_consensus(self):
        """Should aggregate using consensus."""
        ensemble = Ensemble(EnsembleConfig(strategy=EnsembleStrategy.CONSENSUS))
        
        # Unanimous case
        result = ensemble.aggregate([True, True, True])
        assert result is True
        
        # No consensus case
        result = ensemble.aggregate([True, True, False], default=None)
        assert result is None

    def test_aggregate_confidence_threshold(self):
        """Should aggregate using confidence threshold."""
        config = EnsembleConfig(
            strategy=EnsembleStrategy.CONFIDENCE_THRESHOLD,
            confidence_threshold=0.6
        )
        ensemble = Ensemble(config)
        result = ensemble.aggregate([True, True, False])
        assert result is True

    def test_aggregate_batch(self):
        """Should aggregate multiple inputs in batch."""
        ensemble = Ensemble()
        batch = [
            [True, True, False],
            [False, False, True],
            [True, True, True]
        ]
        results = ensemble.aggregate_batch(batch)
        assert results == [True, False, True]

    def test_weighted_average_fallback_for_non_boolean(self):
        """Weighted average should fall back to majority vote for non-booleans."""
        ensemble = Ensemble(EnsembleConfig(strategy=EnsembleStrategy.WEIGHTED_AVERAGE))
        result = ensemble.aggregate(["cat", "cat", "dog"])
        assert result == "cat"


class TestEnsembleStrategy:
    """Tests for EnsembleStrategy enum."""

    def test_strategy_values(self):
        """Should have correct string values."""
        assert EnsembleStrategy.MAJORITY_VOTE.value == "majority_vote"
        assert EnsembleStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert EnsembleStrategy.CONSENSUS.value == "consensus"
        assert EnsembleStrategy.CONFIDENCE_THRESHOLD.value == "confidence_threshold"
