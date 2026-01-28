"""
Tests for sem_filter ensembling integration.

Tests the integration of the Ensemble class with sem_filter operator,
including n_sample parameter, different strategies, and output format.
"""


import pytest

from lotus.sem_ops.ensembling import (
    Ensemble,
    EnsembleConfig,
    EnsembleStrategy,
)
from lotus.types import RawOutputs, SemanticFilterOutput


class TestSemanticFilterOutputProperties:
    """Test SemanticFilterOutput backward compatibility properties."""
    
    def test_raw_outputs_single_sample(self):
        """Test raw_outputs property returns single items when n_sample=1."""
        raw = RawOutputs(
            predictions=[[True], [False], [True]],
            raw_outputs=[["yes"], ["no"], ["yes"]],
            explanations=[["reason1"], ["reason2"], ["reason3"]],
            logprobs=None,
        )
        output = SemanticFilterOutput(
            outputs=[True, False, True],
            _raw_outputs=raw,
        )
        
        assert output.raw_outputs == ["yes", "no", "yes"]
        assert output.explanations == ["reason1", "reason2", "reason3"]
        assert output.logprobs is None
    
    def test_raw_outputs_multiple_samples(self):
        """Test raw_outputs property with multiple samples (returns first)."""
        raw = RawOutputs(
            predictions=[[True, True, False], [False, False, True]],
            raw_outputs=[["yes", "yes", "no"], ["no", "no", "yes"]],
            explanations=[["r1", "r2", "r3"], ["r4", "r5", "r6"]],
            logprobs=None,
        )
        output = SemanticFilterOutput(
            outputs=[True, False],  # Aggregated results
            _raw_outputs=raw,
        )
        
        # Should return first run's data for backward compatibility
        assert output.raw_outputs == ["yes", "no"]
        assert output.explanations == ["r1", "r4"]


class TestEnsembleConfigIntegration:
    """Test EnsembleConfig usage in real scenarios."""
    
    def test_default_config(self):
        """Test default EnsembleConfig values."""
        config = EnsembleConfig()
        assert config.strategy == EnsembleStrategy.MAJORITY_VOTE
        assert config.weights is None
        assert config.default is None
        assert config.confidence_threshold == 0.6
    
    def test_config_with_weights(self):
        """Test EnsembleConfig with weighted average settings."""
        config = EnsembleConfig(
            strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
            weights=[0.5, 0.3, 0.2],
        )
        ensemble = Ensemble(config)
        
        # Test aggregation uses config weights
        result = ensemble.aggregate([True, True, False])
        assert result is True  # 0.5 + 0.3 = 0.8 > 0.5 threshold
    
    def test_config_with_consensus_default(self):
        """Test EnsembleConfig with consensus and default value."""
        config = EnsembleConfig(
            strategy=EnsembleStrategy.CONSENSUS,
            default=False,
        )
        ensemble = Ensemble(config)
        
        # No consensus -> should return default
        result = ensemble.aggregate([True, False, True])
        assert result is False
        
        # Consensus -> should return the agreed value
        result = ensemble.aggregate([True, True, True])
        assert result is True


class TestEnsembleBatchAggregation:
    """Test batch aggregation for sem_filter integration."""
    
    def test_batch_majority_vote(self):
        """Test batch aggregation with majority vote."""
        config = EnsembleConfig(strategy=EnsembleStrategy.MAJORITY_VOTE)
        ensemble = Ensemble(config)
        
        # Simulate outputs from 3 samples for 4 documents
        batch_samples = [
            [True, True, False],   # Doc 0: 2/3 True -> True
            [False, False, False], # Doc 1: 0/3 True -> False
            [True, False, True],   # Doc 2: 2/3 True -> True
            [False, True, False],  # Doc 3: 1/3 True -> False
        ]
        
        results = ensemble.aggregate_batch(batch_samples)
        assert results == [True, False, True, False]
    
    def test_batch_consensus(self):
        """Test batch aggregation with consensus strategy."""
        config = EnsembleConfig(
            strategy=EnsembleStrategy.CONSENSUS,
            default=None,
        )
        ensemble = Ensemble(config)
        
        batch_samples = [
            [True, True, True],    # Doc 0: unanimous True
            [False, False, True],  # Doc 1: no consensus -> None
            [False, False, False], # Doc 2: unanimous False
        ]
        
        results = ensemble.aggregate_batch(batch_samples)
        assert results == [True, None, False]


class TestRawOutputsStructure:
    """Test RawOutputs dataclass structure."""
    
    def test_raw_outputs_creation(self):
        """Test creating RawOutputs with all fields."""
        raw = RawOutputs(
            predictions=[[True, False], [False, True]],
            raw_outputs=[["yes", "no"], ["no", "yes"]],
            explanations=[["r1", "r2"], ["r3", "r4"]],
            logprobs=[[[{"token": "yes", "logprob": -0.1}]], [[{"token": "no", "logprob": -0.2}]]],
        )
        
        assert len(raw.predictions) == 2
        assert len(raw.predictions[0]) == 2
        assert raw.raw_outputs[0][0] == "yes"
        assert raw.explanations[1][1] == "r4"
    
    def test_raw_outputs_without_logprobs(self):
        """Test RawOutputs with None logprobs."""
        raw = RawOutputs(
            predictions=[[True]],
            raw_outputs=[["yes"]],
            explanations=[["reason"]],
            logprobs=None,
        )
        
        assert raw.logprobs is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
