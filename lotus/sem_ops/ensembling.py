"""
Ensembling strategies for test-time scaling in semantic operations.

This module provides various ensembling strategies that can be used to improve
the accuracy and robustness of semantic operators by combining multiple samples
from the language model.

Strategies implemented:
    - majority_vote: Takes the most common result across samples
    - weighted_average: Weighs predictions by confidence scores
    - consensus: Returns result only if all samples agree
    - confidence_threshold: Uses majority vote with minimum confidence

Example:
    >>> from lotus.sem_ops.ensembling import Ensemble
    >>> ensemble = Ensemble(strategy='majority_vote', n_samples=3)
    >>> results = ensemble.aggregate(sample_outputs)
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EnsembleStrategy(Enum):
    """Available ensembling strategies for test-time scaling."""
    
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS = "consensus"
    CONFIDENCE_THRESHOLD = "confidence_threshold"


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble-based test-time scaling.
    
    Attributes:
        n_samples: Number of samples to generate for each input.
        strategy: The ensembling strategy to use.
        temperature: Sampling temperature for the language model.
        confidence_threshold: Minimum confidence required for confidence-based strategies.
    """
    
    n_samples: int = 3
    strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE
    temperature: float = 1.0
    confidence_threshold: float = 0.6


def majority_vote(samples: list[Any]) -> Any:
    """
    Return the most common value from a list of samples.
    
    Uses Counter to find the mode. In case of ties, returns the first
    value that reaches the maximum count (deterministic ordering).
    
    Args:
        samples: List of sample predictions to aggregate.
        
    Returns:
        The most frequently occurring value in the samples.
        
    Example:
        >>> majority_vote([True, True, False])
        True
        >>> majority_vote(['cat', 'dog', 'cat'])
        'cat'
    """
    if not samples:
        raise ValueError("Cannot compute majority vote on empty list")
    
    counter = Counter(samples)
    return counter.most_common(1)[0][0]


def weighted_average(samples: list[bool], weights: list[float] | None = None) -> bool:
    """
    Compute a weighted vote for boolean predictions.
    
    For boolean outputs, calculates the weighted sum and returns True
    if the weighted proportion of True values exceeds 0.5.
    
    Args:
        samples: List of boolean predictions.
        weights: Optional list of weights for each sample. If None,
                 uniform weights are used.
                 
    Returns:
        True if weighted average exceeds 0.5, False otherwise.
        
    Example:
        >>> weighted_average([True, True, False], [0.8, 0.6, 0.4])
        True
    """
    if not samples:
        raise ValueError("Cannot compute weighted average on empty list")
    
    if weights is None:
        weights = [1.0] * len(samples)
    
    if len(samples) != len(weights):
        raise ValueError("Samples and weights must have the same length")
    
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    
    weighted_sum = sum(w * (1.0 if s else 0.0) for s, w in zip(samples, weights))
    return weighted_sum / total_weight > 0.5


def consensus(samples: list[Any], default: Any = None) -> Any:
    """
    Return the result only if all samples agree.
    
    Provides high confidence results by requiring unanimous agreement.
    Returns the default value if samples disagree.
    
    Args:
        samples: List of sample predictions.
        default: Value to return if no consensus is reached.
        
    Returns:
        The unanimous value if all samples agree, otherwise the default.
        
    Example:
        >>> consensus([True, True, True])
        True
        >>> consensus([True, True, False], default=None)
        None
    """
    if not samples:
        raise ValueError("Cannot compute consensus on empty list")
    
    first_value = samples[0]
    if all(s == first_value for s in samples):
        return first_value
    return default


def confidence_threshold(
    samples: list[Any], 
    threshold: float = 0.6
) -> tuple[Any, float]:
    """
    Use majority vote with confidence tracking.
    
    Returns the majority vote result along with the confidence score,
    which is the proportion of samples that agree with the result.
    
    Args:
        samples: List of sample predictions.
        threshold: Minimum proportion required for confidence.
        
    Returns:
        Tuple of (result, confidence). If confidence is below threshold,
        the result may be less reliable.
        
    Example:
        >>> confidence_threshold([True, True, False])
        (True, 0.667)
    """
    if not samples:
        raise ValueError("Cannot compute confidence on empty list")
    
    counter = Counter(samples)
    most_common_value, count = counter.most_common(1)[0]
    confidence = count / len(samples)
    
    return most_common_value, confidence


class Ensemble:
    """
    Manages test-time scaling through ensemble predictions.
    
    This class provides a unified interface for applying various ensembling
    strategies to improve the accuracy of semantic operator predictions.
    
    Attributes:
        config: Configuration object with ensemble parameters.
        
    Example:
        >>> config = EnsembleConfig(n_samples=5, strategy=EnsembleStrategy.MAJORITY_VOTE)
        >>> ensemble = Ensemble(config)
        >>> samples = [True, True, True, False, True]
        >>> result = ensemble.aggregate(samples)
        >>> print(result)  # True
    """
    
    def __init__(self, config: EnsembleConfig | None = None):
        """
        Initialize the ensemble with the given configuration.
        
        Args:
            config: Ensemble configuration. If None, uses default settings.
        """
        self.config = config or EnsembleConfig()
    
    def aggregate(
        self, 
        samples: list[Any], 
        weights: list[float] | None = None,
        default: Any = None
    ) -> Any:
        """
        Aggregate multiple samples using the configured strategy.
        
        Args:
            samples: List of predictions to aggregate.
            weights: Optional weights for weighted strategies.
            default: Default value for consensus strategy.
            
        Returns:
            The aggregated prediction result.
            
        Raises:
            ValueError: If the configured strategy is not recognized.
        """
        strategy = self.config.strategy
        
        if strategy == EnsembleStrategy.MAJORITY_VOTE:
            return majority_vote(samples)
            
        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            if not all(isinstance(s, bool) for s in samples):
                # Fall back to majority vote for non-boolean types
                return majority_vote(samples)
            return weighted_average(samples, weights)
            
        elif strategy == EnsembleStrategy.CONSENSUS:
            return consensus(samples, default=default)
            
        elif strategy == EnsembleStrategy.CONFIDENCE_THRESHOLD:
            result, confidence = confidence_threshold(
                samples, 
                threshold=self.config.confidence_threshold
            )
            return result
            
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")
    
    def aggregate_batch(
        self, 
        batch_samples: list[list[Any]], 
        weights: list[list[float]] | None = None,
        default: Any = None
    ) -> list[Any]:
        """
        Aggregate samples for a batch of inputs.
        
        Args:
            batch_samples: List of sample lists, one per input.
            weights: Optional weights for each sample in each input.
            default: Default value for consensus strategy.
            
        Returns:
            List of aggregated predictions, one per input.
        """
        results = []
        for i, samples in enumerate(batch_samples):
            sample_weights = weights[i] if weights else None
            results.append(self.aggregate(samples, sample_weights, default))
        return results
