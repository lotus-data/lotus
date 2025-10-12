# BioDEX Benchmark

This directory contains scripts for benchmarking adverse drug reaction (ADR) detection using the BioDEX dataset, as described in the Lotus paper.

## Overview

The BioDEX benchmark evaluates different pipeline approaches for identifying adverse drug reactions from medical article abstracts. The benchmark uses the BioDEX-Reactions dataset and implements various semantic operations including mapping, filtering, and join cascades.

## Files

- **`biodex_tester.py`** - Main benchmark script containing pipeline implementations and test harness
- **`metrics.py`** - Evaluation metrics including precision, recall, rank precision, and F1 score
- **`pipeline_tester.py`** - Base classes for pipeline testing framework

## Prerequisites

Before running the scripts, ensure you have:

1. **Lotus installed** - The main Lotus library with semantic operations
2. **Required dependencies**:
   ```bash
   pip install datasets pandas numpy scipy
   ```
3. **API keys configured** - The scripts use GPT-4o-mini by default, so ensure your OpenAI API key is set
4. **BioDEX dataset access** - The scripts automatically download from HuggingFace: `BioDEX/BioDEX-Reactions`

## Data Requirements

The benchmark requires two additional data files that should be placed in the same directory:

- **`biodex-reactions.csv`** - Corpus of adverse drug reactions
- **`biodex_prompts.py`** - Contains `map_dem_df` with few-shot examples for mapping operations

## Running the Benchmark

### Basic Usage

```bash
cd benchmark/biodex
python biodex_tester.py
```

### Configuration

The main configuration is in the `BiodexTester.set_configs()` method:

```python
# Default configuration uses GPT-4o-mini
lm = LM(model="gpt-4o-mini-2024-07-18",
        max_batch_size=64,
        temperature=0.0,
        max_tokens=256)

# Embedding model
rm = SentenceTransformersRM(model="intfloat/e5-base-v2", max_batch_size=4)
```

### Pipeline Options

The script includes several pipeline implementations:

1. **Retrieve** - Basic semantic similarity join
2. **MapRetrieve** - Map patient descriptions to reactions, then retrieve
3. **MapRetrieveFilter** - Add filtering step to map-retrieve pipeline  
4. **RetrieveFilter** - Retrieve candidates then filter with LLM
5. **JoinCascade** - Optimized cascade with approximation techniques
6. **ApproxMapRetrieveFilter** - Approximate version with quantile scoring
7. **ApproxRetrieveFilter** - Approximate retrieve-filter with thresholds

### Sample Parameters

```python
# Test with different recall/precision targets
ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9))
ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=250))

# Test approximate methods with thresholds
ts.add_pipeline(ApproxRetrieveFilter(t_pos=1.0, t_neg=0.995))
```

## Key Parameters

- **`n_samples`** - Number of test samples (default: 250, full dataset: 4249)
- **`truncation_limit`** - Maximum characters for patient descriptions (default: 8000)
- **`recall_target`/`precision_target`** - Target metrics for cascade optimization
- **`t_pos`/`t_neg`** - Thresholds for approximate filtering methods
- **`range`** - Maximum sample range for importance sampling

## Output

Results are saved to the `biodex_results/` directory with the following structure:

```
biodex_results/
├── PipelineName/
│   └── nsamples=250_param1=value1_param2=value2/
│       ├── res.csv          # Detailed results per query
│       ├── metrics.csv      # Average metrics
│       ├── std.csv          # Standard deviations  
│       ├── latency.json     # Timing information
│       └── configs.json     # Model configurations
```

## Evaluation Metrics

The benchmark computes several metrics:

- **Rank Precision@K** - Precision considering ranking order
- **Recall@K** - Standard recall at different cutoffs (5, 10, 20, 50, 100, 200, 300, 400, 500)
- **Precision@K** - Standard precision at cutoffs (5, 10)
- **Latency** - End-to-end pipeline execution time

## Example Results

To view a summary of results:

```python
# Print summary for JoinCascade pipelines
print(ts.summarize_pipeline_results("biodex_results", 250, ["JoinCascade"]))
```

## Paper Results

The results from these scripts were used to generate the BioDEX benchmark results in the Lotus paper, demonstrating the effectiveness of semantic cascade operations for adverse drug reaction detection tasks.

## Notes

- The scripts include extensive approximation and optimization techniques for large-scale evaluation
- Importance sampling is used to reduce computational costs while maintaining statistical guarantees
- The benchmark supports both exact and approximate evaluation modes
- Results include confidence intervals and statistical corrections for precision/recall estimates
