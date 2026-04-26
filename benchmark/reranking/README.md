# Reranking Benchmark

This directory contains scripts for benchmarking semantic reranking operations on BEIR datasets, as described in the Lotus paper.

## Overview

The reranking benchmark evaluates different semantic top-k methods for document ranking tasks using standard BEIR datasets. It compares various approaches including LLM-based evaluation, quick approximation methods, and heap-based sorting.

## Files

- **`bench.py`** - Benchmark script for BEIR datasets (COVID, SciFact)
- **`bench_hs.py`** - Benchmark script for ranking tasks (CIFAR, HellaSwag accuracy ranking)

## Prerequisites

Before running the scripts, ensure you have:

1. **Lotus installed** - The main Lotus library with semantic operations
2. **Required dependencies**:
   ```bash
   pip install beir pandas numpy pickle-mixin
   ```
3. **API keys configured** - Scripts use Llama-3-70B by default (hosted VLLM), update API configuration as needed
4. **BEIR datasets** - Automatically downloaded by the BEIR library

## Running the Benchmarks

### BEIR Dataset Ranking (`bench.py`)

Evaluates document ranking on COVID and SciFact datasets:

```bash
cd benchmark/reranking
python bench.py --dataset scifact --initial_k 100
python bench.py --dataset covid --initial_k 100
```

**Parameters:**
- `--dataset`: Choose from `covid`, `scifact`
- `--initial_k`: Number of initial candidates to retrieve (default: 100)

**Methods tested:**
- `llm-eval`: Direct LLM evaluation of relevance scores

### Accuracy Ranking (`bench_hs.py`)

Evaluates ranking of research papers by reported accuracy metrics:

```bash
cd benchmark/reranking
python bench_hs.py --dataset cifar --trials 20
python bench_hs.py --dataset hellaswag --trials 20
```

**Parameters:**
- `--dataset`: Choose from `cifar`, `hellaswag`
- `--trials`: Number of evaluation trials (default: 20)

**Methods tested:**
- `llm-eval`: LLM-based accuracy evaluation
- `quick`: Quick approximation method
- `quick-sem`: Semantic quick method
- `heap`: Heap-based sorting
- `naive`: Naive baseline approach

## Configuration

### Model Configuration

Both scripts use similar model configurations:

```python
# Retrieval model
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

# Language model (update API details as needed)
lm = LM(
    model="hosted_vllm/meta-llama/Meta-Llama-3-70B-Instruct",
    api_key="sk-xx",  # Update with your API key
    api_base="http://localhost:8200/v1/",  # Update with your endpoint
    temperature=0.0,  # bench.py uses 0.0, bench_hs.py uses 0.7
)
```

### Data Requirements

**For `bench_hs.py`:**
- `data/cifar.csv` - CIFAR-10 research papers with accuracy metrics
- `data/hellaswag.csv` - HellaSwag research papers with accuracy metrics
- `indexes/` directory for semantic indexes (auto-created)

**For `bench.py`:**
- BEIR datasets are automatically downloaded to `data/` directory

## Evaluation Metrics

### BEIR Datasets (`bench.py`)
- **nDCG@10** - Normalized Discounted Cumulative Gain at rank 10
- **Latency** - Query processing time
- **Token Usage** - Total tokens consumed by LLM
- **LLM Calls** - Number of LLM API calls

### Accuracy Ranking (`bench_hs.py`)
- **nDCG@10** - Normalized DCG for accuracy ranking
- **Recall@10** - Recall of top-10 most accurate papers
- **Latency** - Processing time per query
- **Token Usage** - LLM token consumption
- **LLM Calls** - Number of LLM invocations

## Output

Results are saved in multiple formats:

### Pickle Files
```
beir-outputs/
├── dataset_method.pkl    # Complete results with all metrics
└── dataset_method.txt    # Summary statistics
```

### Console Output
Real-time metrics during execution including:
- Per-query results
- Running averages
- Standard deviations

## Example Queries and Instructions

### BEIR Relevance Evaluation
```python
# COVID/SciFact relevance scoring
f"How relevant is {{passage}} to {query}. Only output a single number in the range of 0 to 10 with 10 being highly relevant and 0 means not relevant at all."
```

### Accuracy Ranking Instructions
```python
# CIFAR-10 accuracy ranking
"What {abstract} reports the highest accuracy or lowest error rate on CIFAR-10? Note that error rate is 1 - accuracy. If neither the accuracy nor error rate on CIFAR-10 is not explicitly stated as a number, consider its accuracy to be 0."

# HellaSwag accuracy ranking  
"What {abstract} reports the highest accuracy on HellaSwag? If the accuracy on HellaSwag is not explicitly stated, consider its accuracy to be 0."
```

## Semantic Operations Tested

1. **sem_search** - Initial candidate retrieval
2. **sem_map** - LLM-based relevance scoring
3. **sem_topk** - Semantic top-k selection with various methods:
   - `quick`: Fast approximation
   - `quick-sem`: Semantic approximation
   - `heap`: Heap-based exact sorting
   - `naive`: Baseline method

## Paper Results

The results from these benchmarks were used in the Lotus paper to demonstrate:
- Effectiveness of semantic reranking vs. traditional methods
- Performance trade-offs between accuracy and efficiency
- Scalability of different top-k selection approaches
- Token efficiency of approximation methods

## Notes

- **Indexing**: Semantic indexes are created automatically on first run and reused
- **Batching**: Scripts support batch processing for efficiency
- **Error Handling**: Robust handling of API failures and data issues
- **Reproducibility**: Fixed random seeds and deterministic evaluation
- **Extensibility**: Easy to add new datasets and evaluation methods

## Troubleshooting

1. **API Configuration**: Update the `api_key` and `api_base` in the model configuration
2. **Memory Issues**: Reduce `initial_k` parameter for large datasets
3. **Timeout Issues**: Adjust batch sizes and API timeouts as needed
4. **Data Access**: Ensure BEIR datasets can be downloaded (check network/firewall)
