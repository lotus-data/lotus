import argparse
import time
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import sys

import lotus  
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS


def setup_lotus():
    """Configure Lotus with appropriate models."""
    try:      
        # Configure models - update these with your available models
        rm = SentenceTransformersRM(model="intfloat/e5-base-v2", max_batch_size=4)
        
        # Use GPT-4o-mini as default (update with your available model)
        lm = LM(
            model="gpt-4o-mini-2024-07-18",
            max_batch_size=8,
            temperature=0.0,
            max_tokens=100
        )
        
        # Configure vector store
        vs = FaissVS()
        
        # Configure Lotus with all required components
        lotus.settings.configure(lm=lm, rm=rm, vs=vs)
        print("Lotus configured with GPT-4o-mini and E5-base-v2")
        
        # Verify configuration
        if lotus.settings.rm is None:
            raise ValueError("Retrieval model not properly configured")
        if lotus.settings.lm is None:
            raise ValueError("Language model not properly configured")
        if lotus.settings.vs is None:
            raise ValueError("Vector store not properly configured")
            
        return True
        
    except ImportError as e:
        print(f"Failed to import Lotus: {e}")
        return False
    except Exception as e:
        print(f"Failed to configure Lotus: {e}")
        return False

def dcg(passages):
    """Compute the Discounted Cumulative Gain (DCG) for a list of passages."""
    return np.sum([(2**passage - 1) / np.log2(idx + 2) for idx, passage in enumerate(passages)])

def ndcg(found_passages, gt_passages, k=None):
    """Compute the normalized Discounted Cumulative Gain (nDCG)."""
    if k:
        found_passages = found_passages[:k]
        gt_passages = gt_passages[:k]

    ideal_passages = sorted(gt_passages, reverse=True)

    dcg_score = dcg(found_passages)
    idcg_score = dcg(ideal_passages)

    return dcg_score / idcg_score if idcg_score > 0 else 0

def load_beir_dataset(dataset_name):
    """Load BEIR dataset for testing."""
    try:
        from beir.datasets.data_loader import GenericDataLoader
        
        if dataset_name == "covid":
            data_folder = "data/trec-covid"
            index_dir = "indexes/covid_index"
        elif dataset_name == "scifact":
            data_folder = "data/scifact"
            index_dir = "indexes/scifact_index"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"Loading {dataset_name} dataset...")
        corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split="test")
        
        # Create DataFrame
        all_df = pd.DataFrame({
            "pid": list(corpus.keys()),
            "passage": ["passage: " + e["text"] for e in corpus.values()],
        })
        
        print(f"Loaded {len(all_df)} passages and {len(queries)} queries")
        
        return all_df, queries, qrels, index_dir
        
    except ImportError:
        print("BEIR not installed. Install with: pip install beir")
        return None, None, None, None
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None, None, None, None

def test_semantic_topk(dataset_name="scifact", num_queries=10, initial_k=100):
    """Test semantic top-k operations on BEIR dataset."""
    
    print(f"Testing semantic top-k on {dataset_name} dataset")
    print(f"Using {num_queries} queries with initial_k={initial_k}")
    
    # Verify Lotus is configured
    print(f"Debug: RM = {lotus.settings.rm}")
    print(f"Debug: LM = {lotus.settings.lm}")
    print(f"Debug: VS = {lotus.settings.vs}")
    
    if lotus.settings.rm is None or lotus.settings.lm is None or lotus.settings.vs is None:
        print("Error: Lotus not properly configured. Call setup_lotus() first.")
        return False, {}
    
    # Load dataset
    all_df, queries, qrels, index_dir = load_beir_dataset(dataset_name)
    if all_df is None:
        return False, {}
    
    # Create or load semantic index
    if not os.path.exists(index_dir):
        print(" Creating semantic index (this may take a few minutes)...")
        start_time = time.time()
        all_df = all_df.sem_index("passage", index_dir)
        index_time = time.time() - start_time
        print(f"Indexing completed in {index_time:.2f} seconds")
    else:
        print("Loading existing semantic index...")
        all_df = all_df.load_sem_index("passage", index_dir)
        print("Index loaded")
    
    # Test on subset of queries for validation
    test_queries = dict(list(queries.items())[:num_queries])
    print(f"Testing on {len(test_queries)} queries")
    
    all_ndcgs = []
    all_latencies = []
    all_tokens = []
    all_llm_calls = []
    sort_k = 10
    
    for i, (qid, query) in enumerate(test_queries.items()):
        print(f"\\n Query {i+1}/{len(test_queries)}: {qid}")
        print(f"   Text: {query[:80]}...")
        
        try:
            # Semantic search to get initial candidates
            df = all_df.sem_search("passage", f"query: {query}", initial_k)
            
            start_time = time.time()
            
            # Test semantic top-k (key operation from paper)

            df, stats = df.sem_topk(
                f"What {{passage}} is most relevant to the query: {query}",
                K=sort_k,
                method="quick",  # Paper's efficient method
                return_stats=True,
            )
            
            latency = time.time() - start_time
            all_latencies.append(latency)
            all_tokens.append(stats.get("total_tokens", 0))
            all_llm_calls.append(stats.get("total_llm_calls", 0))
            
            # Compute nDCG for paper validation
            if dataset_name == "covid":
                gt_passages = sorted(list(qrels[qid].values()), reverse=True)[:sort_k]
            elif dataset_name == "scifact":
                gt_pids = list(qrels[qid].keys())
                gt_passages = [1] * len(gt_pids) + [0] * (sort_k - len(gt_pids))
            
            found_passages = []
            for _, row in df.iterrows():
                found_passages.append(qrels[qid].get(row["pid"], 0))
            
            ndcg_score = ndcg(found_passages, gt_passages, k=sort_k)
            all_ndcgs.append(ndcg_score)
            
            print(f"nDCG@10: {ndcg_score:.4f}")
            print(f"Latency: {latency:.2f}s")
            print(f"Tokens: {stats.get('total_tokens', 'N/A')}")
            print(f"LLM calls: {stats.get('total_llm_calls', 'N/A')}")
            
        except Exception as e:
            print(f"Query failed: {e}")
            continue
    
    # Calculate summary metrics
    if all_ndcgs:
        avg_ndcg = np.mean(all_ndcgs)
        avg_latency = np.mean(all_latencies)
        avg_tokens = np.mean(all_tokens) if all_tokens else 0
        avg_llm_calls = np.mean(all_llm_calls) if all_llm_calls else 0
        
        results = {
            "dataset": dataset_name,
            "num_queries": len(all_ndcgs),
            "avg_ndcg": avg_ndcg,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
            "avg_llm_calls": avg_llm_calls,
            "all_ndcgs": all_ndcgs,
            "timestamp": time.time()
        }
        
        return True, results
    else:
        print("No queries processed successfully")
        return False, {}

def validate_paper_claims(results):
    """Validate results against paper claims."""
    
    print("\\n PAPER CLAIMS VALIDATION:")
    print("="*50)
    
    dataset = results["dataset"]
    avg_ndcg = results["avg_ndcg"]
    avg_latency = results["avg_latency"]
    
    # Paper expectations
    if dataset == "scifact":
        expected_ndcg = 0.6
        print(f"SciFact nDCG@10: {avg_ndcg:.4f} (paper expects > {expected_ndcg})")
        ndcg_meets_expectation = avg_ndcg > expected_ndcg
    elif dataset == "covid":
        expected_ndcg = 0.4
        print(f"COVID nDCG@10: {avg_ndcg:.4f} (paper expects > {expected_ndcg})")
        ndcg_meets_expectation = avg_ndcg > expected_ndcg
    else:
        expected_ndcg = 0.3
        print(f"{dataset} nDCG@10: {avg_ndcg:.4f} (baseline expectation > {expected_ndcg})")
        ndcg_meets_expectation = avg_ndcg > expected_ndcg
    
    print(f"Average latency: {avg_latency:.2f}s per query")
    print(f"Average tokens: {results.get('avg_tokens', 'N/A')}")
    print(f"Average LLM calls: {results.get('avg_llm_calls', 'N/A')}")
    
    # Validation results
    print("\\n VALIDATION RESULTS:")
    if ndcg_meets_expectation:
        print(f"nDCG meets paper expectations ({avg_ndcg:.4f} > {expected_ndcg})")
    else:
        print(f"nDCG below paper expectations ({avg_ndcg:.4f} <= {expected_ndcg})")
        print("   This could be due to model differences or configuration")
    
    if avg_latency < 10:  # Reasonable latency expectation
        print(f"Latency is reasonable ({avg_latency:.2f}s per query)")
    else:
        print(f"Latency is high ({avg_latency:.2f}s per query)")
    
    return ndcg_meets_expectation

def main():
    parser = argparse.ArgumentParser(description="Validate reranking results against Lotus paper")
    parser.add_argument("--dataset", choices=["scifact", "covid"], default="scifact",
                       help="BEIR dataset to test (default: scifact)")
    parser.add_argument("--queries", type=int, default=10,
                       help="Number of queries to test (default: 10)")
    parser.add_argument("--initial-k", type=int, default=100,
                       help="Initial candidates to retrieve (default: 100)")
    
    args = parser.parse_args()
    
    print("Reranking Paper Results Validation")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {args.queries}")
    print(f"Initial K: {args.initial_k}")
    
    # Setup Lotus
    if not setup_lotus():
        print(" Failed to setup Lotus")
        return False
    
    # Run semantic top-k test
    print("\\n Running semantic top-k evaluation...")
    success, results = test_semantic_topk(args.dataset, args.queries, args.initial_k)
    
    if not success:
        print("Semantic top-k test failed")
        return False
    
    # Validate against paper
    paper_validation = validate_paper_claims(results)
    
    # Save results
    results_file = Path(f"paper_validation_results_{args.dataset}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n Results saved to {results_file}")
    
    if paper_validation:
        print("\\n Results align with paper expectations!")
    else:
        print("\\n Results differ from paper - check model configuration")
    
    return success and paper_validation

if __name__ == "__main__":
    print("Starting Reranking Paper Validation Test")
    print("This test validates key results from the Lotus paper")
    print("Expected runtime: 3-10 minutes depending on dataset size and API speed")
    
    success = main()
    
    if success:
        print("Reranking validation completed successfully!")
    else:
        print("Reranking validation had issues")
    
    sys.exit(0 if success else 1)

