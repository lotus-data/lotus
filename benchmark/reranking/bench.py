import argparse
import time
import numpy as np
import pickle
from beir.datasets.data_loader import GenericDataLoader
import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
import os

rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
lm = LM(
    model="hosted_vllm/meta-llama/Meta-Llama-3-70B-Instruct",
    api_key="sk-xx",
    api_base="http://localhost:8200/v1/",
    temperature=0.0,
)
lotus.settings.configure(lm=lm, rm=rm)

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

def get_gt(qid, args, sort_k):
    if args.dataset == "covid":
            gt_pids = list(qrels[qid].keys())
            gt_passages = sorted(list(qrels[qid].values()), reverse=True)[:sort_k]
    elif args.dataset == "scifact":
        gt_pids = list(qrels[qid].keys())
        gt_passages = [1] * len(gt_pids) + [0] * (sort_k - len(gt_pids))
    else:
        raise ValueError("Invalid dataset name")
    return gt_passages

def bench(args):
    all_tokens = []
    all_llm_calls = []
    all_latencies = []
    all_ndcgs = []
    all_outputs = []

    sort_k = 10

    for qid, query in queries.items():
        print(f"Evaluating query #{len(all_ndcgs)}")
        df = all_df.sem_search("passage", f"query: {query}", args.initial_k)
        start = time.time()

        if args.method == 'llm-eval':
            df = df.sem_map(
                user_instruction=f"How relevant is {{passage}} to {query}. Only output a single number in the range of 0 to 10 with 10 being highly relevant and 0 means not relevant at all.",
                suffix="relevance",
            )
            df: pd.DataFrame = df.sort_values(by="relevance", ascending=False)[:sort_k]
            
            gt_passages = get_gt(qid, args, sort_k)

            found_passages = []
            for _, row in df.iterrows():
                found_passages.append(qrels[qid].get(row["pid"], 0))
        else:
            df, stats = df.sem_topk(
                f"What {{passage}} is most relevant to the query: {query}",
                K=sort_k,
                method=args.method,
                return_stats=True,
            )
            all_tokens.append(stats["total_tokens"])
            all_llm_calls.append(stats["total_llm_calls"])        
            df = df[:sort_k]

            gt_passages = get_gt(qid, args, sort_k)

            found_passages = []
            for _, row in df.iterrows():
                found_passages.append(qrels[qid].get(row["pid"], 0))

        all_ndcgs.append(ndcg(found_passages, gt_passages, k=sort_k))
        all_latencies.append(time.time() - start)
        all_outputs.append(df)

        print(f"Method: {args.method}, Dataset: {args.dataset}, Query ID: {qid}")
        print(f"nDCG@10: {all_ndcgs[-1]}")
        print("Average nDCG@10: ", np.mean(all_ndcgs))
        print("Average latency: ", np.mean(all_latencies))
        print("Average tokens used: ", np.mean(all_tokens))
        print("Average llm calls: ", np.mean(all_llm_calls))
        print("-" * 80)

    with open(f"beir-outputs/{args.dataset}_{args.method}.pkl", "wb") as fp:
        out = {
            "all_ndcgs": all_ndcgs,
            "all_latencies": all_latencies,
            "all_outputs": all_outputs,
            "all_tokens": all_tokens,
            "all_llm_calls": all_llm_calls,
            "args": args,
        }
        pickle.dump(out, fp)

    with open(f"beir-outputs/{args.dataset}_{args.method}.txt", "w") as fp:
        fp.write(f"Average nDCG@10: {np.mean(all_ndcgs)}\n")
        fp.write(f"Average Latency: {np.mean(all_latencies)}\n")
        fp.write(f"Average Tokens Used: {np.mean(all_tokens)}\n")
        fp.write(f"Average LLM Calls: {np.mean(all_llm_calls)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--method",
    #     type=str,
    #     choices=["quick", "heap", "naive", "quick-sem"],
    #     required=True,
    # )
    parser.add_argument("--initial_k", type=int, default=100)
    parser.add_argument(
        "--dataset", type=str, choices=["covid", "scifact"], default="scifact"
    )
    args = parser.parse_args()
    
    if args.dataset == "covid":
        data_folder = "data/trec-covid"
        index_dir = "indexes/covid_index"
    elif args.dataset == "scifact":
        data_folder = "data/scifact"
        index_dir = "indexes/scifact_index"

    corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(
        split="test"
    )
    all_df = pd.DataFrame(
        {
            "pid": list(corpus.keys()),
            "passage": ["passage: " + e["text"] for e in corpus.values()],
        }
    )
    
    if not os.path.exists(index_dir):
        start_time = time.time()
        all_df.sem_index("passage", index_dir)
        print(f"Indexing time: {time.time() - start_time}")
    else:
        all_df = all_df.load_sem_index("passage", index_dir)
    
    for method in ["llm-eval"]:
        args.method = method
        bench(args)
