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
    temperature=0.7,
    # max_workers=100,
    # max_batch_size=64
)
lotus.settings.configure(lm=lm, rm=rm)

def dcg(passages):
    """Compute the Discounted Cumulative Gain (DCG) for a list of passages."""
    return np.sum([max(0, passage)  / np.log2(idx + 2) for idx, passage in enumerate(passages)])

def ndcg(found_passages, gt_passages, k=None):
    """Compute the normalized Discounted Cumulative Gain (nDCG)."""
    if k:
        found_passages = found_passages[:k]
        gt_passages = gt_passages[:k]

    ideal_passages = sorted(gt_passages, reverse=True)

    dcg_score = dcg(found_passages)
    idcg_score = dcg(ideal_passages)

    return dcg_score / idcg_score if idcg_score > 0 else 0


def recall(gt_pids, found_pids):
    """Compute the recall of a list of found passages."""
    return len(set(found_pids).intersection(set(gt_pids))) / len(gt_pids)

def bench(args):
    all_tokens = []
    all_llm_calls = []
    all_latencies = []
    all_recalls = []
    all_ndcgs = []
    all_outputs = []

    for t in range(args.trials):
        df = pd.read_csv(df_path)
        if os.path.exists(index_dir):
            df = df.load_sem_index("abstract", index_dir)
        else:
            df = df.sem_index("abstract", index_dir)
        gt_df: pd.DataFrame = df.sort_values(by="acc", ascending=False)

        start = time.time()
        if args.method == 'llm-eval':
            df = df.sem_map(
                user_instruction="Output the accuracy of {abstract} on CIFAR-10 on a scale of 0-10. Note that error rate is 1 - accuracy. If niether the accuracy not error rate on CIFAR-10 is not explicitly stated as a number, consider its accuracy to be 0. Do not get influenced by statements saying that it outperforms other methods and look only at concrete accuracy numbers. Only output a single number in the range of 0 to 10 with 10 being highly relevant and 0 means not relevant at all.",
                suffix="relevance",
            )
            df: pd.DataFrame = df.sort_values(by="relevance", ascending=False)[:sort_k]
        else:
            df, stats = df.sem_topk(
                sort_query,
                K=sort_k,
                method=args.method,
                return_stats=True,
            )
            all_tokens.append(stats["total_tokens"])
            all_llm_calls.append(stats["total_llm_calls"])        
        df: pd.DataFrame = df[:sort_k]
        all_latencies.append(time.time() - start)
        
        print(df)
        print(gt_df)
        
        all_ndcgs.append(ndcg(sort_k - df["pid"].to_numpy(), sort_k - gt_df[:sort_k].index.to_numpy(), k=sort_k))
        all_recalls.append(recall(gt_df[:sort_k]["pid"], df["pid"]))
        all_outputs.append(df)
        

        print(f"Method: {args.method}, Dataset: {args.dataset}, Trial number: {t}")
        print(f"nDCG@10: {all_ndcgs[-1]}")
        print(f"Recall@10: {all_recalls[-1]}")
        print("Average nDCG@10: ", np.mean(all_ndcgs))
        print("Average Recall@10: ", np.mean(all_recalls))
        print("Average latency: ", np.mean(all_latencies))
        print("Average tokens used: ", np.mean(all_tokens))
        print("Average llm calls: ", np.mean(all_llm_calls))
        # print std
        print("Std nDCG@10: ", np.std(all_ndcgs))
        print("Std Recall@10: ", np.std(all_recalls))
        print("Std latency: ", np.std(all_latencies))
        print("Std tokens used: ", np.std(all_tokens))
        print("Std llm calls: ", np.std(all_llm_calls))
        print("-" * 80)

    with open(f"beir-outputs/{args.dataset}_{args.method}.pkl", "wb") as fp:
        out = {
            "all_ndcgs": all_ndcgs,
            "all_recalls": all_recalls,
            "all_latencies": all_latencies,
            "all_outputs": all_outputs,
            "all_tokens": all_tokens,
            "all_llm_calls": all_llm_calls,
            "args": args,
        }
        pickle.dump(out, fp)

    with open(f"beir-outputs/{args.dataset}_{args.method}.txt", "w") as fp:
        fp.write(f"Average nDCG@10: {np.mean(all_ndcgs)}\n")
        fp.write(f"Average Recall@10: {np.mean(all_recalls)}\n")
        fp.write(f"Average Latency: {np.mean(all_latencies)}\n")
        fp.write(f"Average Tokens Used: {np.mean(all_tokens)}\n")
        fp.write(f"Average LLM Calls: {np.mean(all_llm_calls)}\n")
        # add std
        fp.write(f"Std nDCG@10: {np.std(all_ndcgs)}\n")
        fp.write(f"Std Recall@10: {np.std(all_recalls)}\n")
        fp.write(f"Std Latency: {np.std(all_latencies)}\n")
        fp.write(f"Std Tokens Used: {np.std(all_tokens)}\n")
        fp.write(f"Std LLM Calls: {np.std(all_llm_calls)}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["cifar", "hellaswag"], default="cifar"
    )
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()


    if args.dataset == "cifar":
        df_path = "data/cifar.csv"
        index_dir = "indexes/cifar_index"
        sort_query = "What {abstract} reports the highest accuracy or lowest error rate on CIFAR-10?  Note that error rate is 1 - accuracy. If niether the accuracy not error rate on CIFAR-10 is not explicitly stated as a number, consider its accuracy to be 0. Do not get influenced by statements saying that it outperforms other methods and look only at concrete accuracy numbers."
        sort_k = 10
    elif args.dataset == "hellaswag":
        df_path = "data/hellaswag.csv"
        index_dir = "indexes/hellaswag_index"
        sort_query = "What {abstract} reports the highest accuracy on HellaSwag? If the accuracy on HellaSwag is not explicitly stated, consider its accuracy to be 0."
        sort_k = 10

    for method in ["llm-eval", "quick", "quick-sem", "heap", "naive"]:
        args.method = method
        bench(args)
