
import os
import time
import ast

import numpy as np
import pandas as pd
from datasets import load_dataset
from metrics import  compute_rank_precision
from pipeline_tester import Pipeline, PipelineTester

import lotus
from lotus.models import LM, SentenceTransformersRM, LiteLLMRM
from lotus.vector_store import FaissVS
from lotus.types import CascadeArgs


class BiodexTester(PipelineTester):
    def __init__(self, n_train_samples=11543, n_samples=4249, truncation_limit=8000):
        self.truncation_limit = truncation_limit
        print("Using truncation limit of: ", self.truncation_limit)
        return super().__init__(n_samples)

    def set_results_dir(self):
        return "biodex_results"

    def set_configs(self):
        # rm = SentenceTransformersRM(model="intfloat/e5-base-v2", max_batch_size=4)
        rm = LiteLLMRM(model="text-embedding-3-small", max_batch_size=1000, truncate_limit=8000)

        # lm = LM(
        #     model="hosted_vllm/meta-llama/Meta-Llama-3-70B-Instruct",
        #     api_base="http://localhost:8200/v1/",
        #     max_batch_size=64,
        #     temperature=0.0,
        #     max_tokens=256,
        # )
        lm = LM(model="gpt-4o-mini-2024-07-18",
                max_batch_size=64,
                temperature=0.0,
                max_tokens=256,)
        
        vs = FaissVS()

        lotus.settings.configure(
            lm=lm,
            rm=rm,
            vs=vs
        )
        
        print(f"lotus.settings.lm.max_batch_size = {lotus.settings.lm.max_batch_size}")
        print(f"lotus.settings.lm.max_tokens = {lotus.settings.lm.max_tokens}")
        print(f"lotus.settings.lm.temperature = {lotus.settings.lm.kwargs['temperature']}")



    def load_queries(self, n_samples):
        df = load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()

        # split and remove trailing or leading whitespace
        df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
        df["reactions_list"] = df["reactions_list"].apply(lambda x: [r.strip() for r in x])
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))

        # truncate the fulltext to 8000 chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[: self.truncation_limit])

        return df[:n_samples]


    def load_corpus(self):
        reactions_df = pd.read_csv("biodex-reactions.csv")
        return reactions_df

    def compute_metrics(self, res_df, gt_col_name="reactions_list", pred_col_name="pred_reaction") -> pd.DataFrame:
        res_df["rank_precision@5"] = res_df.apply(
            lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=5),
            axis=1,
        )
        res_df["rank-precision@10"] = res_df.apply(
            lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=10),
            axis=1,
        )
        
        res_df["rank-precision@25"] = res_df.apply(
            lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=25),
            axis=1,
        )


        res_df["num_ids"] = res_df.apply(lambda x: len(x[pred_col_name]), axis=1)

        # take subset of df with metrics
        df = res_df[[col for col in res_df.columns if "@" in col or "latency" in col or "num_ids" in col]]

        return df



class JoinAndRerank(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, top_k=25, name=""):
        if name != "":
            answers_df = pd.read_csv(f"biodex_cascade_answers_for_lm_rerank_{name}.csv", index_col=0)
        else:
            answers_df = pd.read_csv("biodex_cascade_answers_for_lm_rerank.csv", index_col=0)

        start_t = time.time()

        # Rerank the answer with LLM
        def to_comma_separated(val):
            """
            Safely convert val (which could be a list or a string representing a list)
            into a comma-separated string.
            """
            
            if isinstance(val, list):
                return ", ".join(val)
            elif isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return ", ".join(parsed)
                    else:
                        return val
                except (SyntaxError, ValueError):
                    return val
            else:
                return str(val)

        # 2) Normalize reactions_list so every row is a comma-separated string
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(to_comma_separated)

        # 3) Group by and aggregate
        grouped_df = (
            answers_df
            .groupby(["title", "abstract", "reactions", "reactions_list", "patient_description"], dropna=False)
            .apply(lambda grp: grp["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )

        # 4) Convert that comma-separated string (in grouped_df) back to a list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda s: s.split(", ")
        )
        
        rerank_prompt = (
            f"Given the following {{patient_description}} of a medical article, rank the predicted drug reactions {{pred_reaction}} in order of most confident to least confident that the medical article is describing the drug reaction\n"
            "\n\n"
            f"There may be conditions described in the medical article that are not in the list of predicted drug reactions, pred_reaction. Do not include them in the ranked list. Only focus on the conditions in the list."
            "Always write your answer as a list of comma-separated drug reactions only and nothing else."
        )

        grouped_df = grouped_df.sem_map(
            rerank_prompt
        )
        

        end_t = time.time()

        grouped_df.to_csv(f"biodex_reranked_answers_{name}.csv", index=True)

        # Parse output
        known_prefixes = [
    f"Based on the patient description, the most applicable adverse drug reactions are:\n\n",
    f"Based on the Patient_description, the most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
    f"Based on the Patient_description, the most applicable adverse drug reactions are:\n\n",
    f"Based on the provided Patient_description, the most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
    f"Here is the list of most applicable adverse drug reactions:\n\n",
    "Here is the answer:\n\n",
    f"Here is the list of the most applicable adverse drug reactions:\n\n",
    f"Here is the list of most applicable adverse drug reactions:\n\n",
    f"Here is the list of most applicable adverse drug reactions from the options, ranked from most applicable to least applicable:"
]
        def remove_known_prefixes(text: str, prefixes: list) -> str:
            """
            Removes the first matching prefix from 'text' if found in 'prefixes',
            otherwise returns text unchanged.
            """
            for prefix in prefixes:
                if text.startswith(prefix):
                    return text[len(prefix):]
            return text

        grouped_df["_map"] = grouped_df["_map"].fillna("").apply(
            lambda x: remove_known_prefixes(x, known_prefixes)
        )
        grouped_df.rename(columns={"pred_reaction": "pred_reaction_norank"}, inplace=True)
        grouped_df["pred_reaction"] = grouped_df["_map"].apply(
            lambda x: [reaction.strip() for reaction in x.split(",") if reaction.strip()]
        )

        return grouped_df, (end_t - start_t)



class Join(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, name="test"):
        # lotus.settings.
        start_t = time.time()

        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = """Can the following condition be found in the following medical article?
        Medical article: {patient_description} 
        
        Condition we are looking for: {reaction}


        Determine if {reaction} is described in the medical article, considering the context and meaning beyond just the presence of individual words."""
        print(f"corpus_df = {corpus_df.shape}")

        cascade_args = CascadeArgs(
            recall_target=recall_target,
            precision_target=precision_target,
            failure_probability=0.2,
            sampling_percentage=0.00008218, #len = 500 samples
            map_instruction=map_instruction,
            cascade_IS_random_seed=42, # fixed random seed for sampling
        )
        
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)

        end_t = time.time()

        answers_df.to_csv(f"biodex_cascade_answers_for_lm_rerank_{name}.csv", index=True)
        
        print(f"Time taken: {end_t - start_t}")
        print(f"stats = {stats}")

        # post process for checking answers
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(
            lambda x: ", ".join(x)
        )
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from string back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda x: x.split(", ")
        )

        return grouped_df, (end_t - start_t)  # qid    
    
ALL_SAMPLES = 4249
out = []
if __name__ == "__main__":
    ts = BiodexTester(n_samples=250, truncation_limit=128000)

    name = f"GPT4o-mini"
    ts.add_pipeline(Join(recall_target=0.95, precision_target=0.95, name=name))
    ts.add_pipeline(JoinAndRerank(name=name))


    # RUN ALL PIPELINES
    ts.test_pipelines()
    
    lotus.settings.lm.print_total_usage()


    # PRINT SUMMARY OF PIPELINE RESULTS
    print(ts.summarize_pipeline_results("biodex_results", ts.n_samples, ["Join", "JoinAndRerank"]))



    
