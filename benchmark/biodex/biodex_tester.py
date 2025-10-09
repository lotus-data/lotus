
import os
import time
import ast

import numpy as np
import pandas as pd
from biodex_prompts import map_dem_df
from datasets import load_dataset
from metrics import compute_precision, compute_rank_precision, compute_recall
from pipeline_tester import Pipeline, PipelineTester
from scipy.linalg import lstsq

import lotus
from lotus.models import LM, SentenceTransformersRM, LiteLLMRM
from lotus.types import CascadeArgs


def learn_linear_transformation(S, A):
    # copy S and A
    S = S.copy()
    A = A.copy()
    # Flatten matrices S and A
    S_flat = S.reshape(-1, S.shape[-1])  # Reshape S into a 2D array
    A_flat = A.reshape(-1, A.shape[-1])  # Reshape A into a 2D array

    # Use least squares regression to find the linear transformation
    transformation, _, _, _ = lstsq(S_flat, A_flat)

    return transformation.reshape(S.shape[-1], A.shape[-1])  # Reshape the result into a matrix


class BiodexTester(PipelineTester):
    def __init__(self, n_train_samples=11543, n_samples=4249, truncation_limit=8000):
        self.truncation_limit = truncation_limit
        print("Using truncation limit of: ", self.truncation_limit)
        self.train_queries = self.load_train_queries(n_train_samples)
        return super().__init__(n_samples)
        # intializes self.queries_df, self.corpus_df, self.results_dir, self.pipelines

    def set_results_dir(self):
        return "biodex_results"

    def set_configs(self):
        rm = SentenceTransformersRM(model="intfloat/e5-base-v2", max_batch_size=4)
        # rm = LiteLLMRM(model="text-embedding-3-small")

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

        lotus.settings.configure(
            lm=lm,
            rm=rm,
        )
        
        print(f"lotus.settings.lm.max_batch_size = {lotus.settings.lm.max_batch_size}")
        print(f"lotus.settings.lm.max_tokens = {lotus.settings.lm.max_tokens}")
        print(f"lotus.settings.lm.temperature = {lotus.settings.lm.kwargs['temperature']}")

    def load_train_queries(self, n_samples):
        df = load_dataset("BioDEX/BioDEX-Reactions", split="train").to_pandas()

        # split and remove trailing or leading whitespace
        df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
        df["reactions_list"] = df["reactions_list"].apply(lambda x: [r.strip() for r in x])
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
        df = df.load_sem_index("abstract", "biodex_abstract_e5")

        # truncate the fulltext to 8000 chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[: self.truncation_limit])

        # print the max number of characters in the truncated fulltext
        print(df["patient_description"].apply(len).max())

        return df[:n_samples]

    def load_queries(self, n_samples):
        df = load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()

        # split and remove trailing or leading whitespace
        df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
        df["reactions_list"] = df["reactions_list"].apply(lambda x: [r.strip() for r in x])
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
        df = df.load_sem_index("abstract", "biodex_abstract_e5")

        # truncate the fulltext to 8000 chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[: self.truncation_limit])

        # print the max number of characters in the truncated fulltext
        print(df["patient_description"].apply(len).max())

        return df[:n_samples]

    def load_map_df(self, filename="map_df.csv"):
        df = pd.read_csv(filename)

        # split and remove trailing or leading whitespace
        df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
        df["reactions_list"] = df["reactions_list"].apply(lambda x: [r.strip() for r in x])
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
        df = df.load_sem_index("abstract", "biodex_abstract_e5")

        # truncate the fulltext to 8000 chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[: self.truncation_limit])

        # print the max number of characters in the truncated fulltext
        print(df["patient_description"].apply(len).max())

        return df

    def load_corpus(self):
        reactions_df = pd.read_csv("biodex-reactions.csv")
        reactions_df.load_sem_index("reaction", "biodex_reactions_e5")
        return reactions_df

    # TODO this should be refactored to take a qa_df
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

        for k in [5, 10, 20, 50, 100, 200, 300, 400, 500]:
            res_df[f"recall@{k}"] = res_df.apply(lambda x: compute_recall(x[gt_col_name], x[pred_col_name], k), axis=1)

        res_df["precision@5"] = res_df.apply(lambda x: compute_precision(x[gt_col_name], x[pred_col_name], 5), axis=1)
        res_df["precision@10"] = res_df.apply(lambda x: compute_precision(x[gt_col_name], x[pred_col_name], 10), axis=1)

        res_df["num_ids"] = res_df.apply(lambda x: len(x[pred_col_name]), axis=1)

        # take subset of df with metrics
        df = res_df[[col for col in res_df.columns if "@" in col or "latency" in col or "num_ids" in col]]

        return df

    def get_map_df(self):
        map_df = self.queries_df.sem_map(
            "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
            suffix="map_preds",
            examples=map_dem_df,
        )
        return map_df


class Retrieve(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, K=1):
        start_t = time.time()
        answers_df = queries_df.sem_sim_join(corpus_df, left_on="fulltext", right_on="reaction", K=K)
        end_t = time.time()
        print(f"Time taken: {end_t - start_t}")
        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(lambda x: ", ".join(x))
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from sting back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(lambda x: x.split(", "))

        return grouped_df, (end_t - start_t)  # qid


class MapRetrieve(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, K=1):
        start_t = time.time()
        map_df = queries_df.sem_map(
            # "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
            "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated medical reactions.",
            examples=map_dem_df,
            suffix="map_preds",
        )
        answers_df = map_df.sem_sim_join(corpus_df, left_on="map_preds", right_on="reaction", K=K)
        end_t = time.time()
        print(f"Time taken: {end_t - start_t}")
        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(lambda x: ", ".join(x))
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from sting back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(lambda x: x.split(", "))

        return grouped_df, (end_t - start_t)  # qid


class MapRetrieveFilter(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, K=1):
        start_t = time.time()
        map_df = queries_df.sem_map(
            # "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
            "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated medical reactions.",
            examples=map_dem_df,
            suffix="map_preds",
        )
        answers_df = map_df.sem_sim_join(corpus_df, left_on="map_preds", right_on="reaction", K=K)
        answers_df = answers_df.sem_filter(
            "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
        )

        end_t = time.time()
        print(f"Time taken: {end_t - start_t}")

        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(lambda x: ", ".join(x))
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from sting back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(lambda x: x.split(", "))

        return grouped_df, (end_t - start_t)  # qid

class ApproxMapRetrieveFilter(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, t_pos, t_neg, num_quantile=100000, proxy_df=None):
        start_t = time.time()
        map_df = queries_df.sem_map(
            # "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
            "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated medical reactions.",
            examples=map_dem_df,
            suffix="map_preds",
        )
        
        K=len(corpus_df)
        answers_df = map_df.sem_sim_join(corpus_df, left_on="map_preds", right_on="reaction", K=K)
        
        answers_df = self.quantile_scores(answers_df, '_scores', num_quantile=num_quantile)
        
        # Filter based on t_pos and t_neg
        need_oracle = answers_df[
            (answers_df['_scores'] < t_pos) & (answers_df['_scores'] >= t_neg)
        ].copy()
        # Keep only high-scoring rows in answers_df
        answers_df = answers_df[answers_df['_scores'] >= t_pos]

        num_need_oracle = need_oracle.shape[0]
        print(f"number of reference llm call: {num_need_oracle}")
        
        if not need_oracle.empty:
            need_oracle = need_oracle.sem_filter(
                "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
            )
            
        # Rerank for better RP@k
        answers_df = answers_df.sort_values(by='_scores', ascending=False)
        answers_df = pd.concat([answers_df, need_oracle], ignore_index=True)
        end_t = time.time()
        print(f"Time taken: {end_t - start_t}")

        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(lambda x: ", ".join(x))
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from sting back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(lambda x: x.split(", "))
        
        lat = (end_t - start_t)
        out.append(lat)
        

        return grouped_df, lat  # qid

    def quantile_scores(self, df, col, num_quantile):
        # Get the scores from the specified column
        scores = df[col].values
        
        # Compute the quantile boundaries
        quantile_values = np.percentile(scores, np.linspace(0, 100, num_quantile + 1))
        
        # Digitize the scores into quantiles and normalize
        df[col] = (np.digitize(scores, quantile_values) - 1) / num_quantile
        
        return df

class RetrieveFilter(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, K=1, t_pos=None, t_neg=None):
        start_t = time.time()
        # print(len(queries_df))
        answers_df = queries_df.vec_join(
            corpus_df, left_on="abstract", right_on="reaction", K=K, return_scores=True
        )
        answers_df[['_scores']].to_csv(f"retrieve_filter_proxy_score_{answers_df.shape[0]}.csv", index=False) # save intermediate result for debugging
        # print(len(answers_df))
        # Filter the answers based on the threshold
        if t_pos is not None and t_neg is not None:
            # Extract mid-range scores into need_oracle
            need_oracle = answers_df[
                (answers_df['_scores'] < t_pos) & (answers_df['_scores'] >= t_neg)
            ].copy()
            # Keep only high-scoring rows in answers_df
            answers_df = answers_df[answers_df['_scores'] >= t_pos]
        else:
            need_oracle = answers_df
        num_need_oracle = need_oracle.shape[0]
        print(f"number of llm call: {num_need_oracle}")

        # Run need_oracle through llm_filter
        if not need_oracle.empty:
            need_oracle = need_oracle.llm_filter(
                "Given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?"
            )

        answers_df = answers_df.sort_values(by='_scores', ascending=False)
        answers_df = pd.concat([need_oracle, answers_df], ignore_index=True)
        
        end_t = time.time()
        print(f"Time taken: {end_t - start_t}")
        
        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(
            lambda x: ", ".join(x)
        )
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from sting back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda x: x.split(", ")
        )
        
        print(f"number of llm call: {num_need_oracle}")

        return grouped_df, (end_t - start_t)  # qid


class ApproxRetrieveFilter(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, t_pos=None, t_neg=None, num_quantile=100000):
        start_t = time.time()
        # print(len(queries_df))
        K=len(corpus_df)
        answers_df = queries_df.sem_sim_join(corpus_df, left_on="abstract", right_on="reaction", K=K)
        answers_df = self.quantile_scores(answers_df, '_scores', num_quantile=num_quantile)
        
        # Filter based on t_pos and t_neg
        need_oracle = answers_df[
            (answers_df['_scores'] < t_pos) & (answers_df['_scores'] >= t_neg)
        ].copy()
        # Keep only high-scoring rows in answers_df
        answers_df = answers_df[answers_df['_scores'] >= t_pos]

        num_need_oracle = need_oracle.shape[0]
        print(f"number of llm call: {num_need_oracle}")
        
        if not need_oracle.empty:
            need_oracle = need_oracle.sem_filter(
                "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
            )

        # Rerank for better RP@k
        answers_df = answers_df.sort_values(by='_scores', ascending=False)
        answers_df = pd.concat([answers_df, need_oracle], ignore_index=True)
        end_t = time.time()
        print(f"Time taken: {end_t - start_t}")
        
        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(
            lambda x: ", ".join(x)
        )
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from sting back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda x: x.split(", ")
        )
        
        print(f"number of llm call: {num_need_oracle}")
        lat = (end_t - start_t)
        out.append(lat)
        

        return grouped_df, (end_t - start_t)  # qid

    def quantile_scores(self, df, col, num_quantile):
        # Get the scores from the specified column
        scores = df[col].values
        
        # Compute the quantile boundaries
        quantile_values = np.percentile(scores, np.linspace(0, 100, num_quantile + 1))
        
        # Digitize the scores into quantiles and normalize
        df[col] = (np.digitize(scores, quantile_values) - 1) / num_quantile
        
        return df


class ApproximateJoinTest():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, threshold=0.1, case="search-filter"):
        return self.importance_sampling_run(queries_df, corpus_df, recall_target, precision_target, threshold, case="map-search-filter", prob_gtd=0.8, seed=42, is_weight=0.5, num_quantile=1000)

    def importance_sampling_run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, threshold=0.1, case="map-search-filter", prob_gtd=0.8, seed=42, is_weight=0.5, num_quantile=1000):
        if seed >= 0:
            np.random.seed(seed)
        print("start importance sampling")
        start_t = time.time()

        # Get proxy result
        K = len(corpus_df)
        if case == "map-search-filter":
            filename = f'map_preds_{len(queries_df)}.csv'
            # Check if file exists
            if os.path.exists(filename):
                # Load the file if it exists
                map_df = pd.read_csv(filename)
            else:
                # Run the computation and save the result to the file
                map_df = queries_df.sem_map(
                    "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated medical reactions.",
                    examples=map_dem_df,
                    suffix="map_preds",
                )
                # Save the result to a CSV file
                map_df.to_csv(filename, index=False)
            print("before sem_sim_join")
            proxy_df = map_df.sem_sim_join(corpus_df, left_on="map_preds", right_on="reaction", K=K)
            print("after sem_sim_join")
        elif case == "search-filter":
            K = len(corpus_df)
            proxy_df = queries_df.sem_sim_join(corpus_df, left_on="patient_description", right_on="reaction", K=K)
        else:
            raise ValueError("{case} not supported, please use 'search-filter' or 'map-search-filter'")

        # proxy_df = self.quantile_scores(proxy_df, '_scores', num_quantile=num_quantile)
        # proxy_df[['_scores']].to_csv(f"{case}_full_proxy_score_{proxy_df.shape[0]}.csv", index=True) # save intermediate result for debugging

        # Importance sampling
        num_samples = int(proxy_df.shape[0] * threshold)
        weights = np.sqrt(proxy_df['_scores'].values)
        
        # Defensive Mixing
        weights = is_weight * weights / np.sum(weights) + (1 - is_weight) * np.ones_like(weights) / len(proxy_df)
        weights = weights/np.sum(weights) # differ by some very small floating point amount, need to renormalize

        # Weighted Sampling
        sampled_indices = np.random.choice(proxy_df.index, size=num_samples, p=weights, replace=False)
        sampled_proxy_df = proxy_df.iloc[sampled_indices].copy()
        print(f'sampled_indices = {type(sampled_indices)}, weights={type(weights)}')

        # Calculate reweighting factor m(x)
        sampled_weights = weights[sampled_indices]
        sampled_proxy_df['m'] = (1 / len(proxy_df)) / sampled_weights
        
        # Get oracle result
        unique_sample_df = sampled_proxy_df.drop_duplicates(subset=['patient_description', 'reaction'])
        oracle_df = unique_sample_df.sem_filter(
            "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
        )

        # Map oracle result to proxy result for validation
        sampled_proxy_df['proxy_index'] = sampled_proxy_df.index
        oracle_df = oracle_df.copy()
        oracle_df['oracle_index'] = oracle_df.index

        merged = pd.merge(sampled_proxy_df, oracle_df, on=['patient_description', 'reaction', '_scores', 'm'], how='left', indicator=True)
        merged['oracle'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)

        # Creating score_df with 'proxy_score' and 'oracle'
        score_df = merged[['_scores', 'oracle', 'm']].rename(columns={'_scores': 'proxy_score'})
        score_df.to_csv(f"{case}_recall_{is_weight}_sampled_proxy_df_{num_samples}.csv", index=True) # save intermediate result for debugging

        # Find threshold that meet target recall
        delta = (1 - prob_gtd) / 2    # account for precision
        print(f"delta = {delta}")
        t_neg_candidates = self.weighted_find_tau_neg(score_df, recall_target, delta=delta)
        print(f"t_neg_candidates = {t_neg_candidates}")

        # Find threshold that meet target precision
        # ---------------------------------------------------------------------
        S0_indices = np.random.choice(proxy_df.index, size=int(num_samples/2), p=weights, replace=False)
        S0_proxy_df = proxy_df.iloc[S0_indices].copy()

        # Calculate reweighting factor m(x)
        S0_weights = weights[S0_indices]
        S0_proxy_df['m'] = (1 / len(proxy_df)) / S0_weights
        
        # Get oracle result
        S0_unique = S0_proxy_df.drop_duplicates(subset=['patient_description', 'reaction'])
        S0_oracle = S0_unique.sem_filter(
            "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
        )

        # Map oracle result to proxy result for validation
        S0_proxy_df['proxy_index'] = S0_proxy_df.index
        S0_oracle = S0_oracle.copy()
        S0_oracle['oracle_index'] = S0_oracle.index
        
        merged = pd.merge(S0_proxy_df, S0_oracle, on=['patient_description', 'reaction', '_scores', 'm'], how='left', indicator=True)
        merged['oracle'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)

        # Creating score_df with 'proxy_score' and 'oracle'
        S0_score_df = merged[['_scores', 'oracle', 'm']].rename(columns={'_scores': 'proxy_score'})
        S0_score_df.to_csv(f"search_filter_score_df_{num_samples}.csv", index=False) # save intermediate result for debugging

        Z = [oracle * mx for oracle, mx in S0_score_df[['oracle', 'm']].values]
        mean_z = np.mean(Z) if Z else 0
        std_z = np.std(Z) if Z else 0
        delta = (1 - prob_gtd)
        n_match = len(proxy_df) * self.UB(mean_z, std_z, int(num_samples/2), delta/2)
        print(f"n_match = {n_match}")
        
        # Resample on a pruned proxy_df
        proxy_df['weight'] = weights
        proxy_df = proxy_df.sort_values(by='_scores', ascending=False)
        proxy_df_prime = proxy_df.head(int(n_match / precision_target))

        S1_indices = np.random.choice(proxy_df_prime.index, size=int(num_samples/2), p=proxy_df_prime['weight']/np.sum(proxy_df_prime['weight']), replace=False)
        S1_proxy_df = proxy_df.iloc[S1_indices].copy()
        
        # Get S1 oracle
        S1_unique = S1_proxy_df.drop_duplicates(subset=['patient_description', 'reaction'])
        S1_oracle = S1_unique.sem_filter(
            "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
        )
        
        # Map oracle result to proxy result for validation
        S1_proxy_df['proxy_index'] = S1_proxy_df.index
        S1_oracle = S1_oracle.copy()
        S1_oracle['oracle_index'] = S1_oracle.index
        
        merged = pd.merge(S1_proxy_df, S1_oracle, on=['patient_description', 'reaction', '_scores'], how='left', indicator=True)
        merged['oracle'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)

        S1_score_df = merged[['_scores', 'oracle']].rename(columns={'_scores': 'proxy_score'})
        S1_score_df.to_csv(f"{case}_precision_{is_weight}_sampled_proxy_df_{num_samples}.csv", index=True) # save intermediate result for debugging
        # Find threshold that meet target recall
        t_pos_candidates = self.weighted_find_tau_pos(S1_score_df, precision_target, delta=delta)
        print(f"t_pos_candidates = {t_pos_candidates}")
        
        # Find valid pair
        valid_pairs = [(t_pos, t_neg) for t_pos in t_pos_candidates for t_neg in t_neg_candidates if t_pos > t_neg]
        if not valid_pairs:
            print("No valid (tau_pos, tau_neg) pair found where tau_pos > tau_neg.")
            valid_pairs = [(1.0, 0.0)]
        
        # Maximize t_neg and minimize t_pos
        t_pos, t_neg = self.find_optimal_pair(t_pos_candidates, t_neg_candidates)
        end_t = time.time()
        
        # Statistics
        oracle_calls = sum(1 for x in proxy_df['_scores'] if t_pos > x >= t_neg)
        print(f"# sample to oracle during inference: {oracle_calls}, total llm call: {oracle_calls + num_samples * 2}")

        # Final achieved precision and recall
        pairs = zip(score_df['proxy_score'], score_df['oracle'])
        r = self.recall(t_pos, t_neg, pairs)
        print(f"achieved recall: {r}")
        pairs = zip(S1_score_df['proxy_score'], S1_score_df['oracle'])
        p = self.precision(t_pos, t_neg, pairs)
        print(f"achieved precision: {p}")
        
        print(f"{case}:    t-pairs: ({t_pos}, {t_neg}) , latency: {end_t - start_t}")

        return t_pos, t_neg, oracle_calls, (end_t - start_t), proxy_df  # qid

    
    def weighted_find_tau_neg(self, score_df, recall_target, delta):
        score_df.to_csv(f"score_df_{(len(score_df))}.csv", index=False)
        sorted_pairs = sorted(set(zip(score_df['proxy_score'], score_df['oracle'], score_df['m'])), key=lambda x: x[0], reverse=True)
        # Initialize with an initial threshold for positive and negative
        best_combination = (1, 0)  # t_pos, t_neg

        # Find tau_negative based on recall
        tau_neg_0 = max(x[0] for x in sorted_pairs[::-1] if self.weighted_recall(best_combination[0], x[0], sorted_pairs) >= recall_target)
        best_combination = (best_combination[0], tau_neg_0)

        # Statistical correction for recall
        Z1 = [mx if oracle == 1 else 0 for proxy_score, oracle, mx in sorted_pairs if proxy_score >= best_combination[1]]
        Z2 = [mx if oracle == 1 else 0 for proxy_score, oracle, mx in sorted_pairs if proxy_score < best_combination[1]]

        mean_z1 = np.mean(Z1) if Z1 else 0
        std_z1 = np.std(Z1) if Z1 else 0
        print(f"mean_z1={mean_z1}")
        print(f"std_z1={std_z1}")
        mean_z2 = np.mean(Z2) if Z2 else 0
        std_z2 = np.std(Z2) if Z2 else 0

        # Update to delta over 2
        sample_size = len(sorted_pairs)
        corrected_recall_target = self.UB(mean_z1, std_z1, sample_size, delta/2)/(self.UB(mean_z1, std_z1, sample_size, delta/2) + self.LB(mean_z2, std_z2, sample_size, delta/2))
        print(f"corrected_recall_target = {corrected_recall_target}")
        
        tau_neg_candidates = [x[0] for x in sorted_pairs[::-1] if self.weighted_recall(best_combination[0], x[0], sorted_pairs) >= corrected_recall_target]
        if not tau_neg_candidates:
            tau_neg_candidates.append(0)

        return tau_neg_candidates

    def parse_proxy_score(self, df, method='', num_quantile=1000):
        if method == 'quantile':
            return self.quantile_scores(df, '_scores', num_quantile=num_quantile)
        elif method == "clip":
            return self.clip_scores(df, '_scores')
        return df

    def clip_scores(self, df, col):
        df[col] = np.clip(df[col], 0, 1)
        return df

    def quantile_scores(self, df, col, num_quantile):
        # Get the scores from the specified column
        scores = df[col].values
        
        # Compute the quantile boundaries
        quantile_values = np.percentile(scores, np.linspace(0, 100, num_quantile + 1))
        
        # Digitize the scores into quantiles and normalize
        df[col] = (np.digitize(scores, quantile_values) - 1) / num_quantile
        
        return df

    def weighted_find_tau_pos(self, score_df, precision_target, delta):
        score_df.to_csv(f"score_df_{(len(score_df))}.csv", index=False)
        sorted_pairs = sorted(set(zip(score_df['proxy_score'], score_df['oracle'])), key=lambda x: x[0], reverse=True)

        # Statistical correction for precision
        tau_pos_candidates = []
        for proxy_score, oracle in sorted_pairs:
            possible_threshold = proxy_score
            Z = [1 if oracle == 1 else 0 for proxy_score, oracle in sorted_pairs if proxy_score >= possible_threshold]
            mean_z = np.mean(Z) if Z else 0
            std_z = np.std(Z) if Z else 0
            p_l = self.LB(mean_z, std_z, sum(Z), delta / (2 * len(score_df)) )
            if p_l > precision_target:
                tau_pos_candidates.append(possible_threshold)

        if not tau_pos_candidates:
            tau_pos_candidates.append(1)

        return tau_pos_candidates


    def find_optimal_thresholds(self, score_df, recall_target, precision_target, prob_gtd=0.8):
        sorted_pairs = sorted(set(zip(score_df['proxy_score'], score_df['oracle'])), key=lambda x: x[0], reverse=True)
        # Initialize with an initial threshold for positive and negative
        best_combination = (1, 0)  # t_pos, t_neg

        # Find tau_negative based on recall
        tau_neg_0 = max(x[0] for x in sorted_pairs[::-1] if self.recall(best_combination[0], x[0], sorted_pairs) >= recall_target)
        best_combination = (best_combination[0], tau_neg_0)

        # Statistical correction for recall
        Z1 = [1 if (proxy_score >= best_combination[1] and oracle == 1) else 0 for (proxy_score, oracle) in sorted_pairs] # Z1 is the true positive set using temp-tau
        Z2 = [1 if (proxy_score < best_combination[1] and oracle == 1) else 0 for (proxy_score, oracle) in sorted_pairs] # Z2 is the false negative set using temp-tau

        mean_z1 = np.mean(Z1) if Z1 else 0
        std_z1 = np.std(Z1) if Z1 else 0
        print(f"mean_z1={mean_z1}")
        print(f"std_z1={std_z1}")
        mean_z2 = np.mean(Z2) if Z2 else 0
        std_z2 = np.std(Z2) if Z2 else 0

        # Sta
        delta = (1 - prob_gtd)
        sample_size = len(sorted_pairs)
        corrected_recall_target = self.UB(mean_z1, std_z1, sample_size, delta/2)/(self.UB(mean_z1, std_z1, sample_size, delta/2) + self.LB(mean_z2, std_z2, sample_size, delta/2))
        print(f"corrected_recall_target = {corrected_recall_target}")
        
        tau_neg_candidates = [x[0] for x in sorted_pairs[::-1] if self.recall(best_combination[0], x[0], sorted_pairs) >= corrected_recall_target]
        if not tau_neg_candidates:
            tau_neg_candidates.append(0)

        # Statistical correction for precision
        tau_pos_candidates = []
        for proxy_score, oracle in sorted_pairs:
            possible_threshold = proxy_score
            Z = [1 if oracle == 1 else 0 for proxy_score, oracle in sorted_pairs if proxy_score >= possible_threshold]
            mean_z = np.mean(Z) if Z else 0
            std_z = np.std(Z) if Z else 0
            p_l = self.LB(mean_z, std_z, sum(Z), delta / len(sorted_pairs))
            if p_l > precision_target:
                tau_pos_candidates.append(possible_threshold)

        if not tau_pos_candidates:
            tau_pos_candidates.append(1)

        # Find valid pair
        valid_pairs = [(t_pos, t_neg) for t_pos in tau_pos_candidates for t_neg in tau_neg_candidates if t_pos > t_neg]
        if not valid_pairs:
            print("No valid (tau_pos, tau_neg) pair found where tau_pos > tau_neg.")
            valid_pairs = [(1, 0)]
        
        # Maximize t_neg and minimize t_pos
        best_combination = self.find_optimal_pair(tau_pos_candidates, tau_neg_candidates)

        oracle_calls = sum(1 for x in sorted_pairs if best_combination[0] > x[0] > best_combination[1])
        print(f"# sample to oracle: {oracle_calls}")

        # Final achieved precision and recall
        r = self.recall(best_combination[0], best_combination[1], sorted_pairs)
        print(f"achieved recall: {r}")
        p = self.precision(best_combination[0], best_combination[1], sorted_pairs)
        print(f"achieved precision: {p}")
        return best_combination[0], best_combination[1]

    def recall(self, tau_pos, tau_neg, sorted_pairs):
        # Function to compute recall
        tp = sum(1 for (proxy_score, oracle) in sorted_pairs if proxy_score >= tau_neg and oracle == 1)
        fn = sum(1 for (proxy_score, oracle) in sorted_pairs if proxy_score < tau_neg and oracle == 1)
        # print(f"tp: {tp}, fn: {fn}")
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def precision(self, tau_pos, tau_neg, sorted_pairs):
        # Function to compute recall
        Z = [1 if oracle == 1 else 0 for proxy_score, oracle in sorted_pairs if proxy_score >= tau_pos]
    
        return sum(Z) / len(Z) if len(Z) > 0 else 0
    
    def weighted_recall(self, tau_pos, tau_neg, sorted_pairs):
        # Function to compute recall
        Z1_sum =  sum(mx if oracle == 1 else 0 for proxy_score, oracle, mx in sorted_pairs if proxy_score >= tau_neg)
        den = sum(mx * oracle for (_, oracle, mx) in sorted_pairs)
        # print(f"tp: {tp}, fn: {fn}")
        return Z1_sum / den if den > 0 else 0
    
    def UB(self, mean, std, n, delta):
        # Upper bound of confidence interval
        return mean + std * np.sqrt(np.log(2 / delta) / (2 * n))

    def LB(self, mean, std, n, delta):
        # Lower bound of confidence interval
        return mean - std * np.sqrt(np.log(2 / delta) / (2 * n))

    def find_optimal_pair(self, t_pos, t_neg):
        # Sort t_pos in ascending order to minimize t_pos, and t_neg in descending order to maximize t_neg
        t_pos_sorted = sorted(t_pos)
        t_neg_sorted = sorted(t_neg, reverse=True)

        for neg in t_neg_sorted:
            for pos in t_pos_sorted:
                if pos > neg:
                    # Found a valid pair, return it
                    best_pair = (pos, neg)
                    return best_pair

        # If no valid pair is found, raise an error
        print("Error: No valid (tau_pos, tau_neg) pair found where tau_pos > tau_neg.")
        
        # Fall back to return best recall
        # TODO: or do not use approximation in such case, just do 1,0
        print("Falling back to return the best recall.")
        return (t_neg_sorted[0], t_neg_sorted[0])

def list_to_csv(data_list, file_name):
    """
    Writes a list to a CSV file with each item on a new row.

    Parameters:
        data_list (list): The list of items to write to the CSV file.
        file_name (str): The name of the output CSV file.

    Returns:
        None
    """
    import csv
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write each item as a separate row
        for item in data_list:
            writer.writerow([item])
            
    print(f"Data has been written to {file_name}")
    
class ApproximateJoin():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, threshold=0.1, case="search-filter"):
        return self.importance_sampling_run(queries_df, corpus_df, recall_target, precision_target, threshold, case="map-search-filter", prob_gtd=0.8, seed=42, is_weight=0.5, num_quantile=1000)

    def importance_sampling_run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, threshold=0.1, case="map-search-filter", prob_gtd=0.8, seed=42, is_weight=0.5, num_quantile=1000, proxy_df=None):
        if seed >= 0:
            np.random.seed(seed)
        print("start importance sampling")
        start_t = time.time()

        # Get proxy result
        if not proxy_df:
            K = len(corpus_df)
            if case == "map-search-filter":
                # Run the computation and save the result to the file
                map_df = queries_df.sem_map(
                    "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated medical reactions.",
                    examples=map_dem_df,
                    suffix="map_preds",
                )
                proxy_df = map_df.sem_sim_join(corpus_df, left_on="map_preds", right_on="reaction", K=K)
            elif case == "search-filter":
                K = len(corpus_df)
                proxy_df = queries_df.sem_sim_join(corpus_df, left_on="patient_description", right_on="reaction", K=K)
            else:
                raise ValueError("{case} not supported, please use 'search-filter' or 'map-search-filter'")

        proxy_df = self.quantile_scores(proxy_df, '_scores', num_quantile=num_quantile)
        proxy_df[['_scores']].to_csv(f"biodex_{case}_full_proxy_score_{proxy_df.shape[0]}.csv", index=True) # save intermediate result for debugging

        # Importance sampling
        num_samples = int(proxy_df.shape[0] * threshold)
        weights = np.sqrt(proxy_df['_scores'].values)
        
        # Defensive Mixing
        weights = is_weight * weights / np.sum(weights) + (1 - is_weight) * np.ones_like(weights) / len(proxy_df)
        weights = weights/np.sum(weights) # differ by some very small floating point amount, need to renormalize

        # Weighted Sampling
        sampled_indices = np.random.choice(proxy_df.index, size=num_samples, p=weights, replace=False)
        list_to_csv(sampled_indices, "biodex_sample_indices.csv")
        sampled_proxy_df = proxy_df.iloc[sampled_indices].copy()

        # Calculate reweighting factor m(x)
        sampled_weights = weights[sampled_indices]
        sampled_proxy_df['m'] = (1 / len(proxy_df)) / sampled_weights
        
        # Get oracle result
        unique_sample_df = sampled_proxy_df.drop_duplicates(subset=['patient_description', 'reaction'])
        oracle_df = unique_sample_df.sem_filter(
            "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
        )

        # Map oracle result to proxy result for validation
        sampled_proxy_df['proxy_index'] = sampled_proxy_df.index
        oracle_df = oracle_df.copy()
        oracle_df['oracle_index'] = oracle_df.index

        merged = pd.merge(sampled_proxy_df, oracle_df, on=['patient_description', 'reaction', '_scores', 'm'], how='left', indicator=True)
        merged['oracle'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)

        # Creating score_df with 'proxy_score' and 'oracle'
        score_df = merged[['_scores', 'oracle', 'm']].rename(columns={'_scores': 'proxy_score'})
        score_df.to_csv(f"biodex_{case}_recall_{is_weight}_sampled_proxy_df_{num_samples}.csv", index=True) # save intermediate result for debugging

        # Find threshold that meet target recall
        delta = (1 - prob_gtd) / 2    # account for precision
        t_neg_candidates = self.weighted_find_tau_neg(score_df, recall_target, delta=delta)

        # Find threshold that meet target precision
        # ---------------------------------------------------------------------
        S0_indices = np.random.choice(proxy_df.index, size=int(num_samples/2), p=weights, replace=False)
        S0_proxy_df = proxy_df.iloc[S0_indices].copy()

        # Calculate reweighting factor m(x)
        S0_weights = weights[S0_indices]
        S0_proxy_df['m'] = (1 / len(proxy_df)) / S0_weights
        
        # Get oracle result
        S0_unique = S0_proxy_df.drop_duplicates(subset=['patient_description', 'reaction'])
        S0_oracle = S0_unique.sem_filter(
            "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
        )

        # Map oracle result to proxy result for validation
        S0_proxy_df['proxy_index'] = S0_proxy_df.index
        S0_oracle = S0_oracle.copy()
        S0_oracle['oracle_index'] = S0_oracle.index
        
        merged = pd.merge(S0_proxy_df, S0_oracle, on=['patient_description', 'reaction', '_scores', 'm'], how='left', indicator=True)
        merged['oracle'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)

        # Creating score_df with 'proxy_score' and 'oracle'
        S0_score_df = merged[['_scores', 'oracle', 'm']].rename(columns={'_scores': 'proxy_score'})
        S0_score_df.to_csv(f"search_filter_score_df_{num_samples}.csv", index=False) # save intermediate result for debugging

        Z = [oracle * mx for oracle, mx in S0_score_df[['oracle', 'm']].values]
        mean_z = np.mean(Z) if Z else 0
        std_z = np.std(Z) if Z else 0
        delta = (1 - prob_gtd)
        n_match = len(proxy_df) * self.UB(mean_z, std_z, int(num_samples/2), delta/2)
        
        # Resample on a pruned proxy_df
        proxy_df['weight'] = weights
        proxy_df = proxy_df.sort_values(by='_scores', ascending=False)
        proxy_df_prime = proxy_df.head(int(n_match / precision_target))

        S1_indices = np.random.choice(proxy_df_prime.index, size=int(num_samples/2), p=proxy_df_prime['weight']/np.sum(proxy_df_prime['weight']), replace=False)
        S1_proxy_df = proxy_df.iloc[S1_indices].copy()
        
        # Get S1 oracle
        S1_unique = S1_proxy_df.drop_duplicates(subset=['patient_description', 'reaction'])
        S1_oracle = S1_unique.sem_filter(
            "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?",
        )
        
        # Map oracle result to proxy result for validation
        S1_proxy_df['proxy_index'] = S1_proxy_df.index
        S1_oracle = S1_oracle.copy()
        S1_oracle['oracle_index'] = S1_oracle.index
        
        merged = pd.merge(S1_proxy_df, S1_oracle, on=['patient_description', 'reaction', '_scores'], how='left', indicator=True)
        merged['oracle'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)

        S1_score_df = merged[['_scores', 'oracle']].rename(columns={'_scores': 'proxy_score'})
        # Find threshold that meet target recall
        t_pos_candidates = self.weighted_find_tau_pos(S1_score_df, precision_target, delta=delta)
        
        # Find valid pair
        valid_pairs = [(t_pos, t_neg) for t_pos in t_pos_candidates for t_neg in t_neg_candidates if t_pos > t_neg]
        if not valid_pairs:
            print("No valid (tau_pos, tau_neg) pair found where tau_pos > tau_neg.")
            valid_pairs = [(1.0, 0.0)]
        
        # Maximize t_neg and minimize t_pos
        t_pos, t_neg = self.find_optimal_pair(t_pos_candidates, t_neg_candidates)
        end_t = time.time()
        
        # Statistics
        oracle_calls = sum(1 for x in proxy_df['_scores'] if t_pos > x >= t_neg)

        # Final achieved precision and recall
        pairs = zip(score_df['proxy_score'], score_df['oracle'])
        r = self.recall(t_pos, t_neg, pairs)
        print(f"achieved recall: {r}")
        pairs = zip(S1_score_df['proxy_score'], S1_score_df['oracle'])
        p = self.precision(t_pos, t_neg, pairs)
        print(f"achieved precision: {p}")
        
        print(f"{case}:    t-pairs: ({t_pos}, {t_neg}) , latency: {end_t - start_t}")

        return t_pos, t_neg, oracle_calls, (end_t - start_t), proxy_df  # qid

    
    def weighted_find_tau_neg(self, score_df, recall_target, delta):
        score_df.to_csv(f"score_df_{(len(score_df))}.csv", index=False)
        sorted_pairs = sorted(set(zip(score_df['proxy_score'], score_df['oracle'], score_df['m'])), key=lambda x: x[0], reverse=True)
        # Initialize with an initial threshold for positive and negative
        best_combination = (1, 0)  # t_pos, t_neg

        # Find tau_negative based on recall
        tau_neg_0 = max(x[0] for x in sorted_pairs[::-1] if self.weighted_recall(best_combination[0], x[0], sorted_pairs) >= recall_target)
        best_combination = (best_combination[0], tau_neg_0)

        # Statistical correction for recall
        Z1 = [mx if oracle == 1 else 0 for proxy_score, oracle, mx in sorted_pairs if proxy_score >= best_combination[1]]
        Z2 = [mx if oracle == 1 else 0 for proxy_score, oracle, mx in sorted_pairs if proxy_score < best_combination[1]]

        mean_z1 = np.mean(Z1) if Z1 else 0
        std_z1 = np.std(Z1) if Z1 else 0
        mean_z2 = np.mean(Z2) if Z2 else 0
        std_z2 = np.std(Z2) if Z2 else 0

        # Update to delta over 2
        sample_size = len(sorted_pairs)
        corrected_recall_target = self.UB(mean_z1, std_z1, sample_size, delta/2)/(self.UB(mean_z1, std_z1, sample_size, delta/2) + self.LB(mean_z2, std_z2, sample_size, delta/2))
        
        tau_neg_candidates = [x[0] for x in sorted_pairs[::-1] if self.weighted_recall(best_combination[0], x[0], sorted_pairs) >= corrected_recall_target]
        if not tau_neg_candidates:
            tau_neg_candidates.append(0)

        return tau_neg_candidates

    def parse_proxy_score(self, df, method='', num_quantile=1000):
        if method == 'quantile':
            return self.quantile_scores(df, '_scores', num_quantile=num_quantile)
        elif method == "clip":
            return self.clip_scores(df, '_scores')
        return df

    def clip_scores(self, df, col):
        df[col] = np.clip(df[col], 0, 1)
        return df

    def quantile_scores(self, df, col, num_quantile):
        # Get the scores from the specified column
        scores = df[col].values
        
        # Compute the quantile boundaries
        quantile_values = np.percentile(scores, np.linspace(0, 100, num_quantile + 1))
        
        # Digitize the scores into quantiles and normalize
        df[col] = (np.digitize(scores, quantile_values) - 1) / num_quantile
        
        return df

    def weighted_find_tau_pos(self, score_df, precision_target, delta):
        score_df.to_csv(f"score_df_{(len(score_df))}.csv", index=False)
        sorted_pairs = sorted(set(zip(score_df['proxy_score'], score_df['oracle'])), key=lambda x: x[0], reverse=True)

        # Statistical correction for precision
        tau_pos_candidates = []
        for proxy_score, oracle in sorted_pairs:
            possible_threshold = proxy_score
            Z = [1 if oracle == 1 else 0 for proxy_score, oracle in sorted_pairs if proxy_score >= possible_threshold]
            mean_z = np.mean(Z) if Z else 0
            std_z = np.std(Z) if Z else 0
            p_l = self.LB(mean_z, std_z, sum(Z), delta / (2 * len(score_df)) )
            if p_l > precision_target:
                tau_pos_candidates.append(possible_threshold)

        if not tau_pos_candidates:
            tau_pos_candidates.append(1)

        return tau_pos_candidates


    def find_optimal_thresholds(self, score_df, recall_target, precision_target, prob_gtd=0.8):
        sorted_pairs = sorted(set(zip(score_df['proxy_score'], score_df['oracle'])), key=lambda x: x[0], reverse=True)
        # Initialize with an initial threshold for positive and negative
        best_combination = (1, 0)  # t_pos, t_neg

        # Find tau_negative based on recall
        tau_neg_0 = max(x[0] for x in sorted_pairs[::-1] if self.recall(best_combination[0], x[0], sorted_pairs) >= recall_target)
        best_combination = (best_combination[0], tau_neg_0)

        # Statistical correction for recall
        Z1 = [1 if (proxy_score >= best_combination[1] and oracle == 1) else 0 for (proxy_score, oracle) in sorted_pairs] # Z1 is the true positive set using temp-tau
        Z2 = [1 if (proxy_score < best_combination[1] and oracle == 1) else 0 for (proxy_score, oracle) in sorted_pairs] # Z2 is the false negative set using temp-tau

        mean_z1 = np.mean(Z1) if Z1 else 0
        std_z1 = np.std(Z1) if Z1 else 0
        print(f"mean_z1={mean_z1}")
        print(f"std_z1={std_z1}")
        mean_z2 = np.mean(Z2) if Z2 else 0
        std_z2 = np.std(Z2) if Z2 else 0

        # Sta
        delta = (1 - prob_gtd)
        sample_size = len(sorted_pairs)
        corrected_recall_target = self.UB(mean_z1, std_z1, sample_size, delta/2)/(self.UB(mean_z1, std_z1, sample_size, delta/2) + self.LB(mean_z2, std_z2, sample_size, delta/2))
        print(f"corrected_recall_target = {corrected_recall_target}")
        
        tau_neg_candidates = [x[0] for x in sorted_pairs[::-1] if self.recall(best_combination[0], x[0], sorted_pairs) >= corrected_recall_target]
        if not tau_neg_candidates:
            tau_neg_candidates.append(0)

        # Statistical correction for precision
        tau_pos_candidates = []
        for proxy_score, oracle in sorted_pairs:
            possible_threshold = proxy_score
            Z = [1 if oracle == 1 else 0 for proxy_score, oracle in sorted_pairs if proxy_score >= possible_threshold]
            mean_z = np.mean(Z) if Z else 0
            std_z = np.std(Z) if Z else 0
            p_l = self.LB(mean_z, std_z, sum(Z), delta / len(sorted_pairs))
            if p_l > precision_target:
                tau_pos_candidates.append(possible_threshold)

        if not tau_pos_candidates:
            tau_pos_candidates.append(1)

        # Find valid pair
        valid_pairs = [(t_pos, t_neg) for t_pos in tau_pos_candidates for t_neg in tau_neg_candidates if t_pos > t_neg]
        if not valid_pairs:
            print("No valid (tau_pos, tau_neg) pair found where tau_pos > tau_neg.")
            valid_pairs = [(1, 0)]
        
        # Maximize t_neg and minimize t_pos
        best_combination = self.find_optimal_pair(tau_pos_candidates, tau_neg_candidates)

        oracle_calls = sum(1 for x in sorted_pairs if best_combination[0] > x[0] > best_combination[1])
        print(f"# sample to oracle: {oracle_calls}")

        # Final achieved precision and recall
        r = self.recall(best_combination[0], best_combination[1], sorted_pairs)
        print(f"achieved recall: {r}")
        p = self.precision(best_combination[0], best_combination[1], sorted_pairs)
        print(f"achieved precision: {p}")
        return best_combination[0], best_combination[1]

    def recall(self, tau_pos, tau_neg, sorted_pairs):
        # Function to compute recall
        tp = sum(1 for (proxy_score, oracle) in sorted_pairs if proxy_score >= tau_neg and oracle == 1)
        fn = sum(1 for (proxy_score, oracle) in sorted_pairs if proxy_score < tau_neg and oracle == 1)
        # print(f"tp: {tp}, fn: {fn}")
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def precision(self, tau_pos, tau_neg, sorted_pairs):
        # Function to compute recall
        Z = [1 if oracle == 1 else 0 for proxy_score, oracle in sorted_pairs if proxy_score >= tau_pos]
    
        return sum(Z) / len(Z) if len(Z) > 0 else 0
    
    def weighted_recall(self, tau_pos, tau_neg, sorted_pairs):
        # Function to compute recall
        Z1_sum =  sum(mx if oracle == 1 else 0 for proxy_score, oracle, mx in sorted_pairs if proxy_score >= tau_neg)
        den = sum(mx * oracle for (_, oracle, mx) in sorted_pairs)
        # print(f"tp: {tp}, fn: {fn}")
        return Z1_sum / den if den > 0 else 0
    
    def UB(self, mean, std, n, delta):
        # Upper bound of confidence interval
        return mean + std * np.sqrt(np.log(2 / delta) / (2 * n))

    def LB(self, mean, std, n, delta):
        # Lower bound of confidence interval
        return mean - std * np.sqrt(np.log(2 / delta) / (2 * n))

    def find_optimal_pair(self, t_pos, t_neg):
        # Sort t_pos in ascending order to minimize t_pos, and t_neg in descending order to maximize t_neg
        t_pos_sorted = sorted(t_pos)
        t_neg_sorted = sorted(t_neg, reverse=True)

        for neg in t_neg_sorted:
            for pos in t_pos_sorted:
                if pos > neg:
                    # Found a valid pair, return it
                    best_pair = (pos, neg)
                    return best_pair

        # If no valid pair is found, raise an error
        print("Error: No valid (tau_pos, tau_neg) pair found where tau_pos > tau_neg.")
        
        # Fall back to return best recall
        # TODO: or do not use approximation in such case, just do 1,0
        print("Falling back to return the best recall.")
        return (t_neg_sorted[0], t_neg_sorted[0])

class Join(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, RT=0.8, PT=0.8, learning_threshold=0.01, num_quantile=100000):
        start_t = time.time()
        # Learn map-search-filter
        msf_t_pos, msf_t_neg, msf_num_oracle, _ = ApproximateJoin().importance_sampling_run(
            queries_df, corpus_df, recall_target=RT, precision_target=PT, 
            threshold=learning_threshold, case='map-search-filter',
            num_quantile=num_quantile)
        
        # Learn search-filter
        sf_t_pos, sf_t_neg, sf_num_oracle, _ = ApproximateJoin().importance_sampling_run(
            queries_df, corpus_df, recall_target=RT, precision_target=PT, 
            threshold=learning_threshold, case='map-search-filter',
            num_quantile=num_quantile)
        
        if msf_num_oracle <= sf_num_oracle:
            grouped_df = ApproxMapRetrieveFilter.run(msf_t_pos, msf_t_neg)
        else:
            grouped_df = ApproxRetrieveFilter.run(sf_t_pos, sf_t_neg)
        end_t = time.time()

        return grouped_df, (end_t - start_t)  # qid

class JoinCascade(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, range=250, name="llama"):
        # lotus.settings.
        start_t = time.time()

        # "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?"
        print(f"corpus_df = {corpus_df.shape}")

        cascade_args = CascadeArgs(
            recall_target=recall_target,
            precision_target=precision_target,
            sampling_percentage=0.0001,
            map_instruction=map_instruction, map_examples=map_dem_df,
            cascade_IS_weight=0.9,
            cascade_IS_random_seed=42,
            cascade_IS_max_sample_range=range
        )
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)

        end_t = time.time()

        answers_df.to_csv(f"biodex_cascade_answers_for_lm_rerank_{name}.csv", index=True)
        
        print(f"Time taken: {end_t - start_t}")
        print(f"answers_df = {answers_df.columns.tolist()}")
        print(f"stats = {stats}")

        # post process for checking answers
        # convert reaction_list to string so we can groupby
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


class rerankJoinCascade(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, range=250, name="Llama", top_k=25):
        start_t = time.time()
        
        # "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?"
        print(f"corpus_df = {corpus_df.shape}")
        
        cascade_args = CascadeArgs(
            recall_target=recall_target, 
            precision_target=precision_target,
            sampling_percentage=0.0001,
            map_instruction=map_instruction, map_examples=map_dem_df,
            cascade_IS_weight=0.9,
            cascade_IS_random_seed=42,
            cascade_IS_max_sample_range=range
        )
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)
        
        end_t_1 = time.time()
        
        # Rerank the answer with LLM
        def to_comma_separated(val):
            """
            Safely convert val (which could be a list or a string representing a list)
            into a comma-separated string.
            """
            
            if isinstance(val, list):
                # Already a list of strings
                return ", ".join(val)
            elif isinstance(val, str):
                # Possibly a string like "['foo', 'bar']"
                # Try literal_eval to see if it's a valid Python list
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        # It's a real list, convert to comma-separated
                        return ", ".join(parsed)
                    else:
                        # Not a list—just return the original string
                        return val
                except (SyntaxError, ValueError):
                    # It's not a parseable list, return as is
                    return val
            else:
                # Fallback: convert whatever it is to string
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
            f"Given {{patient_description}}, pick the {top_k} most applicable adverse drug reactions from the options "
            f"that are directly expressed in the following list: {{pred_reaction}}. "
            "Rank from most applicable to least applicable. "
            "Always write your answer as a list of comma-separated adverse drug reactions only and nothing else."
        )

        rerank_num_lm = grouped_df.shape[0]

        grouped_df = grouped_df.sem_map(
            rerank_prompt
        )
        
        end_t_2 = time.time()
        
        print(f"stats = {stats}")
        print(f"rerank used {rerank_num_lm} samples")
        print(f"Total Time taken: {end_t_2 - start_t}")
        print(f"no rerank took: {end_t_1 - start_t}")

        # Parse output
        known_prefixes = [
            f"Based on the patient description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the provided Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            "Here is the answer:\n\n",
            f"Here is the list of the {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions from the options, ranked from most applicable to least applicable:"
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
        
        return grouped_df, (end_t_2 - start_t) 
    

class rerank(Pipeline):
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
                # Already a list of strings
                return ", ".join(val)
            elif isinstance(val, str):
                # Possibly a string like "['foo', 'bar']"
                # Try literal_eval to see if it's a valid Python list
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        # It's a real list, convert to comma-separated
                        return ", ".join(parsed)
                    else:
                        # Not a list—just return the original string
                        return val
                except (SyntaxError, ValueError):
                    # It's not a parseable list, return as is
                    return val
            else:
                # Fallback: convert whatever it is to string
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
            f"Given {{patient_description}}, pick the {top_k} most applicable adverse drug reactions from the options "
            f"that are directly expressed in the following list: {{pred_reaction}}. "
            "Rank from most applicable to least applicable. "
            "Always write your answer as a list of comma-separated adverse drug reactions only and nothing else."
        )

        rerank_num_lm = grouped_df.shape[0]

        grouped_df = grouped_df.sem_map(
            rerank_prompt
        )
        

        end_t = time.time()

        grouped_df.to_csv(f"biodex_reranked_answers_{name}.csv", index=True)

        # Parse output
        known_prefixes = [
            f"Based on the patient description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the provided Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            "Here is the answer:\n\n",
            f"Here is the list of the {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions from the options, ranked from most applicable to least applicable:"
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



class parseRerankJoinCascade(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, queries_df, corpus_df):
        top_answer_df = pd.read_csv("biodex_reranked_answers.csv", index_col=0)
        
        start_t = time.time()
        known_prefixes = [
            "Based on the patient description, the 25 most applicable adverse drug reactions are:\n\n",
            "Based on the Patient_description, the 25 most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            "Based on the Patient_description, the 25 most applicable adverse drug reactions are:\n\n",
            "Based on the provided Patient_description, the 25 most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            "Here is the list of 25 most applicable adverse drug reactions:\n\n",
            "Here is the answer:\n\n",
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
        
        top_answer_df["_map"] = top_answer_df["_map"].fillna("").apply(
            lambda x: remove_known_prefixes(x, known_prefixes)
        )
        
        top_answer_df["pred_reaction"] = top_answer_df["_map"].apply(
            lambda x: [reaction.strip() for reaction in x.split(",") if reaction.strip()]
        )
        top_answer_df["pred_reaction"] = top_answer_df["pred_reaction"].apply(
            lambda reactions: [f"'{r}'" for r in reactions]
        )
        end_t = time.time()

        
        print(f"top_answer_df = {top_answer_df}")

        return top_answer_df, (end_t - start_t) 
    

class checkRes(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, queries_df, corpus_df, csv_file="biodex_reranked_answers.csv"):
        start_t = time.time()
        top_answer_df = pd.read_csv(csv_file, index_col=0)
        
        def parse_list_col(x):
            """
            Try to parse x (a string) as a Python list using literal_eval.
            If parsing fails, return x unchanged.
            """
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    # If x is not a valid Python literal, just return x
                    return x
            return x
        
        # Convert string -> list for the specified columns
        for col in ["pred_reaction", "reactions_list"]:
            if col in top_answer_df.columns:
                top_answer_df[col] = top_answer_df[col].apply(parse_list_col)

        
        end_t = time.time()

        return top_answer_df, (end_t - start_t) 


class JoinCascadeDocETL(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, range=250, name="llama"):
        # lotus.settings.
        start_t = time.time()

        # "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = """Can the following condition be found in the following medical article?
        Medical article: {patient_description} 
        
        Condition we are looking for: {reaction}


        Determine if {reaction} is described in the medical article, considering the context and meaning beyond just the presence of individual words."""
        print(f"corpus_df = {corpus_df.shape}")

        cascade_args = CascadeArgs(
            recall_target=recall_target,
            precision_target=precision_target,
            sampling_percentage=0.00008218, #len = 500
            map_instruction=map_instruction,
            # map_examples=map_dem_df,
            cascade_IS_weight=0.9,
            cascade_IS_random_seed=42,
            cascade_IS_max_sample_range=range
        )
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)

        end_t = time.time()

        answers_df.to_csv(f"biodex_cascade_answers_for_lm_rerank_{name}.csv", index=True)
        
        print(f"Time taken: {end_t - start_t}")
        print(f"answers_df = {answers_df.columns.tolist()}")
        print(f"stats = {stats}")

        # post process for checking answers
        # convert reaction_list to string so we can groupby
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
    ts = BiodexTester(n_samples=250, truncation_limit=8000)

    # Approximate map-search-filter
    # ---------------------------------------------------------------------
    # targets = [ 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,]
    # targets = [0.9]
    # prob_gtds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # for pg in prob_gtds:
    #     results = {}
    #     for target in targets:
    #         t_pos, t_neg, num_oracle, lat, _ = ApproximateJoinTest().importance_sampling_run(
    #             ts.queries_df, ts.corpus_df, recall_target=target, precision_target=target, case="map-search-filter", threshold=0.0001, prob_gtd=pg, seed=42, is_weight=0.5, num_quantile=100000)
    #         results[target] = (t_pos, t_neg, num_oracle, lat)
    #     print(results)

    # map-search-filter: quantile 100000
    # t_pairs = { 0.3: (1.0, 0.99752),
    #             0.4: (1.0, 0.99761), 
    #             # 0.5: (1.0, 0.99751),
    #             0.6: (1.0, 0.99747), 
    #             0.7: (1.0, 0.99729), 
    #             # 0.8: (1.0, 0.99751),
    #             0.9: (1.0, 0.99751),
    #             0.99: (1.0, 0.99465)}
    
    # t_pairs = { 0.99: (1.0, 0.994), 0.8: (1.0, 0.995)}

    # for target, t_pair in t_pairs.items():
    #     t_pos, t_neg =  t_pair
    #     ts.add_pipeline(ApproxMapRetrieveFilter(t_pos=t_pos, t_neg=t_neg, num_quantile=1000))


    # Approximate search-filter
    # ---------------------------------------------------------------------
    # targets = [0.7]
    # results = {}
    # for target in targets:
    #     t_pos, t_neg, num_oracle, lat = ApproximateJoin().importance_sampling_run(
    #         ts.queries_df, ts.corpus_df, recall_target=target, precision_target=target, threshold=0.0001, num_quantile=100000)
    #     results[target] = (t_pos, t_neg, num_oracle, lat)
    # print(results)

    # t_pairs = { 0.8:  (1.0, 0.99012),
    #             0.7:  (1.0, 0.99117), 
    #             0.6:  (1.0, 0.9888) }
        
    # for target, t_pair in t_pairs.items():
    #     t_pos, t_neg = t_pair
    #     ts.add_pipeline(ApproxRetrieveFilter(t_pos=t_pos, t_neg=t_neg))

        
    # With cost model
    # ---------------------------------------------------------------------
    # ts.add_pipeline(Join(RT=0.7, PT=0.7, learning_threshold=0.0001))
    # ts.add_pipeline(JoinCascade(recall_target=0.4, precision_target=0.4))
    ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=50))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=100))
    # ts.add_pipeline(JoinCascadeDocETL(recall_target=0.95, precision_target=0.95, range=100, name="GPT4o-mini"))
    # ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9, range=50, name="GPT4o-mini"))
    # ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9, range=100, name="llama"))
    # ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9, range=250, name="GPT4o-mini"))
    # ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9, range=500, name="GPT4o-mini"))
    # ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9, range=1000, name="GPT4o-mini"))
    
    
    
    # ts.add_pipeline(rerank(name="GPT4o-mini-full"))
    # ts.add_pipeline(rerank())
    # ts.add_pipeline(rerankJoinCascade(recall_target=0.9, precision_target=0.9, range=150, top_k=25, name="GPT4o-mini"))
    # ts.add_pipeline(checkRes(csv_file="biodex_results/JoinCascade/nsamples=250_recall_target=0.7_precision_target=0.7_range=250/res.csv"))
    # ts.add_pipeline(parseRerankJoinCascade())
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=200))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=250))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=300))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=350))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=400))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=450))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=500))
    # ts.add_pipeline(JoinCascade(recall_target=0.3, precision_target=0.3))
    # ts.add_pipeline(JoinCascade(recall_target=0.8, precision_target=0.8))
    # ts.add_pipeline(JoinCascade(recall_target=0.5, precision_target=0.5))

    # RUN ALL PIPELINES
    # ---------------------------------------------------------------------
    ts.test_pipelines()

    # PRINT SUMMARY OF PIPELINE RESULTS
    # print(ts.summarize_pipeline_results("biodex_results", ts.n_samples, ["ApproxMapRetrieveFilter"]))
    print(ts.summarize_pipeline_results("biodex_results", ts.n_samples, ["JoinCascade"]))

    # for target, lat in zip(t_pairs.keys(), out):
    #     print(f"target={target}, lat={lat}")
        

    
