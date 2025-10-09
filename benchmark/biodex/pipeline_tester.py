import json
import os
from abc import abstractmethod
from typing import List, Tuple

import pandas as pd

import lotus


# only use kwargs
class Pipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, queries_df, corpus_df, **kwargs):
        return self.run(queries_df, corpus_df, **kwargs)

    @abstractmethod
    # should return df of qid, ids (as a list) and a latency number
    def run(self, query_df, corpus_df, *args, **kwargs) -> Tuple[pd.DataFrame, float]:
        raise NotImplementedError


class PipelineTester:
    def __init__(self, n_samples=200):
        self.n_samples = n_samples

        print("Setting configs")
        self.set_configs()

        print("Loading queries")
        self.queries_df = self.load_queries(n_samples)

        print("Loading corpus")
        self.corpus_df = self.load_corpus()

        self.results_dir = self.set_results_dir()

        self.pipelines = []

    @abstractmethod
    def set_results_dir(self):
        pass

    @abstractmethod
    def set_configs(self):
        pass

    @abstractmethod
    def load_queries(self, n_samples) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_corpus(self) -> pd.DataFrame:
        pass

    @abstractmethod
    # returns a df with a column for each metric and row for each query
    def compute_metrics(self, res_df, gt_col_name, pred_col_name) -> pd.DataFrame:
        pass

    def add_pipeline(self, pipeline: Pipeline):
        self.pipelines.append(pipeline)
        print(f"Added {pipeline.__class__.__name__}" f" with kwargs: {pipeline.kwargs}")

    def clear_pipelines(self):
        self.pipelines = []

    def test_pipelines(self):
        for pipeline in self.pipelines:
            print(f"Testing {pipeline.__class__.__name__} with kwargs: {pipeline.kwargs}")
            res_df, latency = pipeline(self.queries_df, self.corpus_df, **pipeline.kwargs)
            # print(res_df)
            print(f"Time taken: {latency}")
            print("\n\n")

            # save results to file
            self.write_results(pipeline, res_df, latency)

    # def get_configs(self):
    #     configs_dict = dict()
    #     configs_dict["lm"] = lotus.settings.lm.__class__.__name__
    #     # configs_dict["lm_model"] = vars(lotus.settings.lm)["kwargs"]
    #     configs_dict["rm"] = lotus.settings.rm.__class__.__name__
    #     configs_dict["max_batch_size"] = lotus.settings.max_batch_size
    #     configs_dict["model_max_tokens"] = lotus.settings.model_params["max_tokens"]
    #     configs_dict["model_temperature"] = lotus.settings.model_params["temperature"]
    #     return configs_dict

    def get_configs(self):
        configs_dict = dict()
        configs_dict["lm"] = lotus.settings.lm.__class__.__name__
        # configs_dict["lm_model"] = vars(lotus.settings.lm)["kwargs"]
        configs_dict["rm"] = lotus.settings.rm.__class__.__name__
        configs_dict["max_batch_size"] = lotus.settings.lm.max_batch_size
        configs_dict["model_max_tokens"] = lotus.settings.lm.max_tokens
        configs_dict["model_temperature"] = lotus.settings.lm.kwargs["temperature"]
        return configs_dict

    def write_results(self, pipeline, res_df, latency):
        pipeline_dir = pipeline.__class__.__name__
        param_dir = f"nsamples={self.n_samples}_" + "_".join([f"{k}={v}" for k, v in pipeline.kwargs.items()])

        os.makedirs(os.path.join(self.results_dir, pipeline_dir), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, pipeline_dir, param_dir), exist_ok=True)

        # write res_df
        file_name = "res.csv"
        res_df.to_csv(
            os.path.join(self.results_dir, pipeline_dir, param_dir, file_name),
            index=False,
        )

        # save avg_latency
        with open(os.path.join(self.results_dir, pipeline_dir, param_dir, "latency.json"), "w") as f:
            json.dump({"total_latency": latency, "n_samples": self.n_samples}, f)

        # save configs
        with open(os.path.join(self.results_dir, pipeline_dir, param_dir, "configs.json"), "w") as f:
            json.dump(self.get_configs(), f)

        # save mean metrics
        metrics_df = self.compute_metrics(res_df)
        means = metrics_df.mean()
        means["avg_latency"] = latency / self.n_samples
        means["qps"] = self.n_samples / latency
        std = metrics_df.std()
        means.to_csv(os.path.join(self.results_dir, pipeline_dir, param_dir, "metrics.csv"))
        std.to_csv(os.path.join(self.results_dir, pipeline_dir, param_dir, "std.csv"))

        print(f"Results saved to {os.path.join(self.results_dir, pipeline_dir, param_dir)}")
        print(means)
        print(std)

    @classmethod
    def summarize_pipeline_results(self, save_dir: str, n_samples: int, pipeline_names: List[str]):
        means = pd.DataFrame()
        for pipeline_dir in os.listdir(save_dir):
            if pipeline_dir not in pipeline_names:
                continue
            for i, param_dir in enumerate(os.listdir(os.path.join(save_dir, pipeline_dir))):
                if f"nsamples={n_samples}" in param_dir:
                    print(f"{i}: {pipeline_dir}/{param_dir}")
                    pipeline_df = pd.read_csv(
                        os.path.join(save_dir, pipeline_dir, param_dir, "metrics.csv"),
                        index_col=0,
                    )
                    # print(pipeline_df)
                    # add pipeline_df as col to means (with name f"{pipeline_dir}_{i}"
                    means[f"{pipeline_dir}_{i}"] = pipeline_df["0"]

        return means

    save_dir = "hover_results"
