Instructions for running biodex pipeline:

Setup:
```
conda create lotus-bench python=3.10 -y
conda activate lotus-bench
pip install -r requirements.txt
```

Run the biodex tester file which includes the following two programs:
- Join only
- JoinAndRerank (which uses follows the join with listwise re-ranking, resembling the orignal logic in https://arxiv.org/pdf/2401.12178, and comparble to the docetl baseline)
Both are configured with gpt-4o-mini as the LLM and text-embeddings-3-small as the retreiver model

```
python biodex_tester.py
```

This will run both the Join and JoinAndRerank programs and output summary metrics for each, including RP@K and total_latency, in the directory biodex_results.
Note that the latency for the JoinAndRerank pipeline measures the ranking step only, reusing the Join outputs as intermediate results, so latency numbers from the Join should be added to get the latency for the full JoinAndRerank pipeline.


You can find example results I ran earlier in biodex_results/Mar-17-25, which gave the following for the full JoinAndRerank 

```
+-------------------------+--------+---------+-------------+
|                         |   RP@5 |   RP@10 | Latency (s) |
+=========================+========+=========+=============+
| Lotus JoinAndRerank     | .2669  |  .2722  |  566.55     |
+-------------------------+--------+---------+-------------+
```