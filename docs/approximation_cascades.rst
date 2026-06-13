Optimized Processing with Approximations
========================================

Overview
---------------

LOTUS serves approximations for semantic operators to let you balance speed and accuracy. 
You can set accurayc targets according to the requirements of your application, and LOTUS
will use approximations to optimize the implementation for lower computaitonal overhead, while providing probabilistic accuracy guarantees.
One core technique for providing these approximations is the use of cascades.
Cascades provide a way to optimize certian semantic operators (Join Cascade and Filter Cascade) by blending 
a less costly but potentially inaccurate proxy model with a high-quality oracle model. The method seeks to achieve
preset precision and recall targets with a given probability while controlling computational overhead.

Cascades work by intially using a cheap approximation to score and filters/joins tuples. Using statistically
supported thresholds found from sampling prior, it then assigns each tuple to one of three actions based on the 
proxy's score: accept, reject, or seek clarification from the oracle model. 

When the proxy is accurate, most of the data is resolved quickly and inexpensively, and those not resolved are 
sent to the larger LM. 

Using Cascades
----------------
To use this approximation cascade-based operators, begin by configuring both the main and helper LM using
lotus's configuration settings

.. code-block:: python

    import lotus
    from lotus.models import LM
    from lotus.types import CascadeArgs, ProxyModel

    lotus.settings.configure(
        lm=LM(model="gpt-4o"),
        helper_lm=LM(model="gpt-4o-mini"),
    )

    cascade_args = CascadeArgs(
        recall_target=0.9,
        precision_target=0.9,
        sampling_percentage=0.5,
        failure_probability=0.2,
        proxy_model=ProxyModel.HELPER_LM,
    )

    filtered, stats = df.sem_filter(
        user_instruction="{Course Name} requires a lot of math",
        cascade_args=cascade_args,
        return_stats=True,
    )

CascadeArgs Parameters
----------------------

Accuracy Targets
~~~~~~~~~~~~~~~~

These fields describe the quality/cost tradeoff you want LOTUS to target when
it learns thresholds.

- ``recall_target``: Target recall for the cascade. Increase this when missing
  true positives is costly. Default: ``0.8``.
- ``precision_target``: Target precision for the cascade. Increase this when
  false positives are costly. Default: ``0.8``.
- ``failure_probability``: Allowed probability that the learned thresholds do
  not meet the requested targets. Lower values are more conservative. Default:
  ``0.2``.

Sampling and Calibration
~~~~~~~~~~~~~~~~~~~~~~~~

These fields control how LOTUS samples rows or pairs while learning
thresholds.

- ``sampling_percentage``: Fraction of proxy-scored items sampled for
  threshold learning. Default: ``0.1``.
- ``cascade_IS_weight``: Importance-sampling weight. Higher values bias the
  calibration sample toward high proxy scores; lower values make sampling more
  uniform. Default: ``0.9``.
- ``cascade_IS_max_sample_range``: Maximum prefix of proxy-ranked candidates
  considered for importance sampling. Default: ``200``.
- ``cascade_IS_random_seed``: Optional random seed for reproducible threshold
  sampling. Default: ``None``.
- ``cascade_num_calibration_quantiles``: Number of quantile buckets used to
  calibrate helper-LM probabilities for filter cascades. Default: ``50``.

Proxy Model Selection
~~~~~~~~~~~~~~~~~~~~~

``proxy_model`` chooses the cheap model used before routing uncertain cases to
the main LM.

- ``ProxyModel.HELPER_LM``: Use ``lotus.settings.helper_lm`` as the proxy.
  This is the default for filter cascades and pairwise-judge cascades.
- ``ProxyModel.EMBEDDING_MODEL``: Use the configured retrieval model as an
  embedding proxy where supported.

Filter Cascade Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters apply to ``sem_filter`` and pairwise-judge cascades, because
pairwise judging is implemented through semantic filtering.

- ``helper_filter_instruction``: Optional simplified instruction for the helper
  LM. If omitted, the helper uses the main filter instruction.
- ``filter_pos_cascade_threshold``: Optional precomputed positive threshold.
  Proxy scores at or above this threshold are accepted without the main LM.
- ``filter_neg_cascade_threshold``: Optional precomputed negative threshold.
  Proxy scores at or below this threshold are rejected without the main LM.

``filter_pos_cascade_threshold`` and ``filter_neg_cascade_threshold`` must be
provided together, and the positive threshold must be greater than or equal to
the negative threshold.

Join Cascade Parameters
~~~~~~~~~~~~~~~~~~~~~~~

These parameters apply to ``sem_join`` cascades.

- ``min_join_cascade_size``: Minimum full join size before LOTUS considers a
  join cascade. Default: ``100``.
- ``map_instruction``: Optional instruction for the map-search-filter join
  strategy. This maps left rows into likely right-side values before search.
- ``map_examples``: Optional few-shot examples for ``map_instruction``.
- ``join_cascade_strategy``: Optional fixed join cascade strategy. Supported
  values are ``"search_filter"`` and ``"map_search_filter"``. If omitted,
  LOTUS evaluates both strategies and chooses the cheaper plan.
- ``join_cascade_pos_threshold``: Optional precomputed positive threshold for
  join helper scores.
- ``join_cascade_neg_threshold``: Optional precomputed negative threshold for
  join helper scores.

If ``join_cascade_strategy`` is provided, both join thresholds must also be
provided, and the positive threshold must be greater than or equal to the
negative threshold.

Precomputed Thresholds
~~~~~~~~~~~~~~~~~~~~~~

Thresholds are usually learned automatically. You can provide them manually
when you have already calibrated a cascade and want to skip threshold learning.

.. code-block:: python

    cascade_args = CascadeArgs(
        filter_pos_cascade_threshold=0.62,
        filter_neg_cascade_threshold=0.52,
    )

For LazyFrame pipelines, :class:`lotus.ast.optimizer.CascadeOptimizer` can
learn thresholds on training data and store them in the optimized pipeline.

Interpreting Filter Statistics
------------------------------

For cascade operators, ``return_stats=True`` returns metrics that explain how
much work was handled by the proxy and how much was routed to the main LM.

Example filter stats:

.. code-block:: text

    {
        "pos_cascade_threshold": 0.62,
        "neg_cascade_threshold": 0.52,
        "filters_resolved_by_helper_model": 95,
        "filters_resolved_by_large_model": 8,
        "num_routed_to_helper_model": 95,
        "cascade_args": CascadeArgs(...),
    }

Here is a detailed explanation of each metric

1. **pos_cascade_threshold**
   The Minimum score above which tuples are automatically rejected by the helper model. In the above example, any tuple with a 
   score above 0.62 is accepted without the need for the oracle LM.

2. **neg_cascade_threshold**
   The maximum score below which tuples are automatically rejected by the helper model.  
   Any tuple scoring below 0.52 is rejected without involving the oracle LM.

3. **filters_resolved_by_helper_model**  
   The number of tuples conclusively classified by the helper model.  
   A value of 95 indicates that the majority of items were efficiently handled at this stage.

4. **filters_resolved_by_large_model**  
   The count of tuples requiring the oracle model’s intervention.  
   Here, only 8 items needed escalation, suggesting that the chosen thresholds are effective.

5. **num_routed_to_helper_model**  
   The total number of items initially processed by the helper model.  
   Since 95 items were routed, and only 8 required the oracle, this shows a favorable balance between cost and accuracy.

6. **cascade_args**
   Copy of the cascade configuration, including learned
  thresholds.
