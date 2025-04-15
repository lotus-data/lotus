Optimized Processing with Approximations
=======================

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
   from lotus.types import CascadeArgs


   gpt_4o_mini = LM("gpt-4o-mini")
   gpt_4o = LM("gpt-4o")

   lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)


Once the LMs are set up, specify the cascade parameters-like recall and precision targets, sampling percentage, and 
the acceptable failure probability-using the CascadeArgs object. 

.. code-block:: python

   cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.5, failure_probability=0.2)

After preparing the arguments, call the semantic operator method on the DataFrame

.. code-block:: python

   df, stats = df.sem_filter(user_instruction=user_instruction, cascade_args=cascade_args, return_stats=True)

Note that these parameters guide the trade-off between speed and accuracy when applying the cascade operators

Interpreting Output Statistics
-------------------------------
For cascade operators, Output statistics will contain key performance metrics.

An Example output statistic: 

.. code-block:: text

   {'pos_cascade_threshold': 0.62, 
   'neg_cascade_threshold': 0.52, 
   'filters_resolved_by_helper_model': 95, 
   'filters_resolved_by_large_model': 8, 
   'num_routed_to_helper_model': 95}

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