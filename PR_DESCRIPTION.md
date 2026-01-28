## Purpose
Closes #200 (Test-Time Scaling)
Closes #196 (Audio Data Support)

This PR implements two major features for LOTUS:
1.  **Audio Data Support**: Adds the ability to process audio files using semantic operators, enabling multimodal pipelines with audio inputs.
2.  **Test-Time Scaling (Ensembling)**: Adds test-time scaling strategies to `sem_filter`, allowing users to trade off compute for accuracy by aggregating multiple samples.

## Summary of Changes

### Audio Data Support
-   **New `AudioArray` & `AudioDtype`**: Implemented in `lotus/dtype_extensions/audio.py` to handle audio files (.wav, .mp3, etc.) locally and efficiently.
-   **Multimodal Integration**: Updated `lotus/templates/task_instructions.py`:
    -   `context_formatter` now handles `audio` data, formatting it as `input_audio` for LLM APIs.
    -   `df2multimodal_info` automatically detects `AudioDtype` columns and extracts base64 audio data.
    -   `merge_multimodal_info` supports merging audio data.

### Test-Time Scaling (Ensembling)
-   **`Ensemble` Module**: Created `lotus/sem_ops/ensembling.py` implementing strategies:
    -   `MAJORITY_VOTE`, `WEIGHTED_AVERAGE`, `CONSENSUS`, `CONFIDENCE_THRESHOLD`.
-   **`sem_filter` Integration**: Updated `sem_filter` to accept test-time scaling parameters:
    -   `n_sample`: Number of samples to generate (default: 1).
    -   `ensemble`: Strategy to use (e.g., `EnsembleStrategy.MAJORITY_VOTE`).
    -   `temperature`: Sampling temperature.
-   **Rich Output**: Updated `SemanticFilterOutput` in `lotus/types.py` to include full per-run rollout data:
    -   `all_runs_outputs`, `all_runs_raw_outputs`, `all_runs_explanations`, `all_runs_logprobs`.

## Test Plan
**Audio Verification**:
-   Verified `AudioArray` creation and manipulation with `tests/test_audio_array.py`.
-   Verified multimodal prompt formatting for audio inputs.

**Ensembling Verification**:
-   Verified ensembling strategies (majority vote, etc.) with `tests/test_ensembling.py`.
-   Verified `sem_filter` integration by running with `n_sample=3` and checking aggregated results vs individual runs.
-   Linting and static analysis passed (`ruff`, `mypy`).


## Work done by
Ireddi Rakshitha & Yaswanth Devavarapu
