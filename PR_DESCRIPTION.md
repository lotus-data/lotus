# Pull Request: Test-Time Scaling and Audio Data Support

## Summary

This PR adds two highly requested features to LOTUS:

1. **Test-Time Scaling with Ensembling** (Closes #200)
2. **Audio Data Support via AudioArray** (Closes #196)

## Changes

### Feature 1: Test-Time Scaling (`lotus/sem_ops/ensembling.py`)

Adds ensemble-based test-time scaling strategies for improving semantic operator accuracy:

- **`EnsembleStrategy`** enum with four strategies:
  - `MAJORITY_VOTE` - Returns most common prediction
  - `WEIGHTED_AVERAGE` - Weighs predictions by confidence
  - `CONSENSUS` - Returns result only if unanimous
  - `CONFIDENCE_THRESHOLD` - Majority vote with confidence tracking

- **`EnsembleConfig`** dataclass for configuration:
  - `n_samples` - Number of samples to generate
  - `strategy` - Which ensembling strategy to use
  - `temperature` - Sampling temperature
  - `confidence_threshold` - Minimum confidence for threshold strategy

- **`Ensemble`** class for aggregating predictions

### Feature 2: Audio Data Support (`lotus/dtype_extensions/audio.py`)

Extends LOTUS to support audio data processing:

- **`AudioDtype`** - Custom pandas ExtensionDtype for audio
- **`AudioArray`** - ExtensionArray for storing audio data
- Supports 7 audio formats: `.wav`, `.mp3`, `.mp4`, `.m4a`, `.flac`, `.ogg`, `.webm`
- Includes caching, base64 encoding, and MIME type detection

### Tests

- `tests/test_ensembling.py` - 40+ test cases for all strategies
- `tests/test_audio_array.py` - Comprehensive tests for AudioArray

## Usage Examples

### Test-Time Scaling
```python
from lotus.sem_ops.ensembling import Ensemble, EnsembleConfig, EnsembleStrategy

config = EnsembleConfig(n_samples=5, strategy=EnsembleStrategy.MAJORITY_VOTE)
ensemble = Ensemble(config)
result = ensemble.aggregate([True, True, False, True, False])  # Returns True
```

### Audio Data
```python
from lotus.dtype_extensions import AudioArray
import pandas as pd

audio_files = ['speech.wav', 'music.mp3', 'podcast.flac']
df = pd.DataFrame({'audio': AudioArray(audio_files)})
# Now can use with semantic operators
```

## Checklist

- [x] Code follows project style guidelines
- [x] Comprehensive tests included
- [x] Documentation updated (docstrings)
- [x] All tests pass locally

## Contributors

- @iredd
- @yaswanth
