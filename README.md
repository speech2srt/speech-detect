# speech-detect

A Python library for detecting speech segments, non-speech gaps, and RMS energy curves in audio/video files using FSMN-VAD-ONNX with streaming processing.

## Features

### Core Functionality

- **Speech segment detection**: Detect all speech segments in audio/video files with precise timestamps
- **Non-speech gap derivation**: Automatically compute non-speech gaps (silence periods) from detected speech segments
- **RMS energy detection**: Compute RMS (Root Mean Square) energy curve for audio analysis and visualization

### Advanced Features

- **Adjacent segment merging**: Merge adjacent speech segments with gaps smaller than a threshold (useful for handling brief pauses like breathing or thinking pauses)

### Technical Capabilities

- **Streaming processing**: Process large audio/video files in chunks without loading everything into memory
- **Memory efficient**: Constant memory usage regardless of audio file duration
- **Format support**: Supports all audio/video formats that FFmpeg supports (MP3, WAV, FLAC, Opus, MP4, AVI, etc.)
- **Time range support**: Support start time and duration parameters for partial processing

## Installation

```bash
pip install speech-detect
```

**Note**: This package requires:

- FFmpeg to be installed on your system and available in PATH
- FSMN-VAD-ONNX model files (see Model Setup below)

## Model Setup

This package requires FSMN-VAD-ONNX model files. The model is available on Hugging Face:

**Model Repository**: [funasr/fsmn-vad-onnx](https://huggingface.co/funasr/fsmn-vad-onnx)

### Download the Model

1. Install Git LFS (required for downloading large model files):

   ```bash
   git lfs install
   ```

2. Clone the model repository:

   ```bash
   git clone https://huggingface.co/funasr/fsmn-vad-onnx
   ```

   This will download the model files including `model_quant.onnx`, `config.yaml`, `am.mvn`, etc.

3. Set the `MODEL_FSMN_VAD_DIR` environment variable to point to the model directory:
   ```bash
   export MODEL_FSMN_VAD_DIR=/path/to/fsmn-vad-onnx
   ```

Alternatively, you can specify the model directory when initializing `SpeechDetector`:

```python
from speech_detect import SpeechDetector

detector = SpeechDetector(model_dir="/path/to/fsmn-vad-onnx")
```

## Quick Start

### Detect Speech Segments, Gaps, and RMS Energy Curve

```python
from speech_detect import SpeechDetector

# Initialize detector (reads MODEL_FSMN_VAD_DIR from environment)
detector = SpeechDetector()

# Detect speech segments, non-speech gaps, and RMS energy curve in an audio file
speech_segments, gaps, rms_curve = detector.detect("audio.mp3")

# speech_segments is a list of dictionaries: [{"start": 0, "end": 500}, ...]
for segment in speech_segments:
    start_ms = segment["start"]
    end_ms = segment["end"]
    duration = end_ms - start_ms
    print(f"Speech segment: {start_ms}ms - {end_ms}ms (duration: {duration}ms)")

# gaps is a list of dictionaries: [{"start": 0, "end": 500}, ...]
for gap in gaps:
    start_ms = gap["start"]
    end_ms = gap["end"]
    duration = end_ms - start_ms
    print(f"Non-speech gap: {start_ms}ms - {end_ms}ms (duration: {duration}ms)")

# rms_curve is a list of dictionaries: [{"ms": 0, "value": 0.123}, ...]
for point in rms_curve:
    time_ms = point["ms"]
    rms_value = point["value"]
    print(f"RMS at {time_ms}ms: {rms_value}")
```

### Processing Specific Time Range

```python
# Process only the first 30 seconds
speech_segments, gaps, rms_curve = detector.detect(
    file_path="audio.mp3",
    start_ms=0,
    duration_ms=30000,
)

# Process from 10 seconds, duration 5 seconds
speech_segments, gaps, rms_curve = detector.detect(
    file_path="audio.mp3",
    start_ms=10000,
    duration_ms=5000,
)
```

### Custom Chunk Size

```python
# Use 1-minute chunks instead of default 20-minute chunks
speech_segments, gaps, rms_curve = detector.detect(
    file_path="audio.mp3",
    chunk_duration_sec=60,
)
```

### Merging Adjacent Segments

```python
# Merge adjacent segments with gaps smaller than 300ms
# Useful for handling brief pauses in speech (breathing, thinking pauses)
speech_segments, gaps, rms_curve = detector.detect(
    file_path="audio.mp3",
    merge_gap_threshold_ms=300,
)
```

### Custom RMS Energy Detection Parameters

```python
# Customize RMS calculation window size and output interval
# frame_size_ms: Convolution window size (default: 100ms)
# output_interval_ms: Output sampling interval (default: 50ms)
speech_segments, gaps, rms_curve = detector.detect(
    file_path="audio.mp3",
    rms_frame_size_ms=100,      # 100ms window for RMS calculation
    rms_output_interval_ms=50,   # Output every 50ms for higher resolution
)

# The RMS curve can be used for audio visualization, energy analysis, or as input for other audio processing tasks
```

### Environment Variables for RMS Configuration

You can configure default RMS parameters using environment variables:

```bash
# Set default frame size (default: 100ms)
export RMS_FRAME_SIZE_MS=100

# Set default output interval (default: 50ms)
export RMS_OUTPUT_INTERVAL_MS=50
```

These environment variables will be used as defaults when `rms_frame_size_ms` and `rms_output_interval_ms` parameters are not explicitly provided to the `detect()` method.

## RMS Energy Design Principles

`speech-detect` always emits an RMS energy curve together with VAD results. The feature is designed to stay lightweight while providing meaningful downstream insight:

- **Always-on streaming measurement**: RMS is computed chunk by chunk using a sliding, normalized window, so memory usage stays constant even for multi-hour recordings.
- **Decoupled smoothing vs. resolution**: `rms_frame_size_ms` controls the convolution window (how aggressively noise is smoothed), while `rms_output_interval_ms` controls the sampling density. Allowing the interval to be smaller than the window lets you plot dense, smooth curves without losing smoothing benefits.
- **Deterministic timeline**: Both parameters are expressed in milliseconds and map directly to timestamps in the returned `rms_curve`, making it trivial to align energy data with speech segments, captions, or UI waveforms.
- **Downstream flexibility**: The curve can drive silence gating, highlight low-energy pauses, or simply power visual meters. Because it is always returned, callers can adopt it opportunistically without extra processing steps.

## API Reference

### SpeechDetector

Main class for speech detection. All methods are instance methods.

#### `SpeechDetector.__init__(model_dir=None)`

Initialize speech detector.

**Parameters:**

- `model_dir` (str, optional): Path to the FSMN-VAD model directory. If None, reads from `MODEL_FSMN_VAD_DIR` environment variable.

**Note:** The FSMN-VAD model only has a quantized version, so `quantize=True` is always used internally.

**Raises:**

- `VadModelNotFoundError`: If model directory is not found or not set
- `VadModelInitializationError`: If model initialization fails

#### `SpeechDetector.detect(file_path, chunk_duration_sec=None, start_ms=None, duration_ms=None, merge_gap_threshold_ms=None, rms_frame_size_ms=None, rms_output_interval_ms=None)`

Detect speech segments, non-speech gaps, and RMS energy curve in audio/video file using streaming processing.

**Parameters:**

- `file_path` (str): Path to the audio/video file (supports all FFmpeg formats)
- `chunk_duration_sec` (int, optional): Duration of each chunk in seconds. Defaults to 1200 (20 minutes). Must be > 0 if provided.
- `start_ms` (int, optional): Start position in milliseconds. None means from file beginning. If None but `duration_ms` is provided, defaults to 0.
- `duration_ms` (int, optional): Total duration to process in milliseconds. None means process until end. If specified, processing stops when this duration is reached.
- `merge_gap_threshold_ms` (int, optional): Gap threshold in milliseconds. Adjacent speech segments with gaps smaller than this threshold will be merged into a single segment. None (default) disables merging. If <= 0, a warning will be logged and merging will be disabled. Useful for handling brief pauses in speech (e.g., breathing, thinking pauses) that should be considered part of continuous speech.
- `rms_frame_size_ms` (int, optional): Convolution window size in milliseconds for RMS calculation. Defaults to 100ms (can be overridden by `RMS_FRAME_SIZE_MS` environment variable). If <= 0, a warning will be logged and default value (100ms) will be used.
- `rms_output_interval_ms` (int, optional): Output sampling interval in milliseconds for RMS curve. Defaults to 50ms (can be overridden by `RMS_OUTPUT_INTERVAL_MS` environment variable). If <= 0, a warning will be logged and default value (50ms) will be used. If > `rms_frame_size_ms`, it will be adjusted to `rms_frame_size_ms`.

**Returns:**

- `tuple[list[VadSegment], list[VadSegment], list[RMSPoint]]`: Tuple of (speech_segments, gaps, rms_curve)
  - `speech_segments`: List of speech segments, format: `[{"start": ms, "end": ms}, ...]`
    - Timestamps are relative to audio start (from 0)
    - Unit: milliseconds
  - `gaps`: List of non-speech gaps, format: `[{"start": ms, "end": ms}, ...]`
    - Timestamps are relative to audio start (from 0)
    - Unit: milliseconds
  - `rms_curve`: RMS energy curve data, always computed and returned, format: `[{"ms": int, "value": float}, ...]`
    - `ms`: Time position in milliseconds (relative to audio start, from 0)
    - `value`: RMS energy value (float, typically in range [0.0, 1.0])
    - Unit: milliseconds for time, dimensionless for value

**Raises:**

- `VadProcessingError`: If processing fails

## Data Types

### VadSegment

A TypedDict representing a time segment (can be a speech segment or a non-speech gap).

**Fields:**

- `start` (int): Segment start time in milliseconds
- `end` (int): Segment end time in milliseconds

**Example:**

```python
segment: VadSegment = {"start": 100, "end": 500}
```

### RMSPoint

A TypedDict representing a point on the RMS energy curve.

**Fields:**

- `ms` (int): Time position in milliseconds (relative to audio start, from 0)
- `value` (float): RMS energy value (typically in range [0.0, 1.0])

**Example:**

```python
point: RMSPoint = {"ms": 100, "value": 0.123}
```

## Exceptions

### `VadModelNotFoundError`

Raised when VAD model directory is not found or not set.

**Attributes:**

- `message`: Human-readable error message

### `VadModelInitializationError`

Raised when VAD model initialization fails.

**Attributes:**

- `message`: Primary error message
- `model_dir`: Path to the model directory that caused the error

### `VadProcessingError`

Raised when VAD processing fails.

**Attributes:**

- `message`: Primary error message
- `file_path`: Path to the file being processed
- `details`: Additional error details dictionary

## Requirements

- Python >= 3.10
- FFmpeg (must be installed separately)
- numpy >= 1.26.4
- scipy >= 1.11.0 (for RMS energy calculation)
- funasr-onnx >= 0.4.1
- ffmpeg-audio >= 0.2.0
- jieba >= 0.42.1
- torch >= 2.9.1
- setuptools == 80.8.0 (to avoid UserWarning from jieba about deprecated pkg_resources API)

## License

MIT License
