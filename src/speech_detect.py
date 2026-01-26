"""
Speech Detector

Detects speech segments and non-speech gaps in audio/video files using Fsmn_vad_online model with streaming processing.
Supports all FFmpeg-compatible formats and processes large files efficiently with constant memory usage.
"""

import logging
import os

import numpy as np
from ffmpeg_audio import FFmpegAudio
from funasr_onnx import Fsmn_vad_online

from .exceptions import VadModelInitializationError, VadModelNotFoundError, VadProcessingError
from .rms_calculator import RMSCalculator
from .sd_types import RMSPoint, VadSegment
from .vad_parser import VadParser

logger = logging.getLogger(__name__)


def _get_default_rms_frame_size_ms() -> int:
    """
    Get default RMS frame size from environment variable or use default value.

    Returns:
        int: Frame size in milliseconds (default: 100ms)
    """
    env_value = os.getenv("RMS_FRAME_SIZE_MS")
    if env_value:
        try:
            value = int(env_value)
            if value > 0:
                return value
            else:
                logger.warning(f"RMS_FRAME_SIZE_MS must be > 0, got {env_value}. Using default 100ms")
        except ValueError:
            logger.warning(f"Invalid RMS_FRAME_SIZE_MS={env_value}, using default 100ms")
    return 100


def _get_default_rms_output_interval_ms() -> int:
    """
    Get default RMS output interval from environment variable or use default value.

    Returns:
        int: Output interval in milliseconds (default: 50ms)
    """
    env_value = os.getenv("RMS_OUTPUT_INTERVAL_MS")
    if env_value:
        try:
            value = int(env_value)
            if value > 0:
                return value
            else:
                logger.warning(f"RMS_OUTPUT_INTERVAL_MS must be > 0, got {env_value}. Using default 50ms")
        except ValueError:
            logger.warning(f"Invalid RMS_OUTPUT_INTERVAL_MS={env_value}, using default 50ms")
    return 50


class SpeechDetector:
    """
    Speech Detector (streaming only)

    Performs speech activity detection on streaming audio data using FSMN-VAD-ONNX model
    and derives speech segments and non-speech gaps. Designed for processing large audio/video files
    with constant memory footprint regardless of file duration.
    """

    SAMPLE_RATE = 16000  # Fixed sample rate constant (Hz)

    # Default parameter constants (can be overridden by environment variables)
    RMS_FRAME_SIZE_MS = _get_default_rms_frame_size_ms()
    RMS_OUTPUT_INTERVAL_MS = _get_default_rms_output_interval_ms()

    def __init__(self, model_dir: str = None):
        """
        Initialize speech detector.

        Args:
            model_dir: Path to the model directory. If None, reads from MODEL_FSMN_VAD_DIR
                      environment variable.

        Raises:
            VadModelNotFoundError: Model directory path is not set or does not exist.
            VadModelInitializationError: Model initialization failed.
        """
        # Determine model directory path
        if model_dir is None:
            model_dir = os.getenv("MODEL_FSMN_VAD_DIR")
            if not model_dir:
                raise VadModelNotFoundError("MODEL_FSMN_VAD_DIR environment variable not set. " "Please set it to the path of the FSMN-VAD model directory.")

        # Validate directory exists
        if not os.path.exists(model_dir):
            raise VadModelNotFoundError(f"Model directory not found: {model_dir}")

        # Initialize model (FSMN VAD model only has quantized version, always use quantize=True)
        try:
            self.model = Fsmn_vad_online(model_dir, quantize=True)
            self.model_dir = model_dir
        except Exception as e:
            raise VadModelInitializationError(
                f"Failed to initialize VAD model from {model_dir}: {str(e)}",
                model_dir=model_dir,
            ) from e

    def detect(
        self,
        file_path: str,
        chunk_duration_sec: int = None,
        start_ms: int = None,
        duration_ms: int = None,
        merge_gap_threshold_ms: int = None,
        rms_frame_size_ms: int = None,
        rms_output_interval_ms: int = None,
    ) -> tuple[list[VadSegment], list[VadSegment], list[RMSPoint]]:
        """
        Detect speech segments and non-speech gaps in audio/video file using streaming processing.

        Processes audio file in chunks using ffmpeg-audio package's stream method,
        suitable for large files. Memory usage is constant and independent of total audio duration.

        Args:
            file_path: Path to audio/video file (supports all FFmpeg-compatible formats).
            chunk_duration_sec: Chunk duration in seconds. None uses default (20 minutes).
            start_ms: Start position in milliseconds. None starts from beginning of file.
            duration_ms: Total duration to process in milliseconds. None processes to end of file.
            merge_gap_threshold_ms: Gap threshold in milliseconds. Adjacent speech segments with gaps
                                   smaller than this threshold will be merged into a single segment.
                                   None (default) disables merging. If <= 0, a warning will be logged
                                   and merging will be disabled.
            rms_frame_size_ms: Convolution window size in milliseconds for RMS calculation.
                              Default: 100ms (can be overridden by RMS_FRAME_SIZE_MS environment variable).
            rms_output_interval_ms: Output sampling interval in milliseconds for RMS curve.
                                   Default: 50ms (can be overridden by RMS_OUTPUT_INTERVAL_MS environment variable).
                                   If > rms_frame_size_ms, will be adjusted to rms_frame_size_ms.

        Returns:
            tuple[list[VadSegment], list[VadSegment], list[RMSPoint]]:
                - speech_segments: List of speech segments, format: [{"s": ms, "e": ms}, ...]
                - gaps: List of non-speech gaps, format: [{"s": ms, "e": ms}, ...]
                - rms_curve: RMS curve data, always computed and returned.
                           Format: [{"ms": int, "value": float}, ...]  # List[RMSPoint]
                Timestamps are relative to audio start (0-based), in milliseconds.

        Raises:
            VadProcessingError: Error occurred during processing.
        """
        # Initialize RMS calculator (always enabled)
        # Use default values if not provided (can be overridden by environment variables)
        if rms_frame_size_ms is None:
            rms_frame_size_ms = self.RMS_FRAME_SIZE_MS
        if rms_output_interval_ms is None:
            rms_output_interval_ms = self.RMS_OUTPUT_INTERVAL_MS

        # RMSCalculator will automatically validate and correct invalid parameters with warnings
        rms_calculator = RMSCalculator(
            frame_size_ms=rms_frame_size_ms,
            output_interval_ms=rms_output_interval_ms,
        )

        # Validate merge_gap_threshold_ms parameter
        if merge_gap_threshold_ms is not None and merge_gap_threshold_ms < 0:
            logger.warning(f"merge_gap_threshold_ms must be >= 0, got {merge_gap_threshold_ms}. Merging will be disabled.")
            merge_gap_threshold_ms = None

        parser = VadParser()
        param_dict = {"in_cache": []}
        speech_segments = []
        total_samples = 0

        # Process each chunk in streaming fashion
        chunk_count = 0
        try:
            for chunk in FFmpegAudio.stream(
                file_path,
                chunk_duration_sec=chunk_duration_sec,
                start_ms=start_ms,
                duration_ms=duration_ms,
            ):
                chunk_count += 1
                chunk_samples = len(chunk)

                # Validate chunk format
                if chunk.dtype != np.float32:
                    raise VadProcessingError(
                        f"Chunk dtype must be float32, got {chunk.dtype}",
                        file_path=file_path,
                        details={"chunk_index": chunk_count, "dtype": str(chunk.dtype)},
                    )

                # Accumulate total sample count
                total_samples += chunk_samples

                # Calculate RMS curve (always computed)
                rms_calculator.process_chunk(chunk)

                # param_dict state is automatically maintained across chunks
                result = self.model(audio_in=chunk, param_dict=param_dict)

                # Parse model output
                segments = parser.parse(result)
                speech_segments.extend(segments)

        except VadProcessingError:
            # Re-raise VadProcessingError
            raise
        except Exception as e:
            raise VadProcessingError(
                f"Stream processing failed: {str(e)}",
                file_path=file_path,
                details={"chunk_index": chunk_count, "exception_type": type(e).__name__},
            ) from e

        # Final flush to ensure all data is processed
        try:
            param_dict["is_final"] = True
            final_result = self.model(audio_in=[], param_dict=param_dict)
            final_segments = parser.parse(final_result)
            speech_segments.extend(final_segments)

            # Handle any unclosed segments
            parser.flush()
        except Exception as e:
            raise VadProcessingError(
                f"Final flush failed: {str(e)}",
                file_path=file_path,
                details={"exception_type": type(e).__name__},
            ) from e

        # Merge adjacent segments if threshold is specified and valid
        if merge_gap_threshold_ms is not None:
            if merge_gap_threshold_ms <= 0:
                logger.warning(f"merge_gap_threshold_ms must be > 0 to enable merging, got {merge_gap_threshold_ms}. Merging will be disabled.")
            else:
                speech_segments = self._merge_adjacent_segments(speech_segments, merge_gap_threshold_ms)

        # Derive non-speech gaps from speech segments
        gaps = self._derive_non_speech_gaps(speech_segments, total_samples)

        # Get RMS curve data (always computed)
        rms_curve = rms_calculator.finalize()

        return speech_segments, gaps, rms_curve

    @staticmethod
    def _derive_non_speech_gaps(speech_segments: list[VadSegment], audio_length_samples: int) -> list[VadSegment]:
        """
        Derive non-speech gaps from speech segments.

        Computes gaps between speech segments and at the beginning/end of audio.
        Gaps represent periods of silence or non-speech audio.

        Args:
            speech_segments: List of speech segments, format: [{"start": ms, "end": ms}, ...]
            audio_length_samples: Total number of audio samples.

        Returns:
            list[VadSegment]: List of non-speech gaps, format: [{"s": ms, "e": ms}, ...]
        """
        # Calculate total audio duration in milliseconds
        duration_ms = int(audio_length_samples / SpeechDetector.SAMPLE_RATE * 1000)

        # If no speech segments, entire audio is non-speech
        if not speech_segments:
            return [{"s": 0, "e": duration_ms}]

        gaps = []

        # Check for gap at the beginning (before first speech segment)
        first_speech = speech_segments[0]
        if first_speech["s"] > 0:
            gaps.append({"s": 0, "e": first_speech["s"]})

        # Check for gaps between speech segments
        for i in range(len(speech_segments) - 1):
            prev_end = speech_segments[i]["e"]
            next_start = speech_segments[i + 1]["s"]
            if next_start > prev_end:
                gaps.append({"s": prev_end, "e": next_start})

        # Check for gap at the end (after last speech segment)
        last_speech = speech_segments[-1]
        if last_speech["e"] < duration_ms:
            gaps.append({"s": last_speech["e"], "e": duration_ms})

        return gaps

    @staticmethod
    def _merge_adjacent_segments(speech_segments: list[VadSegment], threshold_ms: int) -> list[VadSegment]:
        """
        Merge adjacent speech segments if the gap between them is smaller than threshold.

        This is useful for handling brief pauses in speech (e.g., breathing, thinking pauses)
        that should be considered part of continuous speech rather than separate segments.

        Args:
            speech_segments: List of speech segments, format: [{"s": ms, "e": ms}, ...]
                           Must be sorted by start time (which is guaranteed by streaming processing).
            threshold_ms: Gap threshold in milliseconds. Segments with gaps smaller than this
                         will be merged. Must be >= 0.

        Returns:
            list[VadSegment]: Merged speech segments, format: [{"s": ms, "e": ms}, ...]
        """
        if not speech_segments or len(speech_segments) <= 1:
            return speech_segments

        if threshold_ms < 0:
            raise ValueError(f"threshold_ms must be >= 0, got {threshold_ms}")

        merged = []
        current_segment = speech_segments[0].copy()

        for next_segment in speech_segments[1:]:
            gap = next_segment["s"] - current_segment["e"]

            if gap <= threshold_ms:
                # Merge: extend current segment to include next segment
                current_segment["e"] = next_segment["e"]
            else:
                # Gap is too large, save current segment and start new one
                merged.append(current_segment)
                current_segment = next_segment.copy()

        # Add the last segment
        merged.append(current_segment)

        return merged
