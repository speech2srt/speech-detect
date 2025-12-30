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
from .vad_parser import VadParser
from .vad_types import VadSegment

logger = logging.getLogger(__name__)


class SpeechDetector:
    """
    Speech Detector (streaming only)

    Performs speech activity detection on streaming audio data using FSMN-VAD-ONNX model
    and derives speech segments and non-speech gaps. Designed for processing large audio/video files
    with constant memory footprint regardless of file duration.
    """

    SAMPLE_RATE = 16000  # Fixed sample rate constant (Hz)

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
    ) -> tuple["list[VadSegment]", "list[VadSegment]"]:
        """
        Detect speech segments and non-speech gaps in audio/video file using streaming processing.

        Processes audio file in chunks using ffmpeg-audio package's stream method,
        suitable for large files. Memory usage is constant and independent of total audio duration.

        Args:
            file_path: Path to audio/video file (supports all FFmpeg-compatible formats).
            chunk_duration_sec: Chunk duration in seconds. None uses default (20 minutes).
            start_ms: Start position in milliseconds. None starts from beginning of file.
            duration_ms: Total duration to process in milliseconds. None processes to end of file.

        Returns:
            tuple[list[VadSegment], list[VadSegment]]: Tuple of (speech_segments, gaps).
                                                     - speech_segments: List of speech segments, format: [{"start": ms, "end": ms}, ...]
                                                     - gaps: List of non-speech gaps, format: [{"start": ms, "end": ms}, ...]
                                                     Timestamps are relative to audio start (0-based), in milliseconds.

        Raises:
            VadProcessingError: Error occurred during processing.
        """
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

        # Derive non-speech gaps from speech segments
        gaps = self._derive_non_speech_gaps(speech_segments, total_samples)

        return speech_segments, gaps

    @staticmethod
    def _derive_non_speech_gaps(speech_segments: "list[VadSegment]", audio_length_samples: int) -> "list[VadSegment]":
        """
        Derive non-speech gaps from speech segments.

        Computes gaps between speech segments and at the beginning/end of audio.
        Gaps represent periods of silence or non-speech audio.

        Args:
            speech_segments: List of speech segments, format: [{"start": ms, "end": ms}, ...]
            audio_length_samples: Total number of audio samples.

        Returns:
            list[VadSegment]: List of non-speech gaps, format: [{"start": ms, "end": ms}, ...]
        """
        # Calculate total audio duration in milliseconds
        duration_ms = int(audio_length_samples / SpeechDetector.SAMPLE_RATE * 1000)

        # If no speech segments, entire audio is non-speech
        if not speech_segments:
            return [{"start": 0, "end": duration_ms}]

        gaps = []

        # Check for gap at the beginning (before first speech segment)
        first_speech = speech_segments[0]
        if first_speech["start"] > 0:
            gaps.append({"start": 0, "end": first_speech["start"]})

        # Check for gaps between speech segments
        for i in range(len(speech_segments) - 1):
            prev_end = speech_segments[i]["end"]
            next_start = speech_segments[i + 1]["start"]
            if next_start > prev_end:
                gaps.append({"start": prev_end, "end": next_start})

        # Check for gap at the end (after last speech segment)
        last_speech = speech_segments[-1]
        if last_speech["end"] < duration_ms:
            gaps.append({"start": last_speech["end"], "end": duration_ms})

        return gaps
