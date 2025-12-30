"""
Speech Detect - A Python library for detecting speech segments and non-speech gaps in audio/video files.

This package provides utilities for:
- Streaming VAD detection using FSMN-VAD-ONNX model
- Speech segment detection
- Non-speech gap detection
- Support for all FFmpeg-supported audio/video formats
"""

import logging

from .exceptions import VadModelInitializationError, VadModelNotFoundError, VadProcessingError
from .speech_detect import SpeechDetector
from .vad_types import VadSegment

__version__ = "0.1.0"

# Configure library root logger
# Use NullHandler to ensure library remains silent when user hasn't configured logging
# If user configures logging (e.g., logging.basicConfig()), logs will bubble up to root logger for processing
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "SpeechDetector",
    "VadSegment",
    "VadModelNotFoundError",
    "VadModelInitializationError",
    "VadProcessingError",
]
