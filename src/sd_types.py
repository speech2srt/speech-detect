"""
Type definitions for Speech Detect package.

Defines TypedDict structures for VAD segments and energy curve data used throughout the package.
"""

from typing import TypedDict


class VadSegment(TypedDict):
    """
    VAD segment type.

    Represents a time segment (can be speech segment or non-speech gap).
    Timestamps are relative to the start of the audio stream.

    This type is used for:
    - Speech segments: Detected speech time periods
    - Non-speech gaps: Non-speech time periods

    Attributes:
        s: Segment start time in milliseconds (integer).
        e: Segment end time in milliseconds (integer).
    """

    s: int
    e: int


class RMSPoint(TypedDict):
    """
    RMS point type.

    Represents a point on the RMS energy curve.

    Attributes:
        ms: Time position in milliseconds (integer).
        value: RMS energy value (float).
    """

    ms: int
    value: float
