"""
VAD State Machine Parser

Parses fragmented output from Fsmn_vad_online model and converts it into
complete {s, e} segment semantics. Handles partial segments that span
multiple chunks in streaming processing.
"""

import logging

from .sd_types import VadSegment

logger = logging.getLogger(__name__)


class VadParser:
    """
    State machine parser for Fsmn_vad_online fragmented output.

    Converts fragmented VAD model output into complete {s, e} segment semantics.
    Maintains state across chunks to handle segments that span multiple processing chunks.
    """

    def __init__(self):
        """Initialize VAD parser."""
        self.current_start_ms = -1

    def _process_segment_pair(self, beg_ms: int, end_ms: int, completed_segments: list[VadSegment]) -> None:
        """
        Process a single [start, end] pair, supporting three cases:
        1. [[beg, -1]] -> Speech start detected but not ended (partial segment)
        2. [[-1, end]] -> Speech end detected (completes previous start)
        3. [[beg, end]] -> Complete segment within single chunk

        Args:
            beg_ms: Start time in milliseconds (int or float).
            end_ms: End time in milliseconds (int or float).
            completed_segments: List to store completed segments.
        """
        # Convert to integers
        if not isinstance(beg_ms, (int, float)):
            beg_ms = -1
        else:
            beg_ms = int(beg_ms)

        if not isinstance(end_ms, (int, float)):
            end_ms = -1
        else:
            end_ms = int(end_ms)

        # Case 1: Speech start detected but not ended (partial segment)
        if beg_ms != -1 and end_ms == -1:
            # Fsmn_vad_online returns global timestamps, use directly
            self.current_start_ms = beg_ms
            # Don't add to completed_segments yet, wait for end

        # Case 2: Speech end detected (completes previous start)
        elif beg_ms == -1 and end_ms != -1:
            if self.current_start_ms != -1:
                completed_segments.append(
                    {
                        "s": self.current_start_ms,
                        "e": end_ms,  # Fsmn_vad_online returns global timestamps
                    }
                )
                self.current_start_ms = -1  # Reset state
            else:
                # Edge case: end without start (possibly from previous chunk), log warning
                logger.warning(f"[VAD Parser] Found end without start: end_ms={end_ms}")

        # Case 3: Complete segment within single chunk
        elif beg_ms != -1 and end_ms != -1:
            completed_segments.append(
                {
                    "s": beg_ms,  # Fsmn_vad_online returns global timestamps
                    "e": end_ms,
                }
            )
            # If there was an unclosed start, reset it (new complete segment encountered)
            self.current_start_ms = -1

    def parse(self, vad_output: list) -> list[VadSegment]:
        """
        Parse VAD model inference output and return completed segments.

        Handles nested list structures and partial segments. Maintains state
        across calls to handle segments spanning multiple chunks.

        Args:
            vad_output: VAD model output, format: [[beg, end], ...] or [[beg, -1]], [[-1, end]]
                       Note: Fsmn_vad_online returns global timestamps (relative to stream start).

        Returns:
            list[VadSegment]: List of completed speech segments, format: [{"s": ms, "e": ms}, ...]
        """
        if not vad_output:
            return []

        # Unwrap all nested layers
        # Format: [[[beg, end], ...]] or [[beg, end], ...] -> [[beg, end], ...]
        while isinstance(vad_output, list) and len(vad_output) == 1:
            vad_output = vad_output[0]
            if not isinstance(vad_output, list):
                break

        completed_segments = []

        # Process normalized output format: [[beg, end], ...]
        for seg in vad_output:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                beg_ms, end_ms = seg[0], seg[1]
                # Convert to integers (handle possible floats)
                beg_ms = int(beg_ms) if isinstance(beg_ms, (int, float)) else -1
                end_ms = int(end_ms) if isinstance(end_ms, (int, float)) else -1
                # Process segment pair (supports [beg, -1], [-1, end], [beg, end])
                self._process_segment_pair(beg_ms, end_ms, completed_segments)
            else:
                # Skip invalid segment formats
                logger.warning(f"[VAD Parser] Invalid segment format: {seg}")

        return completed_segments

    def flush(self) -> list[VadSegment]:
        """
        Flush parser state at end of stream.

        Handles any unclosed segments. If a start was detected but not ended,
        it's discarded (typically treated as silence at end of stream).

        Returns:
            list[VadSegment]: List of remaining completed segments (usually empty).
        """
        # If there's an unclosed start, it typically means audio was cut off mid-speech
        # Use simple discard logic (treat as silence at end)
        if self.current_start_ms != -1:
            logger.warning(f"[VAD Parser] Unclosed segment at end of stream: start_ms={self.current_start_ms}")
            self.current_start_ms = -1
        return []
