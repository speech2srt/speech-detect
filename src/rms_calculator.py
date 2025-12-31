"""
RMS Calculator

Computes RMS curve using streaming convolution for efficient processing.
"""

import logging

import numpy as np
from scipy import signal

from .sd_types import RMSPoint

logger = logging.getLogger(__name__)


class RMSCalculator:
    """
    Stream-based RMS calculator using convolution.

    Processes audio data in chunks and computes RMS values with sliding window
    convolution, then samples at fixed intervals (e.g., 100ms or 50ms).
    """

    # 采样率常量（固定为 16000 Hz）
    SAMPLE_RATE = 16000

    # 默认参数常量
    DEFAULT_FRAME_SIZE_MS = 100
    DEFAULT_OUTPUT_INTERVAL_MS = 100

    def __init__(
        self,
        frame_size_ms: int = DEFAULT_FRAME_SIZE_MS,
        output_interval_ms: int = DEFAULT_OUTPUT_INTERVAL_MS,
    ):
        """
        Initialize RMS calculator.

        Args:
            frame_size_ms: Convolution window size in milliseconds. Invalid values (<= 0) will use default (100ms).
            output_interval_ms: Output sampling interval in milliseconds. Invalid values (<= 0) will use default (100ms).
                               Can be 50ms or 100ms. If > frame_size_ms, will be adjusted to frame_size_ms.
        """
        # 验证并修正 frame_size_ms
        if frame_size_ms <= 0:
            logger.warning(f"frame_size_ms must be > 0, got {frame_size_ms}. Using default: {self.DEFAULT_FRAME_SIZE_MS}ms")
            frame_size_ms = self.DEFAULT_FRAME_SIZE_MS

        # 验证并修正 output_interval_ms
        if output_interval_ms <= 0:
            logger.warning(f"output_interval_ms must be > 0, got {output_interval_ms}. Using default: {self.DEFAULT_OUTPUT_INTERVAL_MS}ms")
            output_interval_ms = self.DEFAULT_OUTPUT_INTERVAL_MS

        # 确保 output_interval_ms <= frame_size_ms
        if output_interval_ms > frame_size_ms:
            logger.warning(
                f"output_interval_ms ({output_interval_ms}) must be <= frame_size_ms ({frame_size_ms}). " f"Adjusting output_interval_ms to {frame_size_ms}ms"
            )
            output_interval_ms = frame_size_ms

        self.frame_size_ms = frame_size_ms
        self.output_interval_ms = output_interval_ms

        # 计算窗口大小（采样点数）
        self.window_size = int(frame_size_ms * self.SAMPLE_RATE / 1000)
        self.output_interval_samples = int(output_interval_ms * self.SAMPLE_RATE / 1000)

        # 归一化的矩形窗口（用于卷积）
        self.window = np.ones(self.window_size, dtype=np.float32) / self.window_size

        # Buffer 用于累积数据和处理边界
        # 需要保留至少 window_size // 2 个点用于边界处理
        self.buffer = np.array([], dtype=np.float32)

        # RMS 曲线结果（List[RMSPoint]）
        self.rms_curve: list[RMSPoint] = []

        # 当前处理位置（毫秒）
        self.current_time_ms = 0

        # 上次输出的时间（用于去重）
        self.last_output_time_ms = -output_interval_ms

    def process_chunk(self, chunk: np.ndarray):
        """
        Process a chunk of audio data.

        Args:
            chunk: Audio data chunk (float32, -1.0 ~ 1.0)

        Raises:
            ValueError: If chunk dtype is not float32.
        """
        if chunk.dtype != np.float32:
            raise ValueError(f"chunk dtype must be float32, got {chunk.dtype}")

        # 1. 累积到 buffer
        self.buffer = np.concatenate([self.buffer, chunk])

        # 2. 处理 buffer 中的数据
        # 需要至少一个窗口大小才能计算卷积
        min_buffer_size = self.window_size

        # 每次处理较大的块（例如 1 秒），提高效率
        process_chunk_size = self.SAMPLE_RATE  # 1 秒

        while len(self.buffer) >= min_buffer_size:
            # 确定本次处理的数据量
            process_size = min(len(self.buffer), process_chunk_size)

            # 如果剩余数据不足一个窗口大小，停止处理
            if process_size < min_buffer_size:
                break

            # 提取要处理的数据
            process_data = self.buffer[:process_size]

            # 3. 计算平方
            sq_signal = process_data**2

            # 4. 使用 scipy 卷积计算滑动窗口平均值
            # mode='same' 保证输出长度与输入一致
            # 注意：边界处会有失真，第一个点（索引 0）对应 -window_size//2 到 window_size//2 的窗口
            mean_sq = signal.convolve(sq_signal, self.window, mode="same", method="auto")

            # 5. 开根号得到 RMS（加 epsilon 防止 log0 错误）
            rms_curve = np.sqrt(np.maximum(mean_sq, 1e-10))

            # 6. 按固定间隔采样保存（始终从 i=0 开始，确保跨 chunk 连续）
            for i in range(0, len(rms_curve), self.output_interval_samples):
                time_ms = int(self.current_time_ms + (i * 1000 / self.SAMPLE_RATE))

                # 只保存未输出过的点（避免重复）
                if time_ms > self.last_output_time_ms:
                    # 使用 RMSPoint 类型定义（TypedDict）
                    # 限制 RMS 值精度为 6 位小数，减小文件大小
                    self.rms_curve.append({"ms": time_ms, "value": round(float(rms_curve[i]), 6)})
                    self.last_output_time_ms = time_ms

            # 7. 更新 buffer 和当前时间
            # 保留最后 window_size // 2 个点，用于边界处理
            keep_size = self.window_size // 2
            self.buffer = self.buffer[process_size - keep_size :]
            self.current_time_ms += (process_size - keep_size) * 1000 / self.SAMPLE_RATE

    def finalize(self) -> list[RMSPoint]:
        """
        Finalize RMS calculation and return result.

        Discards any remaining data that is less than a full window size.

        Returns:
            list[RMSPoint]: RMS curve data, list of RMSPoint objects.
                          Format: [{"ms": int, "value": float}, ...]
        """
        # 丢弃不足一个窗口大小的剩余数据（可选日志）
        if len(self.buffer) > 0:
            pass

        # 在返回前统一丢弃首个点（如果存在），避免卷积边界失真
        if self.rms_curve:
            self.rms_curve = self.rms_curve[1:]

        return self.rms_curve

    def reset(self):
        """Reset calculator state for reuse."""
        self.buffer = np.array([], dtype=np.float32)
        self.rms_curve = []
        self.current_time_ms = 0
        self.last_output_time_ms = -self.output_interval_ms
