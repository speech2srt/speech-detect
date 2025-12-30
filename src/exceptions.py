"""
Exception classes for VAD detection errors.

Provides a hierarchy of exceptions for different error conditions,
enabling precise error handling and debugging.
"""


class VadModelNotFoundError(Exception):
    """
    Raised when VAD model directory is not found or not set.

    This exception indicates that the model directory path is either not provided,
    not set in environment variable MODEL_FSMN_VAD_DIR, or the directory does not exist.
    """

    def __init__(self, message: str):
        """
        Initialize exception.

        Args:
            message: Human-readable error message describing the issue.
        """
        super().__init__(message)
        self.message = message


class VadModelInitializationError(Exception):
    """
    Raised when VAD model initialization fails.

    This exception indicates that the Fsmn_vad_online model failed to initialize
    from the provided model directory, possibly due to missing files or corrupted model.
    """

    def __init__(self, message: str, model_dir: str = None):
        """
        Initialize exception.

        Args:
            message: Primary error message (required).
            model_dir: Path to the model directory that caused the error (optional).
        """
        super().__init__(message)
        self.message = message
        self.model_dir = model_dir


class VadProcessingError(Exception):
    """
    Raised when VAD processing fails.

    This exception indicates that an error occurred during the VAD detection process,
    such as audio format issues, streaming errors, or model inference failures.
    """

    def __init__(self, message: str, file_path: str = None, details: dict = None):
        """
        Initialize exception.

        Args:
            message: Primary error message (required).
            file_path: Path to the file being processed (optional).
            details: Additional error details dictionary (optional).
        """
        super().__init__(message)
        self.message = message
        self.file_path = file_path
        self.details = details or {}
