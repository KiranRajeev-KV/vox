"""Abstract transcriber interface and shared types."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from faster_whisper.transcribe import TranscriptionInfo


class TranscriptionError(Exception):
    """Raised when transcription fails unrecoverably."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Normalized result from any transcriber backend."""

    text: str
    info: TranscriptionInfo


@dataclass(frozen=True)
class StreamingUpdate:
    """Emitted by a streaming transcriber on each chunk decode."""

    partial_text: str  # all text so far (may change with more audio)
    stable_text: str  # confirmed/committed text (won't change)
    is_final: bool  # True when streaming session is complete
    language: str | None
    duration: float  # seconds of audio processed


class TranscriberBase(abc.ABC):
    """Abstract base for transcription backends.

    Every backend must implement transcribe() for offline mode.
    Streaming backends additionally implement start_streaming(),
    feed_chunk(), and stop_streaming().
    """

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe audio and return text + metadata.

        Args:
            audio: 1D float32 numpy array at 16kHz.

        Returns:
            TranscriptionResult with text and TranscriptionInfo.

        Raises:
            TranscriptionError: If transcription fails.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release model resources (GPU memory, etc.)."""

    # --- Optional streaming methods (non-streaming backends raise NotImplementedError) ---

    def start_streaming(self) -> None:
        """Begin a streaming transcription session.

        Call feed_chunk() repeatedly, then stop_streaming() to finalize.
        """
        raise NotImplementedError(f"Streaming not supported by {type(self).__name__}")

    def feed_chunk(self, audio_chunk: np.ndarray) -> None:
        """Feed an audio chunk into the streaming pipeline.

        Args:
            audio_chunk: 1D float32 numpy array at 16kHz.
        """
        raise NotImplementedError(f"Streaming not supported by {type(self).__name__}")

    def get_update(self) -> StreamingUpdate | None:
        """Get the latest streaming update without blocking.

        Returns:
            StreamingUpdate if available, None otherwise.
        """
        raise NotImplementedError(f"Streaming not supported by {type(self).__name__}")

    def stop_streaming(self) -> TranscriptionResult:
        """End the streaming session and return the final result.

        Returns:
            TranscriptionResult with the complete transcribed text.

        Raises:
            TranscriptionError: If streaming was not active.
        """
        raise NotImplementedError(f"Streaming not supported by {type(self).__name__}")

    @property
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming mode."""
        return False
