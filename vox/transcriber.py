"""Transcriber factory — picks backend based on streaming config."""

from __future__ import annotations

from vox.config import StreamingSettings, TranscriptionSettings
from vox.transcriber_base import TranscriberBase, TranscriptionError, TranscriptionResult


def make_transcriber(
    transcription: TranscriptionSettings,
    streaming: StreamingSettings,
) -> TranscriberBase:
    """Create a transcriber backend based on the streaming configuration.

    When streaming is enabled, Vox uses CarelessWhisper (causal streaming).
    When streaming is disabled, Vox uses faster-whisper (CTranslate2, offline).

    Args:
        transcription: Core transcription settings (model, device, etc.).
        streaming: Streaming settings (enabled, chunk_size_ms, etc.).

    Returns:
        A TranscriberBase implementation appropriate for the config.
    """
    if streaming.enabled:
        from vox.transcriber_cw import CarelessWhisperTranscriber

        return CarelessWhisperTranscriber(transcription, streaming)
    else:
        from vox.transcriber_fw import FasterWhisperTranscriber

        return FasterWhisperTranscriber(transcription)


__all__ = [
    "TranscriptionError",
    "TranscriptionResult",
    "TranscriberBase",
    "make_transcriber",
]
