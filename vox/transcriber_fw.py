"""Transcription via faster-whisper (CTranslate2)."""

from __future__ import annotations

import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from vox.config import TranscriptionSettings
from vox.transcriber_base import TranscriberBase, TranscriptionError, TranscriptionResult

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(TranscriberBase):
    """Wrapper around faster-whisper WhisperModel.

    Loads the model on init. Fails fast if the model cannot load.
    Accepts numpy audio arrays, returns transcribed text and metadata.
    """

    def __init__(self, config: TranscriptionSettings) -> None:
        self._config = config
        logger.info(
            "Loading Whisper model: %s (device=%s, compute=%s)",
            config.model,
            config.device,
            config.compute_type,
        )
        try:
            self._model = WhisperModel(
                config.model,
                device=config.device,
                compute_type=config.compute_type,
            )
        except RuntimeError as exc:
            if "CUDA" in str(exc) or "out of memory" in str(exc).lower():
                msg = (
                    f"Whisper model '{config.model}' failed to load (CUDA OOM). "
                    f"Switch to a smaller model in config.toml, e.g. "
                    f'model = "small" or model = "medium". '
                    f"Original error: {exc}"
                )
                logger.error(msg)
                raise TranscriptionError(msg) from exc
            raise

        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe audio and return (text, metadata).

        Args:
            audio: 1D float32 numpy array at 16kHz.

        Returns:
            TranscriptionResult with text and TranscriptionInfo.
            If VAD finds no speech, returns empty text.

        Raises:
            TranscriptionError: If transcription fails.
        """
        start = time.perf_counter()

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._config.language,
                vad_filter=self._config.vad,
                beam_size=5,
            )
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        text_parts: list[str] = []
        for segment in segments:
            text_parts.append(segment.text)

        duration = time.perf_counter() - start
        full_text = " ".join(text_parts).strip()

        logger.info(
            "Transcribed %.2fs audio in %.2fs (lang=%s, prob=%.2f)",
            info.duration,
            duration,
            info.language,
            info.language_probability,
        )

        if not full_text:
            logger.info("VAD found no speech, returning empty text")

        return TranscriptionResult(full_text, info)

    def shutdown(self) -> None:
        """Release the Whisper model and free GPU memory."""
        del self._model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
