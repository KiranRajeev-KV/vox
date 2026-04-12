"""Transcription via CarelessWhisper (causal streaming)."""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue

import numpy as np
from faster_whisper.transcribe import TranscriptionInfo

from vox.config import StreamingSettings, TranscriptionSettings
from vox.transcriber_base import (
    StreamingUpdate,
    TranscriberBase,
    TranscriptionError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# Whisper constants
SAMPLE_RATE = 16000


class CarelessWhisperTranscriber(TranscriberBase):
    """Wraps CarelessWhisper for both streaming and offline modes.

    Offline mode (streaming disabled in config):
        transcribe(audio) → single call, runs non_causal_transcribe.
        Accuracy matches base Whisper (LoRA toggled off during inference).

    Streaming mode (streaming enabled in config):
        start_streaming()  → initializes decoder, clears buffers, starts worker
        feed_chunk(audio)  → appends audio to buffer (called from audio callback)
        get_update()       → returns latest StreamingUpdate (non-blocking)
        stop_streaming()   → stops worker, returns final TranscriptionResult
    """

    @property
    def supports_streaming(self) -> bool:
        return True

    def __init__(
        self,
        transcription_config: TranscriptionSettings,
        streaming_config: StreamingSettings,
    ) -> None:
        self._transcription_config = transcription_config
        self._streaming_config = streaming_config
        self._model: object | None = None
        self._streaming_active = False
        self._stream_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._update_queue: Queue[StreamingUpdate | None] = Queue(maxsize=1)
        self._last_text: str = ""
        self._total_duration: float = 0.0
        self._language: str | None = None
        self._chunk_samples: int = int(streaming_config.chunk_size_ms * SAMPLE_RATE / 1000)
        self._session_audio: list[np.ndarray] = []

        # Worker thread for GPU decode (keeps audio callback real-time)
        self._worker_thread: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._work_available = threading.Event()

        # Load model eagerly at startup so the first hotkey press is fast.
        self._load_model()

    def _load_model(self) -> None:
        """Load the CarelessWhisper model from HuggingFace."""
        if self._model is not None:
            return

        import importlib.util

        try:
            spec = importlib.util.find_spec("whisper_rt")
        except ValueError:
            # Module is present but malformed (e.g. mocked in tests with
            # __spec__ not set). Assume it exists — let the real import
            # attempt below surface the actual error.
            spec = object()  # type: ignore[assignment]

        if spec is None:
            # Not on any import path — try the submodule
            import sys
            from pathlib import Path

            submodule_path = (
                Path(__file__).resolve().parent.parent / "third_party" / "WhisperRT-Streaming"
            )
            if not submodule_path.is_dir():
                msg = (
                    "whisper_rt module not found. Install it via: "
                    "git submodule add https://github.com/tomer9080/WhisperRT-Streaming "
                    "third_party/WhisperRT-Streaming"
                )
                logger.error(msg)
                raise TranscriptionError(msg) from None

            sys.path.insert(0, str(submodule_path))
            spec = importlib.util.find_spec("whisper_rt")
            if spec is None:
                msg = (
                    "whisper_rt module not found in third_party/WhisperRT-Streaming. "
                    "Ensure the git submodule is initialized: "
                    "git submodule update --init"
                )
                logger.error(msg)
                raise TranscriptionError(msg) from None

        # Now try the actual import — if it fails, a transitive dependency is missing
        try:
            import whisper_rt  # type: ignore[import-not-found,no-redef]
        except (ModuleNotFoundError, ImportError) as exc:
            msg = (
                f"whisper_rt failed to load (missing dependency). "
                f"Run: uv sync --extra streaming. "
                f"Original error: {exc}"
            )
            logger.error(msg)
            raise TranscriptionError(msg) from exc

        # Monkey-patch upstream bug: StreamingDecoder.last_logits is not initialized,
        # causing TypeError: 'NoneType' object is not subscriptable on the first decode.
        # Upstream bug: _check_last_tokens() is called before last_logits is ever set.
        if not getattr(self._model, "_vox_patch_applied", False):
            try:
                from whisper_rt.streaming_decoding import (  # type: ignore[import-not-found]
                    StreamingDecoder,
                )

                # Patch 1: Initialize last_logits to avoid AttributeError
                _orig_init = StreamingDecoder.__init__

                def _patched_init(decoder: object, *args: object, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
                    _orig_init(decoder, *args, **kwargs)  # type: ignore[operator]
                    decoder.last_logits = None  # type: ignore[attr-defined]

                StreamingDecoder.__init__ = _patched_init  # type: ignore[method-assign]

                # Patch 2: Guard against None in _check_last_tokens
                _orig_check = StreamingDecoder._check_last_tokens  # type: ignore[attr-defined]

                def _patched_check(  # type: ignore[no-untyped-def]
                    decoder: object,
                    logits: object,
                    tokens: object,
                    next_tokens: object,
                    check_tokens: object,
                ) -> tuple:  # type: ignore[type-arg]
                    if not check_tokens:
                        return [], tokens
                    if getattr(decoder, "last_logits", None) is None:
                        # No previous logits yet — skip stability check
                        return [], tokens
                    return _orig_check(decoder, logits, tokens, next_tokens, check_tokens)  # type: ignore[operator]

                StreamingDecoder._check_last_tokens = _patched_check  # type: ignore[method-assign]
                self._model._vox_patch_applied = True  # type: ignore[attr-defined]
            except Exception:
                logger.exception(
                    "Failed to apply StreamingDecoder monkey patch — "
                    "streaming will likely crash on first decode"
                )

        # Streaming requires its own model name. No fallback to [transcription].model
        # since CarelessWhisper and faster-whisper use different checkpoints.
        model_name = self._streaming_config.model
        if not model_name:
            raise TranscriptionError(
                "streaming.model is not set in config.toml. "
                "CarelessWhisper requires an explicit model name. "
                "Supported: 'base', 'small', 'large-v2'."
            )

        logger.info(
            "Loading CarelessWhisper model: %s (chunk=%dms, beam=%d)",
            model_name,
            self._streaming_config.chunk_size_ms,
            self._streaming_config.beam_size,
        )

        try:
            self._model = whisper_rt.load_streaming_model(
                name=model_name,
                gran=self._streaming_config.chunk_size_ms,
                multilingual=False,
                device=self._transcription_config.device,
            )
            self._model.eval()  # type: ignore[union-attr]
        except Exception as exc:
            msg = (
                f"CarelessWhisper model '{model_name}' failed to load. "
                f"Ensure you are logged in to HuggingFace (hf auth login), "
                f"the model exists, and there is enough VRAM. "
                f"Original error: {exc}"
            )
            logger.error(msg)
            raise TranscriptionError(msg) from exc

        logger.info("CarelessWhisper model loaded successfully")

    def _make_info(self, text: str, duration: float, language: str | None) -> TranscriptionInfo:
        """Build a TranscriptionInfo-like object from streaming results.

        Args:
            text: The transcribed text.
            duration: Audio duration in seconds.
            language: Detected language code.

        Returns:
            A TranscriptionInfo-compatible object for pipeline compatibility.
        """
        from unittest.mock import MagicMock

        info: object = MagicMock()
        info.duration = duration
        info.duration_after_vad = duration
        info.language = language
        info.language_probability = 1.0 if language else 0.0
        return info  # type: ignore[return-value]

    # ── Offline mode ──────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe audio in offline mode.

        Args:
            audio: 1D float32 numpy array at 16kHz.

        Returns:
            TranscriptionResult with text and TranscriptionInfo.

        Raises:
            TranscriptionError: If transcription fails.
        """
        self._load_model()

        with self._stream_lock:
            if self._streaming_active:
                raise TranscriptionError(
                    "transcribe() called while streaming is active. Call stop_streaming() first."
                )

        return self._transcribe_offline(audio)

    def _transcribe_offline(self, audio: np.ndarray) -> TranscriptionResult:
        """Run CarelessWhisper in non-causal offline mode (LoRA off)."""
        start = time.perf_counter()

        try:
            assert self._model is not None
            self._model._cancel_streaming_mode()  # type: ignore[union-attr]
            result = self._model.non_causal_transcribe(audio)  # type: ignore[union-attr]
            self._model._revert_streaming_mode()  # type: ignore[union-attr]
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Offline transcription failed: {exc}") from exc

        # non_causal_transcribe returns (text, info) or segments
        if isinstance(result, tuple) and len(result) == 2:  # type: ignore[arg-type]
            text, info = result
        else:
            # Assume it returns an iterable of segments
            text_parts: list[str] = []
            for segment in result:  # type: ignore[union-attr]
                text_parts.append(segment.text)  # type: ignore[union-attr]
            text = " ".join(text_parts).strip()
            info = self._make_info(text, len(audio) / SAMPLE_RATE, None)

        duration = time.perf_counter() - start
        logger.info(
            "Transcribed (offline CW) %.2fs audio in %.2fs (lang=%s)",
            info.duration if hasattr(info, "duration") else len(audio) / SAMPLE_RATE,  # type: ignore[union-attr]
            duration,
            info.language if hasattr(info, "language") else "unknown",  # type: ignore[union-attr]
        )

        return TranscriptionResult(text, info)  # type: ignore[arg-type]

    # ── Streaming mode ────────────────────────────────────────────────

    def start_streaming(self) -> None:
        """Begin a streaming transcription session."""
        with self._stream_lock:
            if self._streaming_active:
                logger.warning("Streaming already active, ignoring")
                return
            self._load_model()
            with self._audio_lock:
                self._session_audio.clear()
            self._last_text = ""
            self._total_duration = 0.0
            self._language = None
            self._streaming_active = True
            # Drain any stale update
            try:
                self._update_queue.get_nowait()
            except Empty:
                pass
            # Reset the model's streaming state
            assert self._model is not None
            self._model.reset(use_stream=True)  # type: ignore[union-attr]
            logger.info(
                "CarelessWhisper streaming session started (chunk=%dms)",
                self._streaming_config.chunk_size_ms,
            )

        # Start worker thread for GPU decode (separate from audio callback)
        self._worker_stop.clear()
        self._work_available.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="CWWorkerThread",
            daemon=True,
        )
        self._worker_thread.start()

    def feed_chunk(self, audio_chunk: np.ndarray) -> None:
        """Feed an audio chunk into the streaming pipeline.

        This is called from the sounddevice callback thread.
        Audio is buffered; a separate worker thread performs GPU decode.

        Args:
            audio_chunk: 1D float32 numpy array at 16kHz.
        """
        if not self._streaming_active:
            return

        with self._audio_lock:
            self._session_audio.append(audio_chunk)
        # Signal worker thread that work is available
        self._work_available.set()

    def _worker_loop(self) -> None:
        """Worker thread that performs GPU decode on buffered audio.

        Runs on a dedicated thread, not the sounddevice callback thread,
        so that real-time audio capture is never blocked by GPU inference.
        """
        while not self._worker_stop.is_set():
            # Wait for work or stop signal
            self._work_available.wait(timeout=0.05)
            self._work_available.clear()

            if self._worker_stop.is_set():
                break

            if not self._streaming_active:
                continue

            # Process accumulated audio
            self._process_session_audio()

    def _process_session_audio(self) -> None:
        """Process accumulated audio if enough is available.

        Must be called from the worker thread, not the audio callback.
        """
        assert self._model is not None

        while True:
            with self._audio_lock:
                available = sum(len(a) for a in self._session_audio)
                if available < self._chunk_samples:
                    break  # Not enough audio yet

                # Take one chunk's worth
                chunk_audio = np.concatenate(self._session_audio, axis=0)
                samples_needed = self._chunk_samples
                if len(chunk_audio) < samples_needed:
                    break

                audio_to_process = chunk_audio[:samples_needed]
                remaining = chunk_audio[samples_needed:]

                self._session_audio.clear()
                if len(remaining) > 0:
                    self._session_audio.append(remaining)

            # Transcribe this chunk (outside the lock — GPU decode is slow)
            try:
                text = self._decode_chunk(audio_to_process)
            except Exception:
                logger.exception("Streaming decode failed")
                # Discard the failed chunk's remaining audio and continue
                continue

            self._total_duration += self._streaming_config.chunk_size_ms / 1000
            self._last_text = text

            update = StreamingUpdate(
                partial_text=text,
                stable_text=text,
                is_final=False,
                language=self._language,
                duration=self._total_duration,
            )

            # Non-blocking put — drop old update if queue is full
            try:
                self._update_queue.put_nowait(update)
            except Full:
                pass  # Consumer hasn't read previous update; drop is expected
            except Exception:
                logger.exception("Failed to queue streaming update")

    def _decode_chunk(self, audio: np.ndarray) -> str:
        """Decode a single audio chunk through the CarelessWhisper model.

        Args:
            audio: Audio segment of exactly chunk_size_ms duration.

        Returns:
            Transcribed text for this chunk.
        """
        import torch  # type: ignore[import-not-found]

        assert self._model is not None

        # Convert to mel spectrogram
        audio_tensor = torch.from_numpy(audio).float().to(self._transcription_config.device)

        # Build decoding options
        from vox.transcriber_cw_decode import DecodingOptions

        options = DecodingOptions(
            language=self._transcription_config.language or "en",
            beam_size=(
                self._streaming_config.beam_size if self._streaming_config.beam_size > 0 else None
            ),
            without_timestamps=True,
            fp16=False,
            gran=self._streaming_config.chunk_size_ms // 20,
            use_ca_kv_cache=True,
            stream_decode=True,
        )

        # Calculate mel for this chunk
        mel = self._model.spec_streamer.calc_mel_with_new_frame(audio_tensor)  # type: ignore[union-attr]

        # Decode
        result = self._model.decode(mel.squeeze(0), options)  # type: ignore[union-attr]

        # Extract language on first chunk
        if self._language is None and hasattr(result, "language"):  # type: ignore[arg-type]
            self._language = result.language  # type: ignore[union-attr]

        return result.text if hasattr(result, "text") else ""  # type: ignore[union-attr, arg-type]

    def get_update(self) -> StreamingUpdate | None:
        """Get the latest streaming update without blocking.

        Returns:
            StreamingUpdate if available, None otherwise.
        """
        try:
            return self._update_queue.get_nowait()
        except Empty:
            return None

    def stop_streaming(self) -> TranscriptionResult:
        """End the streaming session and return the final result.

        Returns the text accumulated during streaming. Audio is
        processed incrementally by the worker thread during recording —
        this method just stops the worker and returns what we have.
        """
        import time as _time

        t0 = _time.monotonic()
        with self._stream_lock:
            if not self._streaming_active:
                raise TranscriptionError("Streaming not active")
            self._streaming_active = False

        # Stop worker thread
        self._worker_stop.set()
        self._work_available.set()  # wake up worker so it can exit
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        elapsed_ms = (_time.monotonic() - t0) * 1000
        logger.debug("stop_streaming: worker thread joined (%.1fms)", elapsed_ms)

        # Get the last update if available
        final_update = self.get_update()
        logger.debug("stop_streaming: got update (%.1fms)", (_time.monotonic() - t0) * 1000)
        final_text = final_update.partial_text if final_update else self._last_text
        final_duration = final_update.duration if final_update else self._total_duration
        final_language = final_update.language if final_update else self._language

        # Emit final update (for any late consumers)
        final = StreamingUpdate(
            partial_text=final_text,
            stable_text=final_text,
            is_final=True,
            language=final_language,
            duration=final_duration,
        )
        try:
            self._update_queue.put_nowait(final)
        except Exception:
            pass

        logger.info(
            "CarelessWhisper streaming session ended (%.2fs audio, lang=%s, total_time=%.1fms)",
            final_duration,
            final_language or "unknown",
            (_time.monotonic() - t0) * 1000,
        )

        info = self._make_info(final_text, final_duration, final_language)
        return TranscriptionResult(final_text, info)  # type: ignore[arg-type]

    def shutdown(self) -> None:
        """Release the CarelessWhisper model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
