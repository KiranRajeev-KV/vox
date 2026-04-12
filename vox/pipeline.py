"""Pipeline — thread wiring, state machine, and main loop."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time

import numpy as np
from openai import OpenAI

from vox.config import Settings
from vox.dictionary import VocabularyCorrector
from vox.history import History, SessionRecord
from vox.hotkey import HotkeyListener
from vox.indicator import Indicator
from vox.output import Outputter
from vox.processor import LLMCleaner
from vox.recorder import Recorder
from vox.sounds import SoundCues
from vox.transcriber import TranscriptionError, make_transcriber
from vox.transcriber_base import TranscriberBase, TranscriptionInfo, TranscriptionResult

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates all Vox components via a state machine.

    Runs in the main thread. Indicator and hotkey run as daemon threads.
    The state machine: IDLE → RECORDING → TRANSCRIBING → PROCESSING →
    PASTING → IDLE. LLM cleanup runs in a background daemon thread after
    paste and does not block the next recording.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._state = "IDLE"
        self._shutdown = threading.Event()

        self._control_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        self._recorder = Recorder(settings.audio, self._audio_queue)
        self._transcriber: TranscriberBase = make_transcriber(
            settings.transcription, settings.streaming
        )
        self._corrector = VocabularyCorrector(settings.dictionary)
        self._outputter = Outputter(settings.output)
        self._llm_cleaner = LLMCleaner(settings.llm)
        self._indicator = Indicator(settings.indicator)
        self._sound_cues = SoundCues(settings.sounds)
        self._hotkey = HotkeyListener(settings.hotkey, self._control_queue)
        self._history = History(settings.history) if settings.history.enabled else None

        # Streaming state
        self._streaming_enabled = settings.streaming.enabled
        self._streamer_thread: threading.Thread | None = None
        self._streamer_stop = threading.Event()
        self._streaming_result: TranscriptionResult | None = None

    def run(self) -> None:
        """Start the pipeline and enter the main loop."""
        logger.info("Starting Vox pipeline")

        self._warmup_llm()

        indicator_thread = threading.Thread(
            target=self._indicator.run,
            name="IndicatorThread",
            daemon=True,
        )
        indicator_thread.start()

        hotkey_thread = threading.Thread(
            target=self._hotkey.run,
            name="HotkeyThread",
            daemon=True,
        )
        hotkey_thread.start()

        logger.info("Pipeline running, press %s to begin", self._settings.hotkey.trigger)

        while not self._shutdown.is_set():
            try:
                command = self._control_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._handle_command(command)

        self._shutdown_threads()
        logger.info("Pipeline stopped")

    def _warmup_llm(self) -> None:
        """Pre-load the LLM model into VRAM if using a local endpoint.

        Sends a dummy request with keep_alive=0 to force the model into
        VRAM and then unload it immediately. This avoids a 1-3s cold-start
        delay on the first real LLM call after paste.
        """
        if not self._settings.llm.enabled:
            return
        if "localhost" not in self._settings.llm.base_url:
            return

        logger.info("Warming up LLM endpoint: %s", self._settings.llm.base_url)
        try:
            client = OpenAI(
                base_url=self._settings.llm.base_url,
                api_key=self._settings.llm.api_key or "none",
                max_retries=0,
            )
            client.chat.completions.create(
                model=self._settings.llm.model,
                messages=[{"role": "user", "content": "ok"}],
                timeout=30,
                extra_body={"keep_alive": 0},
            )
            logger.info("LLM warmup complete")
        except Exception:
            logger.warning(
                "LLM warmup failed (will retry on first real call): %s",
                self._settings.llm.base_url,
            )

    def _handle_command(self, command: str) -> None:
        if command == "start" and self._state == "IDLE":
            self._state = "RECORDING"
            self._indicator.show()
            self._sound_cues.play_start()
            self._recorder.start_recording()
            if self._streaming_enabled:
                self._start_streaming()

        elif command == "stop" and self._state == "RECORDING":
            self._indicator.hide()
            self._sound_cues.play_stop()
            self._recorder.stop_recording()
            if self._streaming_enabled:
                self._stop_streaming()
            self._process_audio()

        elif command == "shutdown":
            self._shutdown.set()

    def _start_streaming(self) -> None:
        """Start the streaming transcription thread."""
        self._streamer_stop.clear()
        self._streaming_result = None
        try:
            self._transcriber.start_streaming()
        except TranscriptionError:
            logger.warning("Failed to start streaming, falling back to offline mode")
            self._streaming_enabled = False
            return

        # Register audio callback on the recorder to feed streaming directly.
        # Drain the queue AFTER streaming is confirmed working, so that
        # audio recorded before streaming started doesn't block the offline path.
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._recorder.set_streaming_callback(self._transcriber.feed_chunk)

        self._streamer_thread = threading.Thread(
            target=self._streamer_loop,
            name="StreamerThread",
            daemon=True,
        )
        self._streamer_thread.start()
        logger.info(
            "Streaming started (chunk=%dms)",
            self._settings.streaming.chunk_size_ms,
        )

    def _streamer_loop(self) -> None:
        """Wait for streaming to complete.

        Audio is fed directly to the transcriber via the recorder's
        streaming callback. This thread just updates the indicator
        with partial text.
        """
        interval = self._settings.streaming.chunk_size_ms / 1000.0
        while not self._streamer_stop.is_set():
            # Get partial update and show on indicator
            update = self._transcriber.get_update()
            if update is not None and update.partial_text:
                self._indicator.show_text(update.partial_text)

            self._streamer_stop.wait(timeout=interval)

    def _stop_streaming(self) -> None:
        """Stop the streaming transcription thread and collect final result."""
        # Clear the callback so no more audio is fed to the transcriber
        self._recorder.set_streaming_callback(None)
        self._streamer_stop.set()
        if self._streamer_thread is not None:
            self._streamer_thread.join(timeout=5.0)
            self._streamer_thread = None

        try:
            self._streaming_result = self._transcriber.stop_streaming()
        except TranscriptionError:
            logger.warning("Streaming stop failed, will fall back to offline transcription")
            self._streaming_result = None

        logger.info("Streaming stopped")

    def _process_audio(self) -> None:
        """Process recorded audio through the full pipeline."""
        session_start = time.monotonic()

        # If streaming was active, use the streaming result.
        if self._streaming_result is not None:
            result: TranscriptionResult | None = self._streaming_result
            self._streaming_result = None
        else:
            # Fallback: use offline transcription.
            try:
                audio = self._audio_queue.get(timeout=30)
            except queue.Empty:
                logger.error("No audio received after stop signal")
                self._state = "IDLE"
                return

            result = self._transcribe(audio)

        if result is None:
            self._state = "IDLE"
            return

        if not result.text:
            logger.info("No speech detected, returning to IDLE")
            self._state = "IDLE"
            return

        # Apply vocabulary corrections before paste and LLM cleanup.
        corrected_text = self._corrector.correct(result.text)

        window_class = self._outputter.get_active_window_class()
        raw_len = self._paste(corrected_text, result.info, window_class)
        paste_time = time.monotonic()

        transcription_latency_ms = int((paste_time - session_start) * 1000)
        full_pipeline_latency_ms = int((time.monotonic() - session_start) * 1000)

        self._save_session(
            result.text,
            corrected_text,
            result.info,
            window_class,
            transcription_latency_ms,
            full_pipeline_latency_ms,
        )

        if self._settings.llm.enabled:
            threading.Thread(
                target=self._llm_replace,
                args=(corrected_text, raw_len, window_class, paste_time),
                name="LLMThread",
                daemon=True,
            ).start()

        self._state = "IDLE"

    def _transcribe(self, audio: np.ndarray) -> TranscriptionResult | None:
        self._state = "TRANSCRIBING"
        try:
            return self._transcriber.transcribe(audio)
        except TranscriptionError as exc:
            if "CUDA" in str(exc) or "out of memory" in str(exc).lower():
                logger.error("Whisper CUDA OOM — switch to a smaller model in config.toml")
                try:
                    subprocess.run(
                        [
                            "notify-send",
                            "Vox",
                            "CUDA OOM: switch to smaller model in config.toml",
                        ],
                        check=False,
                        timeout=5,
                    )
                except Exception:
                    pass
            else:
                logger.exception("Transcription failed")
            return None
        except Exception:
            logger.exception("Transcription failed")
            return None

    def _paste(self, text: str, info: TranscriptionInfo, window_class: str) -> int:
        self._state = "PASTING"
        raw_len = self._outputter.paste(text)
        logger.info(
            "Pasted %d chars (window=%s, lang=%s)",
            raw_len,
            window_class,
            info.language or "unknown",
        )
        return raw_len

    def _save_session(
        self,
        raw_text: str,
        processed_text: str,
        info: TranscriptionInfo,
        window_class: str,
        transcription_latency_ms: int,
        full_pipeline_latency_ms: int,
    ) -> None:
        """Save the session to the history database.

        Failures are logged but never crash the pipeline.
        """
        if self._history is None:
            return

        try:
            record = SessionRecord(
                id=None,
                created_at="",
                raw_text=raw_text,
                clean_text=processed_text,
                duration_ms=int(info.duration * 1000),
                duration_after_vad_ms=int(info.duration_after_vad * 1000),
                word_count=len(processed_text.split()),
                app_context=window_class,
                language=info.language or "unknown",
                model_used=self._settings.transcription.model,
                transcription_latency_ms=transcription_latency_ms,
                full_pipeline_latency_ms=full_pipeline_latency_ms,
            )
            self._history.save_session(record)
            logger.info(
                "Session saved: %d words, %.1fs audio, lang=%s, latency=%dms",
                record.word_count,
                info.duration,
                record.language,
                full_pipeline_latency_ms,
            )
        except Exception:
            logger.exception("Failed to save session to history")

    def _llm_replace(self, text: str, raw_len: int, window_class: str, paste_time: float) -> None:
        """Background LLM cleanup and replace."""
        result = self._llm_cleaner.clean(text)
        if result and result != text:
            logger.info("LLM cleaned text, applying replace")
            self._outputter.replace(result, raw_len, window_class, paste_time)
        else:
            logger.debug("LLM returned no changes or failed, skipping replace")

    def shutdown(self) -> None:
        """Signal the pipeline to shut down gracefully."""
        self._shutdown.set()

    def _shutdown_threads(self) -> None:
        if self._state == "RECORDING":
            self._recorder.stop_recording()
            logger.info("Stopped active recording on shutdown")

        if self._streaming_enabled and self._streamer_thread is not None:
            self._streamer_stop.set()
            self._streamer_thread.join(timeout=2.0)
            self._streamer_thread = None

        self._hotkey.stop()
        self._indicator.stop()
        if self._history is not None:
            self._history.close()
        self._transcriber.shutdown()
        logger.info("Pipeline shut down gracefully")
