"""Audio capture from microphone using sounddevice."""

from __future__ import annotations

import logging
import queue
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from vox.config import AudioSettings

logger = logging.getLogger(__name__)


def validate_device(device_id: int | None) -> None:
    """Raise RuntimeError if the configured device is not available."""
    devices = sd.query_devices()
    if device_id is not None:
        if device_id >= len(devices):
            raise RuntimeError(
                f"Audio device index {device_id} not found. "
                f"Available devices:\n{sd.query_devices()}"
            )
        dev_info = devices[device_id]
        if dev_info["max_input_channels"] < 1:
            raise RuntimeError(
                f"Audio device {device_id} ({dev_info['name']}) has no input channels. "
                f"Available devices:\n{sd.query_devices()}"
            )
    else:
        default = sd.query_devices(kind="input")
        if default is None:
            raise RuntimeError(
                f"No default input device found. Available devices:\n{sd.query_devices()}"
            )


class Recorder:
    """Capture audio from the microphone and push to audio_queue on stop.

    The pipeline calls start_recording() and stop_recording() directly.
    Audio chunks are collected via the sounddevice callback and
    concatenated into a single numpy array on stop.
    """

    def __init__(
        self,
        config: AudioSettings,
        audio_queue: queue.Queue[np.ndarray],
    ) -> None:
        self._config = config
        self._audio_queue = audio_queue
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._streaming_callback: Callable[[np.ndarray], None] | None = None
        validate_device(config.device)

    def set_streaming_callback(self, callback: Callable[[np.ndarray], None] | None) -> None:
        """Set or clear the streaming audio callback.

        The callback is invoked from the sounddevice real-time callback
        thread with each captured audio chunk (already flattened to 1D).

        Args:
            callback: Function to call with each audio chunk, or None to clear.
        """
        self._streaming_callback = callback

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict[str, float],
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("sounddevice callback status: %s", status)
        chunk = indata.copy()  # Single copy, reused for both paths
        self._chunks.append(chunk)
        # Also push to streaming callback if registered.
        if self._streaming_callback is not None:
            audio_1d = chunk if chunk.ndim == 1 else chunk.flatten()
            self._streaming_callback(audio_1d)

    def start_recording(self) -> None:
        """Start capturing audio from the microphone."""
        if self._recording:
            logger.warning("Recorder already recording, ignoring start signal")
            return

        self._chunks.clear()
        try:
            self._stream = sd.InputStream(
                device=self._config.device,
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype=np.float32,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            logger.exception("Failed to start audio recording")
            self._stream = None
            return

        self._recording = True
        logger.info(
            "Recording started (device=%s, sr=%d, ch=%d)",
            self._config.device or "default",
            self._config.sample_rate,
            self._config.channels,
        )

    def stop_recording(self) -> None:
        """Stop capturing and put the audio array into audio_queue."""
        if not self._recording:
            logger.warning("Recorder not recording, ignoring stop signal")
            return

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error stopping audio stream")
            self._stream = None

        self._recording = False

        if not self._chunks:
            logger.info("No audio captured, skipping")
            return

        audio = np.concatenate(self._chunks, axis=0)

        if audio.ndim > 1:
            if self._config.channels > 1:
                audio = audio.mean(axis=1)
            else:
                audio = audio.flatten()

        duration = len(audio) / self._config.sample_rate
        logger.info("Recording stopped, captured %.2fs of audio", duration)
        self._audio_queue.put(audio)
