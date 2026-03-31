"""Audio cues — play start/stop sounds."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from vox.config import SoundsSettings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file, return (audio_data as float32, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    elif sample_width == 1:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= np.iinfo(dtype).max

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sample_rate


def _apply_volume(audio: np.ndarray, volume: float) -> np.ndarray:
    """Scale audio by volume (0.0=silent, 1.0=full)."""
    return audio * volume


def play_sound(path: Path, volume: float = 0.7) -> None:
    """Load and play a WAV file. Non-blocking — playback continues
    in the background so recording starts immediately.

    Errors are logged but never raised — a missing sound file
    should not break the recording flow.
    """
    try:
        audio, sample_rate = _load_wav(path)
        audio = _apply_volume(audio, volume)
        sd.play(audio, sample_rate)
    except Exception:
        logger.exception("Failed to play sound: %s", path)


class SoundCues:
    """Convenience wrapper around start/stop sound playback."""

    def __init__(self, config: SoundsSettings) -> None:
        self._config = config
        self._start_path = _PROJECT_ROOT / config.start_sound
        self._stop_path = _PROJECT_ROOT / config.stop_sound

    def play_start(self) -> None:
        """Play the start-of-recording cue."""
        if self._config.enabled:
            play_sound(self._start_path, self._config.volume)

    def play_stop(self) -> None:
        """Play the end-of-recording cue."""
        if self._config.enabled:
            play_sound(self._stop_path, self._config.volume)


def generate_tone(
    path: Path,
    duration: float = 0.2,
    freq_start: float = 440.0,
    freq_end: float = 880.0,
    sample_rate: int = 16000,
) -> None:
    """Generate a WAV file with a frequency-swept tone.

    Args:
        path: Output WAV file path.
        duration: Length in seconds.
        freq_start: Starting frequency in Hz.
        freq_end: Ending frequency in Hz.
        sample_rate: Sample rate in Hz.
    """
    n_samples = int(sample_rate * duration)

    # Linear frequency sweep
    freq = np.linspace(freq_start, freq_end, n_samples)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    signal = np.sin(phase)

    # Apply fade in/out to avoid clicks
    fade_len = int(sample_rate * 0.02)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_out.shape[0] :] *= fade_out

    # Convert to 16-bit PCM
    signal_int16 = (signal * 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal_int16.tobytes())

    logger.info("Generated tone: %s (%.0f→%.0f Hz, %.2fs)", path, freq_start, freq_end, duration)
