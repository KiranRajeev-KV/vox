"""Tests for sounds — WAV loading, playback, and tone generation."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from vox.config import SoundsSettings
from vox.sounds import (
    SoundCues,
    _apply_volume,
    _load_wav,
    generate_tone,
    play_sound,
)


def _make_sounds_settings(**kwargs: object) -> SoundsSettings:
    defaults: dict[str, object] = {
        "enabled": True,
        "start_sound": "assets/start.wav",
        "stop_sound": "assets/stop.wav",
        "volume": 0.7,
    }
    defaults.update(kwargs)
    return SoundsSettings(**defaults)  # type: ignore[arg-type]


def _write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = 16000,
    sample_width: int = 2,
    n_channels: int = 1,
) -> None:
    """Write a numpy array to a WAV file with specified params."""
    if sample_width == 2:
        data = (samples * 32767).astype(np.int16).tobytes()
    elif sample_width == 4:
        data = (samples * 2147483647).astype(np.int32).tobytes()
    elif sample_width == 1:
        data = ((samples + 1) * 127.5).astype(np.uint8).tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


class TestLoadWav:
    def test_load_mono_16bit(self, tmp_path: Path) -> None:
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        path = tmp_path / "mono_16.wav"
        _write_wav(path, audio, sample_width=2, n_channels=1)

        loaded, sr = _load_wav(path)
        assert sr == 16000
        assert loaded.ndim == 1
        assert len(loaded) == 5
        assert loaded.dtype == np.float32
        np.testing.assert_array_almost_equal(loaded, audio, decimal=3)

    def test_load_stereo_16bit(self, tmp_path: Path) -> None:
        left = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        right = np.array([0.0, 0.25, 0.5], dtype=np.float32)
        stereo = np.column_stack([left, right])
        path = tmp_path / "stereo_16.wav"
        _write_wav(path, stereo.flatten(), sample_width=2, n_channels=2)

        loaded, sr = _load_wav(path)
        assert loaded.ndim == 1
        expected = stereo.mean(axis=1)
        np.testing.assert_array_almost_equal(loaded, expected, decimal=3)

    def test_load_8bit(self, tmp_path: Path) -> None:
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        path = tmp_path / "mono_8.wav"
        _write_wav(path, audio, sample_width=1)

        loaded, sr = _load_wav(path)
        assert sr == 16000
        assert loaded.ndim == 1
        assert len(loaded) == 3
        assert loaded.dtype == np.float32

    def test_load_32bit(self, tmp_path: Path) -> None:
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        path = tmp_path / "mono_32.wav"
        _write_wav(path, audio, sample_width=4)

        loaded, sr = _load_wav(path)
        assert sr == 16000
        assert loaded.ndim == 1
        assert len(loaded) == 3
        assert loaded.dtype == np.float32


class TestApplyVolume:
    def test_full_volume(self) -> None:
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = _apply_volume(audio, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_half_volume(self) -> None:
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = _apply_volume(audio, 0.5)
        expected = np.array([0.25, -0.25], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_silent(self) -> None:
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = _apply_volume(audio, 0.0)
        assert np.all(result == 0.0)


class TestPlaySound:
    @patch("vox.sounds.sd.play")
    def test_plays_successfully(self, mock_play: MagicMock, tmp_path: Path) -> None:
        audio = np.array([0.5], dtype=np.float32)
        path = tmp_path / "test.wav"
        _write_wav(path, audio)

        play_sound(path, volume=0.5)
        mock_play.assert_called_once()

    @patch("vox.sounds.sd.play")
    def test_missing_file_logs_error(self, mock_play: MagicMock, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.wav"
        play_sound(missing)
        mock_play.assert_not_called()

    @patch("vox.sounds.sd.play")
    def test_invalid_wav_logs_error(self, mock_play: MagicMock, tmp_path: Path) -> None:
        bad = tmp_path / "bad.wav"
        bad.write_text("not a wav file")
        play_sound(bad)
        mock_play.assert_not_called()


class TestSoundCues:
    def test_play_start_enabled(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "cue.wav"
        _write_wav(wav_path, np.array([0.5], dtype=np.float32))

        config = _make_sounds_settings(
            enabled=True,
            start_sound=str(wav_path),
            stop_sound=str(wav_path),
        )
        cues = SoundCues(config)

        with patch("vox.sounds.sd.play") as mock_play:
            cues.play_start()
            mock_play.assert_called_once()

    def test_play_stop_enabled(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "cue.wav"
        _write_wav(wav_path, np.array([0.5], dtype=np.float32))

        config = _make_sounds_settings(
            enabled=True,
            start_sound=str(wav_path),
            stop_sound=str(wav_path),
        )
        cues = SoundCues(config)

        with patch("vox.sounds.sd.play") as mock_play:
            cues.play_stop()
            mock_play.assert_called_once()

    def test_play_start_disabled(self, tmp_path: Path) -> None:
        config = _make_sounds_settings(enabled=False)
        cues = SoundCues(config)

        with patch("vox.sounds.play_sound") as mock_play:
            cues.play_start()
            mock_play.assert_not_called()

    def test_play_stop_disabled(self, tmp_path: Path) -> None:
        config = _make_sounds_settings(enabled=False)
        cues = SoundCues(config)

        with patch("vox.sounds.play_sound") as mock_play:
            cues.play_stop()
            mock_play.assert_not_called()


class TestGenerateTone:
    def test_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        generate_tone(path)
        assert path.exists()

    def test_valid_wav_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        generate_tone(path)
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 0

    def test_correct_duration(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        duration = 0.5
        sample_rate = 16000
        generate_tone(path, duration=duration, sample_rate=sample_rate)
        with wave.open(str(path), "rb") as wf:
            expected_frames = int(sample_rate * duration)
            assert abs(wf.getnframes() - expected_frames) <= 1

    def test_correct_sample_rate(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        generate_tone(path, sample_rate=48000)
        with wave.open(str(path), "rb") as wf:
            assert wf.getframerate() == 48000

    def test_mono(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        generate_tone(path)
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1

    def test_16bit(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        generate_tone(path)
        with wave.open(str(path), "rb") as wf:
            assert wf.getsampwidth() == 2

    def test_default_params(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        generate_tone(path)
        assert path.exists()
        with wave.open(str(path), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1

    def test_custom_params(self, tmp_path: Path) -> None:
        path = tmp_path / "tone.wav"
        generate_tone(
            path,
            duration=1.0,
            freq_start=200.0,
            freq_end=2000.0,
            sample_rate=44100,
        )
        with wave.open(str(path), "rb") as wf:
            assert wf.getframerate() == 44100
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert abs(wf.getnframes() - 44100) <= 1
