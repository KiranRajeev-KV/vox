"""Tests for recorder — sounddevice audio capture."""

from __future__ import annotations

import logging
import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.config import AudioSettings
from vox.recorder import Recorder, validate_device


def _make_audio_settings(**kwargs: object) -> AudioSettings:
    defaults: dict[str, object] = {
        "device": None,
        "sample_rate": 16000,
        "channels": 1,
    }
    defaults.update(kwargs)
    return AudioSettings(**defaults)  # type: ignore[arg-type]


def _make_recorder(
    audio_queue: queue.Queue[np.ndarray] | None = None,
    **kwargs: object,
) -> tuple[Recorder, queue.Queue[np.ndarray]]:
    q = audio_queue or queue.Queue()
    config = _make_audio_settings(**kwargs)
    with patch("vox.recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = [
            {
                "name": "Default Device",
                "max_input_channels": 2,
                "max_output_channels": 2,
            }
        ]
        mock_sd.query_devices.side_effect = lambda **kw: (
            {"name": "Default", "max_input_channels": 1}
            if kw.get("kind") == "input"
            else [
                {
                    "name": "Default Device",
                    "max_input_channels": 2,
                    "max_output_channels": 2,
                }
            ]
        )
        recorder = Recorder(config, q)
    return recorder, q


class TestValidateDevice:
    @patch("vox.recorder.sd")
    def test_valid_device_index(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}
        ]
        validate_device(0)

    @patch("vox.recorder.sd")
    def test_invalid_device_index(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}
        ]
        with pytest.raises(RuntimeError, match="not found"):
            validate_device(5)

    @patch("vox.recorder.sd")
    def test_output_only_device(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = [
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2}
        ]
        with pytest.raises(RuntimeError, match="no input channels"):
            validate_device(0)

    @patch("vox.recorder.sd")
    def test_none_uses_default(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}
        ]
        mock_sd.query_devices.side_effect = lambda **kwargs: (
            {"name": "Default", "max_input_channels": 1}
            if kwargs.get("kind") == "input"
            else [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
        )
        validate_device(None)

    @patch("vox.recorder.sd")
    def test_no_default_device(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = []
        mock_sd.query_devices.side_effect = lambda **kwargs: (
            None if kwargs.get("kind") == "input" else []
        )
        with pytest.raises(RuntimeError, match="No default input device"):
            validate_device(None)


class TestRecorderInit:
    @patch("vox.recorder.sd")
    def test_validates_device_on_init(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}
        ]
        mock_sd.query_devices.side_effect = lambda **kwargs: (
            {"name": "Default", "max_input_channels": 1}
            if kwargs.get("kind") == "input"
            else [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
        )
        q: queue.Queue[np.ndarray] = queue.Queue()
        Recorder(_make_audio_settings(), q)
        assert mock_sd.query_devices.called

    @patch("vox.recorder.sd")
    def test_initial_state(self, mock_sd: MagicMock) -> None:
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}
        ]
        mock_sd.query_devices.side_effect = lambda **kwargs: (
            {"name": "Default", "max_input_channels": 1}
            if kwargs.get("kind") == "input"
            else [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
        )
        q: queue.Queue[np.ndarray] = queue.Queue()
        recorder = Recorder(_make_audio_settings(), q)
        assert recorder._recording is False
        assert recorder._chunks == []


class TestStartRecording:
    def test_starts_stream(self) -> None:
        recorder, q = _make_recorder()
        mock_stream = MagicMock()
        with patch("vox.recorder.sd.InputStream", return_value=mock_stream) as mock_ctor:
            recorder.start_recording()
            mock_ctor.assert_called_once()
            mock_stream.start.assert_called_once()

    def test_sets_recording_true(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        assert recorder._recording is True

    def test_clears_chunks(self) -> None:
        recorder, q = _make_recorder()
        recorder._chunks = [np.array([1.0, 2.0])]
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        assert recorder._chunks == []

    def test_double_start_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
            recorder.start_recording()
        assert "already recording" in caplog.text.lower()


class TestStopRecording:
    def test_stops_and_closes_stream(self) -> None:
        recorder, q = _make_recorder()
        mock_stream = MagicMock()
        with patch("vox.recorder.sd.InputStream", return_value=mock_stream):
            recorder.start_recording()
            recorder.stop_recording()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_sets_recording_false(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
            recorder.stop_recording()
        assert recorder._recording is False

    def test_pushes_audio_to_queue(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        recorder._chunks = [np.array([[0.5], [0.3]], dtype=np.float32)]
        recorder.stop_recording()
        audio = q.get_nowait()
        assert audio.ndim == 1
        assert len(audio) == 2

    def test_audio_is_float32(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        recorder._chunks = [np.array([[0.5]], dtype=np.float32)]
        recorder.stop_recording()
        audio = q.get_nowait()
        assert audio.dtype == np.float32

    def test_audio_is_1d(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        recorder._chunks = [np.array([[0.5], [0.3]], dtype=np.float32)]
        recorder.stop_recording()
        audio = q.get_nowait()
        assert audio.ndim == 1

    def test_stop_when_not_recording(self, caplog: pytest.LogCaptureFixture) -> None:
        recorder, q = _make_recorder()
        recorder.stop_recording()
        assert "not recording" in caplog.text.lower()

    def test_stop_with_no_chunks(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
            recorder.stop_recording()
        assert q.empty()


class TestAudioProcessing:
    def test_single_chunk_audio(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        chunk = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        recorder._chunks = [chunk]
        recorder.stop_recording()
        audio = q.get_nowait()
        np.testing.assert_array_almost_equal(audio, [0.1, 0.2, 0.3])

    def test_multiple_chunks_concatenated(self) -> None:
        recorder, q = _make_recorder()
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        recorder._chunks = [
            np.array([[0.1], [0.2]], dtype=np.float32),
            np.array([[0.3], [0.4]], dtype=np.float32),
        ]
        recorder.stop_recording()
        audio = q.get_nowait()
        np.testing.assert_array_almost_equal(audio, [0.1, 0.2, 0.3, 0.4])

    def test_multichannel_downmix(self) -> None:
        recorder, q = _make_recorder(channels=2)
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        stereo = np.array([[0.4, 0.6], [0.2, 0.8]], dtype=np.float32)
        recorder._chunks = [stereo]
        recorder.stop_recording()
        audio = q.get_nowait()
        expected = stereo.mean(axis=1)
        np.testing.assert_array_almost_equal(audio, expected)

    def test_audio_duration_calculation(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        recorder, q = _make_recorder(sample_rate=16000)
        with patch("vox.recorder.sd.InputStream", return_value=MagicMock()):
            recorder.start_recording()
        recorder._chunks = [np.zeros((16000, 1), dtype=np.float32)]
        recorder.stop_recording()
        assert "1.00" in caplog.text
