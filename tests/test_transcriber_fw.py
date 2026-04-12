"""Tests for transcriber_fw — faster-whisper wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.config import TranscriptionSettings
from vox.transcriber_base import TranscriptionError, TranscriptionResult
from vox.transcriber_fw import FasterWhisperTranscriber


def _make_transcription_settings(**kwargs: object) -> TranscriptionSettings:
    defaults: dict[str, object] = {
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "int8",
        "language": None,
        "vad": True,
    }
    defaults.update(kwargs)
    return TranscriptionSettings(**defaults)  # type: ignore[arg-type]


def _make_mock_info(**kwargs: object) -> MagicMock:
    info = MagicMock()
    info.duration = 3.0
    info.duration_after_vad = 2.5
    info.language = "en"
    info.language_probability = 0.95
    for key, value in kwargs.items():
        setattr(info, key, value)
    return info


def _make_mock_segments(texts: list[str]) -> list[MagicMock]:
    segments = []
    for text in texts:
        seg = MagicMock()
        seg.text = text
        segments.append(seg)
    return segments


class TestFasterWhisperTranscriberInit:
    @patch("vox.transcriber_fw.WhisperModel")
    def test_loads_model(self, mock_model_cls: MagicMock) -> None:
        config = _make_transcription_settings(model="large-v3", device="cuda", compute_type="int8")
        FasterWhisperTranscriber(config)
        mock_model_cls.assert_called_once_with("large-v3", device="cuda", compute_type="int8")

    @patch("vox.transcriber_fw.WhisperModel")
    def test_cuda_oom_raises_transcription_error(self, mock_model_cls: MagicMock) -> None:
        mock_model_cls.side_effect = RuntimeError("CUDA out of memory")
        config = _make_transcription_settings()
        with pytest.raises(TranscriptionError, match="CUDA OOM"):
            FasterWhisperTranscriber(config)

    @patch("vox.transcriber_fw.WhisperModel")
    def test_cuda_error_raises_transcription_error(self, mock_model_cls: MagicMock) -> None:
        mock_model_cls.side_effect = RuntimeError("CUDA error")
        config = _make_transcription_settings()
        with pytest.raises(TranscriptionError):
            FasterWhisperTranscriber(config)

    @patch("vox.transcriber_fw.WhisperModel")
    def test_other_runtime_error_reraises(self, mock_model_cls: MagicMock) -> None:
        mock_model_cls.side_effect = RuntimeError("Some other error")
        config = _make_transcription_settings()
        with pytest.raises(RuntimeError, match="Some other error"):
            FasterWhisperTranscriber(config)


class TestFasterWhisperTranscribe:
    @patch("vox.transcriber_fw.WhisperModel")
    def test_transcribes_audio(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.return_value = (
            iter(_make_mock_segments(["hello world"])),
            _make_mock_info(),
        )
        config = _make_transcription_settings()
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        transcriber.transcribe(audio)
        mock_model.transcribe.assert_called_once_with(
            audio,
            language=None,
            vad_filter=True,
            beam_size=5,
        )

    @patch("vox.transcriber_fw.WhisperModel")
    def test_returns_result_object(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        info = _make_mock_info()
        mock_model.transcribe.return_value = (
            iter(_make_mock_segments(["hello world"])),
            info,
        )
        config = _make_transcription_settings()
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber.transcribe(audio)
        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.info is info

    @patch("vox.transcriber_fw.WhisperModel")
    def test_empty_text_on_no_speech(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.return_value = (iter([]), _make_mock_info())
        config = _make_transcription_settings()
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber.transcribe(audio)
        assert result.text == ""

    @patch("vox.transcriber_fw.WhisperModel")
    def test_strips_whitespace(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.return_value = (
            iter(_make_mock_segments(["  hello  "])),
            _make_mock_info(),
        )
        config = _make_transcription_settings()
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber.transcribe(audio)
        assert result.text == "hello"

    @patch("vox.transcriber_fw.WhisperModel")
    def test_joins_multiple_segments(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.return_value = (
            iter(_make_mock_segments(["hello", "world", "test"])),
            _make_mock_info(),
        )
        config = _make_transcription_settings()
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber.transcribe(audio)
        assert result.text == "hello world test"

    @patch("vox.transcriber_fw.WhisperModel")
    def test_transcription_error_on_failure(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.side_effect = Exception("model error")
        config = _make_transcription_settings()
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(TranscriptionError, match="model error"):
            transcriber.transcribe(audio)

    @patch("vox.transcriber_fw.WhisperModel")
    def test_respects_language_config(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.return_value = (
            iter(_make_mock_segments(["test"])),
            _make_mock_info(),
        )
        config = _make_transcription_settings(language="en")
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        transcriber.transcribe(audio)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"

    @patch("vox.transcriber_fw.WhisperModel")
    def test_respects_vad_config(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.return_value = (
            iter(_make_mock_segments(["test"])),
            _make_mock_info(),
        )
        config = _make_transcription_settings(vad=False)
        transcriber = FasterWhisperTranscriber(config)
        audio = np.zeros(16000, dtype=np.float32)
        transcriber.transcribe(audio)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is False

    @patch("vox.transcriber_fw.WhisperModel")
    def test_shutdown_deletes_model(self, mock_model_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_model.transcribe.return_value = (
            iter(_make_mock_segments(["test"])),
            _make_mock_info(),
        )
        config = _make_transcription_settings()
        transcriber = FasterWhisperTranscriber(config)
        assert hasattr(transcriber, "_model")
        transcriber.shutdown()
        assert not hasattr(transcriber, "_model")
