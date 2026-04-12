"""Tests for transcriber factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vox.config import StreamingSettings, TranscriptionSettings
from vox.transcriber import make_transcriber
from vox.transcriber_base import TranscriberBase
from vox.transcriber_cw import CarelessWhisperTranscriber
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


def _make_streaming_settings(**kwargs: object) -> StreamingSettings:
    defaults: dict[str, object] = {
        "enabled": False,
        "model": "large-v2",
        "chunk_size_ms": 300,
        "beam_size": 0,
    }
    defaults.update(kwargs)
    return StreamingSettings(**defaults)  # type: ignore[arg-type]


class TestMakeTranscriber:
    def test_returns_faster_whisper_when_streaming_disabled(self) -> None:
        tc = _make_transcription_settings()
        sc = _make_streaming_settings(enabled=False)
        with patch("vox.transcriber_fw.WhisperModel"):
            transcriber = make_transcriber(tc, sc)
        assert isinstance(transcriber, FasterWhisperTranscriber)
        assert isinstance(transcriber, TranscriberBase)

    def test_returns_careless_whisper_when_streaming_enabled(self) -> None:
        tc = _make_transcription_settings()
        sc = _make_streaming_settings(enabled=True)
        mock_model = MagicMock()
        mock_model.spec_streamer.calc_mel_with_new_frame.return_value = MagicMock()
        mock_cws = MagicMock()
        mock_cws.load_streaming_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper_rt": mock_cws}):
            transcriber = make_transcriber(tc, sc)
        assert isinstance(transcriber, CarelessWhisperTranscriber)
        assert isinstance(transcriber, TranscriberBase)

    def test_supports_streaming_property(self) -> None:
        tc = _make_transcription_settings()
        sc = _make_streaming_settings(enabled=False)
        with patch("vox.transcriber_fw.WhisperModel"):
            transcriber = make_transcriber(tc, sc)
        # FasterWhisperTranscriber does NOT support streaming
        assert transcriber.supports_streaming is False

    def test_re_exports_shared_types(self) -> None:
        import vox.transcriber as transcriber_module

        # Ensure the factory module re-exports the shared types
        assert hasattr(transcriber_module, "TranscriptionError")
        assert hasattr(transcriber_module, "TranscriptionResult")
        assert hasattr(transcriber_module, "TranscriberBase")
