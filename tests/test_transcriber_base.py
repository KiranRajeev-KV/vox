"""Tests for transcriber_base — ABC contract and shared types."""

from __future__ import annotations

import numpy as np
import pytest

from vox.transcriber_base import (
    TranscriberBase,
    TranscriptionError,
    TranscriptionResult,
)


def _make_mock_info() -> object:
    """Create a minimal TranscriptionInfo mock."""
    from unittest.mock import MagicMock

    info = MagicMock()
    info.duration = 3.0
    info.duration_after_vad = 2.5
    info.language = "en"
    info.language_probability = 0.95
    return info


class TestTranscriptionError:
    def test_can_wrap_another_exception(self) -> None:
        inner = ValueError("inner")
        wrapped = TranscriptionError("outer")
        wrapped.__cause__ = inner
        assert isinstance(wrapped, TranscriptionError)
        assert wrapped.__cause__ is inner


class TestTranscriptionResult:
    def test_constructable(self) -> None:
        info = _make_mock_info()
        result = TranscriptionResult(text="hello", info=info)  # type: ignore[arg-type]
        assert result.text == "hello"
        assert result.info is info


class TestTranscriberBaseABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            TranscriberBase()  # type: ignore[abstract]

    def test_subclass_must_implement_transcribe(self) -> None:
        class IncompleteTranscriber(TranscriberBase):
            def shutdown(self) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteTranscriber()

    def test_subclass_must_implement_shutdown(self) -> None:
        class IncompleteTranscriber(TranscriberBase):
            def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
                raise NotImplementedError

        with pytest.raises(TypeError):
            IncompleteTranscriber()

    def test_complete_implementation_works(self) -> None:
        class FakeTranscriber(TranscriberBase):
            def __init__(self) -> None:
                self._shutdown_called = False

            def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
                info = _make_mock_info()
                return TranscriptionResult(text="fake", info=info)  # type: ignore[arg-type]

            def shutdown(self) -> None:
                self._shutdown_called = True

        t = FakeTranscriber()
        result = t.transcribe(np.zeros(16000, dtype=np.float32))
        assert result.text == "fake"
        t.shutdown()
        assert t._shutdown_called is True

    def test_streaming_methods_raise_not_implemented_by_default(self) -> None:
        class MinimalTranscriber(TranscriberBase):
            def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
                raise NotImplementedError

            def shutdown(self) -> None:
                pass

        t = MinimalTranscriber()
        with pytest.raises(NotImplementedError):
            t.start_streaming()
        with pytest.raises(NotImplementedError):
            t.feed_chunk(np.zeros(16000, dtype=np.float32))
        with pytest.raises(NotImplementedError):
            t.stop_streaming()

    def test_supports_streaming_property(self) -> None:
        class MinimalTranscriber(TranscriberBase):
            def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
                raise NotImplementedError

            def shutdown(self) -> None:
                pass

        assert MinimalTranscriber().supports_streaming is False
