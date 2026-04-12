"""Tests for transcriber_cw — CarelessWhisper streaming adapter."""

from __future__ import annotations

import dataclasses
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.config import StreamingSettings, TranscriptionSettings
from vox.transcriber_base import StreamingUpdate, TranscriptionError, TranscriptionResult
from vox.transcriber_cw import CarelessWhisperTranscriber


def _make_transcription_settings(**kwargs: object) -> TranscriptionSettings:
    defaults: dict[str, object] = {
        "model": "large-v2",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
        "vad": True,
    }
    defaults.update(kwargs)
    return TranscriptionSettings(**defaults)  # type: ignore[arg-type]


def _make_streaming_settings(**kwargs: object) -> StreamingSettings:
    defaults: dict[str, object] = {
        "enabled": True,
        "model": "large-v2",
        "chunk_size_ms": 300,
        "beam_size": 0,
    }
    defaults.update(kwargs)
    return StreamingSettings(**defaults)  # type: ignore[arg-type]


def _make_audio(samples: int = 4800) -> np.ndarray:
    """300ms of silence at 16kHz (default chunk)."""
    return np.zeros(samples, dtype=np.float32)


def _make_mock_model() -> MagicMock:
    model = MagicMock()
    mock_mel = MagicMock()
    mock_mel.squeeze.return_value = mock_mel
    model.spec_streamer.calc_mel_with_new_frame.return_value = mock_mel
    model.spec_streamer.reset = MagicMock()
    decode_result = MagicMock()
    decode_result.text = "hello world"
    decode_result.language = "en"
    model.decode.return_value = decode_result
    return model


def _make_transcriber(
    tc: TranscriptionSettings | None = None,
    sc: StreamingSettings | None = None,
    model_to_load: MagicMock | None = None,
) -> tuple[CarelessWhisperTranscriber, MagicMock]:
    """Create a CarelessWhisperTranscriber with whisper_rt mocked.

    Returns (transcriber, loaded_model).
    """
    if tc is None:
        tc = _make_transcription_settings()
    if sc is None:
        sc = _make_streaming_settings()
    if model_to_load is None:
        model_to_load = _make_mock_model()
    mock_module = MagicMock()
    mock_module.load_streaming_model.return_value = model_to_load
    with patch.dict("sys.modules", {"whisper_rt": mock_module}):
        transcriber = CarelessWhisperTranscriber(tc, sc)
    return transcriber, model_to_load


class TestCarelessWhisperTranscriberInit:
    def test_stores_configs(self) -> None:
        tc = _make_transcription_settings()
        sc = _make_streaming_settings()
        mock_model = _make_mock_model()
        transcriber, loaded_model = _make_transcriber(tc, sc, model_to_load=mock_model)
        assert transcriber._transcription_config is tc
        assert transcriber._streaming_config is sc
        assert transcriber._model is mock_model
        assert transcriber._streaming_active is False

    def test_chunk_samples_calculation(self) -> None:
        sc = _make_streaming_settings(chunk_size_ms=300)
        transcriber, _ = _make_transcriber(sc=sc)
        assert transcriber._chunk_samples == 4800

    def test_chunk_samples_100ms(self) -> None:
        sc = _make_streaming_settings(chunk_size_ms=100)
        transcriber, _ = _make_transcriber(sc=sc)
        assert transcriber._chunk_samples == 1600

    def test_supports_streaming(self) -> None:
        transcriber, _ = _make_transcriber()
        assert transcriber.supports_streaming is True


class TestCarelessWhisperModelLoading:
    def test_import_error_raises_transcription_error(self) -> None:
        tc = _make_transcription_settings()
        sc = _make_streaming_settings()
        with (
            patch.dict("sys.modules", {"whisper_rt": None}),
            patch("importlib.util.find_spec", return_value=None),
        ):
            with pytest.raises(TranscriptionError, match="not found"):
                CarelessWhisperTranscriber(tc, sc)

    def test_loads_model_once(self) -> None:
        mock_model = _make_mock_model()
        mock_cws = MagicMock()
        mock_cws.load_streaming_model.return_value = mock_model
        tc = _make_transcription_settings()
        sc = _make_streaming_settings()
        with patch.dict("sys.modules", {"whisper_rt": mock_cws}):
            transcriber = CarelessWhisperTranscriber(tc, sc)
            transcriber._load_model()
            mock_cws.load_streaming_model.assert_called_once()

    def test_load_failure_raises_transcription_error(self) -> None:
        mock_cws = MagicMock()
        mock_cws.load_streaming_model.side_effect = RuntimeError("HF auth failed")
        tc = _make_transcription_settings()
        sc = _make_streaming_settings()
        with patch.dict("sys.modules", {"whisper_rt": mock_cws}):
            with pytest.raises(TranscriptionError, match="HF auth failed"):
                CarelessWhisperTranscriber(tc, sc)


class TestOfflineTranscription:
    def test_offline_mode_calls_non_causal_transcribe(self) -> None:
        mock_model = _make_mock_model()
        mock_model.non_causal_transcribe.return_value = ("hello world", MagicMock())
        transcriber, _ = _make_transcriber(model_to_load=mock_model)

        audio = _make_audio()
        result = transcriber.transcribe(audio)

        mock_model._cancel_streaming_mode.assert_called_once()
        mock_model.non_causal_transcribe.assert_called_once_with(audio)
        mock_model._revert_streaming_mode.assert_called_once()
        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"

    def test_offline_mode_with_segments(self) -> None:
        mock_model = _make_mock_model()
        seg1 = MagicMock()
        seg1.text = "hello"
        seg2 = MagicMock()
        seg2.text = "world"
        mock_model.non_causal_transcribe.return_value = [seg1, seg2]
        transcriber, _ = _make_transcriber(model_to_load=mock_model)

        result = transcriber.transcribe(_make_audio())
        assert result.text == "hello world"

    def test_error_raises_transcription_error(self) -> None:
        mock_model = _make_mock_model()
        mock_model.non_causal_transcribe.side_effect = RuntimeError("decode failed")
        transcriber, _ = _make_transcriber(model_to_load=mock_model)

        with pytest.raises(TranscriptionError, match="decode failed"):
            transcriber.transcribe(_make_audio())


class TestStreamingLifecycle:
    def test_start_streaming_initializes_state(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.start_streaming()
        assert transcriber._streaming_active is True
        transcriber.stop_streaming()

    def test_start_streaming_ignored_if_already_active(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.start_streaming()
        # Second start should be a no-op
        transcriber.start_streaming()
        transcriber.stop_streaming()

    def test_feed_chunk_buffers_audio(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.start_streaming()
        audio = _make_audio(1000)
        transcriber.feed_chunk(audio)
        # Audio should be in _session_audio (worker may not have processed yet)
        assert len(transcriber._session_audio) > 0
        transcriber.stop_streaming()

    def test_feed_chunk_triggers_decode_at_threshold(self) -> None:
        mock_model = _make_mock_model()
        transcriber, loaded_model = _make_transcriber(model_to_load=mock_model)
        transcriber.start_streaming()

        # Feed enough audio to trigger a decode
        audio = _make_audio(4800)
        transcriber.feed_chunk(audio)

        # Wait for worker thread to process
        time.sleep(0.2)

        assert loaded_model.decode.called
        update = transcriber.get_update()
        assert update is not None
        assert update.partial_text == "hello world"
        assert update.is_final is False
        transcriber.stop_streaming()

    def test_feed_chunk_emits_streaming_update(self) -> None:
        mock_model = _make_mock_model()
        transcriber, _ = _make_transcriber(model_to_load=mock_model)
        transcriber.start_streaming()

        transcriber.feed_chunk(_make_audio(4800))
        time.sleep(0.2)

        update = transcriber.get_update()
        assert update is not None
        assert isinstance(update, StreamingUpdate)
        assert update.duration > 0
        transcriber.stop_streaming()

    def test_get_update_returns_none_when_empty(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.start_streaming()
        # No audio fed → no update
        assert transcriber.get_update() is None
        transcriber.stop_streaming()

    def test_stop_streaming_returns_result(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.start_streaming()
        transcriber.feed_chunk(_make_audio(4800))
        time.sleep(0.1)
        result = transcriber.stop_streaming()
        assert isinstance(result, TranscriptionResult)
        assert transcriber._streaming_active is False

    def test_stop_streaming_when_inactive_raises(self) -> None:
        transcriber, _ = _make_transcriber()
        with pytest.raises(TranscriptionError, match="not active"):
            transcriber.stop_streaming()

    def test_transcribe_while_streaming_raises(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.start_streaming()
        with pytest.raises(TranscriptionError, match="streaming is active"):
            transcriber.transcribe(_make_audio())
        transcriber.stop_streaming()

    @pytest.mark.skip(reason="Requires real torch + CarelessWhisper model")
    def test_multiple_chunks_accumulate_text(self) -> None:
        mock_model = _make_mock_model()
        result1 = MagicMock()
        result1.text = "hello"
        result2 = MagicMock()
        result2.text = "hello world"
        mock_model.decode.side_effect = [result1, result2]
        transcriber, loaded_model = _make_transcriber(model_to_load=mock_model)
        transcriber.start_streaming()

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.from_numpy.return_value.float.return_value.to.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            transcriber.feed_chunk(_make_audio(4800))
            transcriber.feed_chunk(_make_audio(4800))

        assert loaded_model.decode.call_count >= 1
        transcriber.stop_streaming()


class TestStreamingThreadSafety:
    def test_feed_chunk_from_multiple_threads(self) -> None:
        mock_model = _make_mock_model()
        transcriber, _ = _make_transcriber(model_to_load=mock_model)
        transcriber.start_streaming()

        errors: list[Exception] = []

        def feed() -> None:
            try:
                for _ in range(10):
                    transcriber.feed_chunk(_make_audio(480))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=feed) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        transcriber.stop_streaming()
        assert len(errors) == 0

    def test_concurrent_start_and_feed(self) -> None:
        mock_model = _make_mock_model()
        transcriber, _ = _make_transcriber(model_to_load=mock_model)

        started = threading.Event()
        errors: list[Exception] = []

        def starter() -> None:
            try:
                transcriber.start_streaming()
                started.set()
            except Exception as e:
                errors.append(e)

        def feeder() -> None:
            started.wait(timeout=1)
            try:
                transcriber.feed_chunk(_make_audio(4800))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=starter)
        t2 = threading.Thread(target=feeder)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        transcriber.stop_streaming()
        assert len(errors) == 0


class TestShutdown:
    def test_shutdown_deletes_model(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.shutdown()
        assert transcriber._model is None

    def test_shutdown_is_idempotent(self) -> None:
        transcriber, _ = _make_transcriber()
        transcriber.shutdown()
        transcriber.shutdown()


class TestStreamingUpdate:
    def test_is_frozen(self) -> None:
        update = StreamingUpdate(
            partial_text="hello",
            stable_text="hello",
            is_final=False,
            language="en",
            duration=0.3,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            update.partial_text = "changed"  # type: ignore[misc]

    def test_has_expected_fields(self) -> None:
        update = StreamingUpdate(
            partial_text="test",
            stable_text="test",
            is_final=True,
            language=None,
            duration=1.0,
        )
        assert update.partial_text == "test"
        assert update.stable_text == "test"
        assert update.is_final is True
        assert update.language is None
        assert update.duration == 1.0
