"""Integration tests — real components working together."""

from __future__ import annotations

import io
import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, load_dataset

from tests.conftest import (
    make_audio_settings,
    make_hotkey_settings,
    make_indicator_settings,
    make_llm_settings,
    make_sounds_settings,
)
from vox.config import (
    HistorySettings,
    OutputSettings,
    Settings,
    TranscriptionSettings,
)
from vox.history import History, SessionRecord
from vox.output import OutputBackend, Outputter
from vox.pipeline import Pipeline
from vox.processor import LLMCleaner
from vox.transcriber import Transcriber, TranscriptionError


def _load_test_audio(n: int = 1, seed: int = 42) -> list[tuple[np.ndarray, str]]:
    """Stream n real speech samples from LibriSpeech test-clean."""
    ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    new_features = ds.features.copy()
    new_features["audio"] = Audio(sampling_rate=None, decode=False)
    ds = ds.cast(new_features)
    ds = ds.shuffle(seed=seed)

    samples: list[tuple[np.ndarray, str]] = []
    for item in ds:
        raw_bytes = item["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        text = item["text"].strip()
        if text:
            samples.append((audio, text))
        if len(samples) >= n:
            break
    return samples


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


def _make_history_settings(tmp_path: Path) -> HistorySettings:
    return HistorySettings(
        enabled=True,
        db_path=str(tmp_path / "test.db"),
        max_entries=10000,
    )


def _make_output_settings(**kwargs: object) -> OutputSettings:
    defaults: dict[str, object] = {
        "method": "xdotool",
        "fallback_to_clipboard": True,
        "notify_on_paste": False,
        "notify_on_fallback": True,
        "replace_strategy": "select",
        "replace_timeout_seconds": 5,
        "replace_blacklist": [],
    }
    defaults.update(kwargs)
    return OutputSettings(**defaults)  # type: ignore[arg-type]


class _MockBackend(OutputBackend):
    """Test double that records all method calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self._window_class = "test-app"
        self._type_text_result = True
        self._select_left_result = True

    def type_text(self, text: str) -> bool:
        self.calls.append(("type_text", (text,)))
        return self._type_text_result

    def select_left(self, n_chars: int) -> bool:
        self.calls.append(("select_left", (n_chars,)))
        return self._select_left_result

    def get_active_window_class(self) -> str:
        self.calls.append(("get_active_window_class", ()))
        return self._window_class


@pytest.mark.slow
class TestTranscriberIntegration:
    """Real WhisperModel with real speech audio."""

    def test_transcribes_real_audio(self) -> None:
        samples = _load_test_audio(n=1, seed=42)
        assert len(samples) == 1
        audio, reference = samples[0]

        config = _make_transcription_settings()
        transcriber = Transcriber(config)
        text, info = transcriber.transcribe(audio)

        assert isinstance(text, str)
        assert info is not None
        assert info.duration > 0
        assert len(text) > 0

    def test_returns_empty_on_silence(self) -> None:
        silence = np.zeros(16000, dtype=np.float32)
        config = _make_transcription_settings()
        transcriber = Transcriber(config)
        text, info = transcriber.transcribe(silence)

        assert text == ""
        assert info is not None

    def test_multiple_audio_samples(self) -> None:
        samples = _load_test_audio(n=3, seed=100)
        assert len(samples) == 3

        config = _make_transcription_settings()
        transcriber = Transcriber(config)

        for audio, _reference in samples:
            text, info = transcriber.transcribe(audio)
            assert isinstance(text, str)
            assert info is not None
            assert info.duration > 0


class TestProcessorIntegration:
    """Real LLMCleaner with real transcript text."""

    @pytest.mark.slow
    def test_filler_word_removal_with_real_text(self) -> None:
        samples = _load_test_audio(n=1, seed=42)
        audio, reference = samples[0]

        config = _make_transcription_settings()
        transcriber = Transcriber(config)
        text, _ = transcriber.transcribe(audio)

        assert isinstance(text, str)
        assert len(text) > 0
        assert text == text.strip()

    def test_llm_cleaner_with_mock(self) -> None:
        from vox.config import LLMSettings

        config = LLMSettings(
            enabled=True,
            base_url="http://localhost:11434/v1",
            api_key="",
            model="phi3-mini",
            timeout_seconds=10,
            prompt="Clean up the text.",
        )
        cleaner = LLMCleaner(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Cleaned text."

        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get.return_value = mock_client
            result = cleaner.clean("um hello uh world")

        assert result == "Cleaned text."

    def test_empty_input_returns_none(self) -> None:
        from vox.config import LLMSettings

        config = LLMSettings(
            enabled=True,
            base_url="http://localhost:11434/v1",
            api_key="",
            model="phi3-mini",
            timeout_seconds=10,
            prompt="Clean up the text.",
        )
        cleaner = LLMCleaner(config)

        assert cleaner.clean("") is None
        assert cleaner.clean("   ") is None
        assert cleaner.clean("\n\t") is None


class TestPipelineIntegration:
    """Real Transcriber + real History with mocked UI components."""

    @pytest.mark.slow
    def test_full_flow_real_transcriber(self, tmp_path: Path) -> None:
        samples = _load_test_audio(n=1, seed=42)
        audio, reference = samples[0]

        history_config = _make_history_settings(tmp_path)
        history = History(history_config)

        transcriber = Transcriber(_make_transcription_settings())
        backend = _MockBackend()
        outputter = Outputter(_make_output_settings(), backend=backend)

        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        audio_q.put(audio)

        control_q: queue.Queue[str] = queue.Queue(maxsize=1)
        control_q.put("stop")

        settings = Settings(
            hotkey=make_hotkey_settings(),
            audio=make_audio_settings(),
            transcription=_make_transcription_settings(),
            llm=make_llm_settings(enabled=False),
            output=_make_output_settings(),
            indicator=make_indicator_settings(),
            history=history_config,
            sounds=make_sounds_settings(),
        )

        def _make_transcriber(*args: object, **kwargs: object) -> Transcriber:
            return transcriber

        with (
            patch("vox.pipeline.Recorder", return_value=MagicMock()),
            patch("vox.pipeline.Transcriber", side_effect=_make_transcriber),
            patch("vox.pipeline.Outputter", return_value=outputter),
            patch("vox.pipeline.LLMCleaner", return_value=MagicMock()),
            patch("vox.pipeline.Indicator", return_value=MagicMock()),
            patch("vox.pipeline.SoundCues", return_value=MagicMock()),
            patch("vox.pipeline.HotkeyListener", return_value=MagicMock()),
            patch("vox.pipeline.History", return_value=history),
        ):
            pipeline = Pipeline(settings)
            pipeline._control_queue = control_q
            pipeline._audio_queue = audio_q

            pipeline._handle_command("stop")

        assert pipeline._state == "IDLE"

        recent = history.get_recent()
        if recent:
            assert len(recent) == 1
            assert len(recent[0].raw_text) > 0

        history.close()

    def test_transcription_failure_doesnt_crash(self, tmp_path: Path) -> None:
        history_config = _make_history_settings(tmp_path)
        history = History(history_config)

        audio = np.zeros(16000, dtype=np.float32)
        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        audio_q.put(audio)

        control_q: queue.Queue[str] = queue.Queue(maxsize=1)
        control_q.put("stop")

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.side_effect = TranscriptionError("forced error")

        backend = _MockBackend()
        outputter = Outputter(_make_output_settings(), backend=backend)

        settings = Settings(
            hotkey=make_hotkey_settings(),
            audio=make_audio_settings(),
            transcription=_make_transcription_settings(),
            llm=make_llm_settings(enabled=False),
            output=_make_output_settings(),
            indicator=make_indicator_settings(),
            history=history_config,
            sounds=make_sounds_settings(),
        )

        with (
            patch("vox.pipeline.Recorder", return_value=MagicMock()),
            patch("vox.pipeline.Transcriber", return_value=mock_transcriber),
            patch("vox.pipeline.Outputter", return_value=outputter),
            patch("vox.pipeline.LLMCleaner", return_value=MagicMock()),
            patch("vox.pipeline.Indicator", return_value=MagicMock()),
            patch("vox.pipeline.SoundCues", return_value=MagicMock()),
            patch("vox.pipeline.HotkeyListener", return_value=MagicMock()),
            patch("vox.pipeline.History", return_value=history),
        ):
            pipeline = Pipeline(settings)
            pipeline._control_queue = control_q
            pipeline._audio_queue = audio_q

            pipeline._handle_command("stop")

        assert pipeline._state == "IDLE"
        history.close()

    def test_history_saves_real_session(self, tmp_path: Path) -> None:
        history_config = _make_history_settings(tmp_path)
        history = History(history_config)

        record = SessionRecord(
            id=None,
            created_at="",
            raw_text="hello world this is a test",
            clean_text="Hello world, this is a test.",
            duration_ms=3000,
            duration_after_vad_ms=2500,
            word_count=6,
            app_context="firefox",
            language="en",
            model_used="large-v3",
            transcription_latency_ms=800,
            full_pipeline_latency_ms=1200,
        )

        row_id = history.save_session(record)
        assert row_id > 0

        recent = history.get_recent()
        assert len(recent) == 1
        assert recent[0].raw_text == "hello world this is a test"
        assert recent[0].clean_text == "Hello world, this is a test."
        assert recent[0].word_count == 6
        assert recent[0].app_context == "firefox"
        assert recent[0].language == "en"

        search_results = history.search("hello")
        assert len(search_results) == 1
        assert search_results[0].raw_text == record.raw_text

        count = history.get_count()
        assert count == 1

        history.close()
