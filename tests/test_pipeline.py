"""Tests for pipeline — orchestration and state machine."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from vox.config import Settings
from vox.pipeline import Pipeline
from vox.transcriber import TranscriptionError


def _make_mock_settings(**kwargs: object) -> Settings:
    from tests.conftest import (
        make_audio_settings,
        make_dictionary_settings,
        make_history_settings,
        make_hotkey_settings,
        make_indicator_settings,
        make_llm_settings,
        make_output_settings,
        make_sounds_settings,
        make_streaming_settings,
        make_transcription_settings,
    )

    defaults: dict[str, object] = {
        "hotkey": make_hotkey_settings(),
        "audio": make_audio_settings(),
        "transcription": make_transcription_settings(),
        "llm": make_llm_settings(enabled=False),
        "output": make_output_settings(),
        "indicator": make_indicator_settings(),
        "history": make_history_settings(enabled=True),
        "sounds": make_sounds_settings(),
        "dictionary": make_dictionary_settings(),
        "streaming": make_streaming_settings(),
    }
    defaults.update(kwargs)
    return Settings(**defaults)  # type: ignore[arg-type]


def _make_mock_info(**kwargs: object) -> MagicMock:
    info = MagicMock()
    info.duration = 3.0
    info.duration_after_vad = 2.5
    info.language = "en"
    info.language_probability = 0.95
    for key, value in kwargs.items():
        setattr(info, key, value)
    return info


def _make_mock_transcription_result(**kwargs: object) -> MagicMock:
    """Create a mock TranscriptionResult with sensible defaults."""
    result = MagicMock()
    result.text = "hello world"
    result.info = _make_mock_info()
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


def _make_pipeline(
    mock_transcriber: MagicMock | None = None,
    mock_outputter: MagicMock | None = None,
    mock_history: MagicMock | None = None,
    mock_llm_cleaner: MagicMock | None = None,
    settings_overrides: dict[str, object] | None = None,
) -> tuple[Pipeline, dict[str, MagicMock]]:
    from tests.conftest import (
        make_history_settings,
        make_llm_settings,
    )

    history_settings = make_history_settings(enabled=True)
    llm_settings = make_llm_settings(enabled=False)

    if settings_overrides:
        if "history" in settings_overrides:
            history_settings = settings_overrides["history"]
        if "llm" in settings_overrides:
            llm_settings = settings_overrides["llm"]

    settings = _make_mock_settings(
        history=history_settings,
        llm=llm_settings,
    )

    mock_recorder = MagicMock()
    if mock_transcriber is None:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = _make_mock_transcription_result()
    if mock_outputter is None:
        mock_outputter = MagicMock()
    if mock_llm_cleaner is None:
        mock_llm_cleaner = MagicMock()
    mock_indicator = MagicMock()
    mock_sound_cues = MagicMock()
    mock_hotkey = MagicMock()
    if mock_history is None:
        mock_history = MagicMock()

    mocks = {
        "recorder": mock_recorder,
        "transcriber": mock_transcriber,
        "outputter": mock_outputter,
        "llm_cleaner": mock_llm_cleaner,
        "indicator": mock_indicator,
        "sound_cues": mock_sound_cues,
        "hotkey": mock_hotkey,
        "history": mock_history,
    }

    with (
        patch("vox.pipeline.Recorder", return_value=mock_recorder),
        patch("vox.pipeline.make_transcriber", return_value=mock_transcriber),
        patch("vox.pipeline.Outputter", return_value=mock_outputter),
        patch("vox.pipeline.LLMCleaner", return_value=mock_llm_cleaner),
        patch("vox.pipeline.Indicator", return_value=mock_indicator),
        patch("vox.pipeline.SoundCues", return_value=mock_sound_cues),
        patch("vox.pipeline.HotkeyListener", return_value=mock_hotkey),
        patch("vox.pipeline.History", return_value=mock_history),
    ):
        pipeline = Pipeline(settings)

    pipeline._recorder = mock_recorder
    pipeline._transcriber = mock_transcriber
    pipeline._outputter = mock_outputter
    pipeline._llm_cleaner = mock_llm_cleaner
    pipeline._indicator = mock_indicator
    pipeline._sound_cues = mock_sound_cues
    pipeline._hotkey = mock_hotkey
    pipeline._history = mock_history

    return pipeline, mocks


class TestPipelineInit:
    def test_initial_state_is_idle(self) -> None:
        pipeline, _ = _make_pipeline()
        assert pipeline._state == "IDLE"

    def test_history_none_when_disabled(self) -> None:
        from tests.conftest import make_history_settings

        settings = _make_mock_settings(history=make_history_settings(enabled=False))
        with (
            patch("vox.pipeline.Recorder", return_value=MagicMock()),
            patch("vox.pipeline.make_transcriber", return_value=MagicMock()),
            patch("vox.pipeline.Outputter", return_value=MagicMock()),
            patch("vox.pipeline.LLMCleaner", return_value=MagicMock()),
            patch("vox.pipeline.Indicator", return_value=MagicMock()),
            patch("vox.pipeline.SoundCues", return_value=MagicMock()),
            patch("vox.pipeline.HotkeyListener", return_value=MagicMock()),
        ):
            pipeline = Pipeline(settings)
        assert pipeline._history is None


class TestStateTransitions:
    def test_start_transitions_to_recording(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._handle_command("start")
        assert pipeline._state == "RECORDING"
        mocks["indicator"].show.assert_called_once()
        mocks["sound_cues"].play_start.assert_called_once()
        mocks["recorder"].start_recording.assert_called_once()

    def test_stop_transitions_through_full_cycle(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.return_value = _make_mock_transcription_result()
        mocks["outputter"].paste.return_value = 11
        mocks["outputter"].get_active_window_class.return_value = "test-app"
        pipeline._handle_command("stop")
        assert pipeline._state == "IDLE"
        mocks["indicator"].hide.assert_called_once()
        mocks["sound_cues"].play_stop.assert_called_once()
        mocks["recorder"].stop_recording.assert_called_once()
        mocks["transcriber"].transcribe.assert_called_once()
        mocks["outputter"].paste.assert_called_once()

    def test_start_ignored_when_not_idle(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        pipeline._handle_command("start")
        assert pipeline._state == "RECORDING"
        assert mocks["indicator"].show.call_count == 0

    def test_stop_ignored_when_not_recording(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "IDLE"
        pipeline._handle_command("stop")
        assert pipeline._state == "IDLE"
        mocks["recorder"].stop_recording.assert_not_called()

    def test_shutdown_sets_flag(self) -> None:
        pipeline, _ = _make_pipeline()
        pipeline._handle_command("shutdown")
        assert pipeline._shutdown.is_set()


class TestAudioProcessing:
    def test_transcription_failure_returns_to_idle(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.side_effect = TranscriptionError("fail")
        pipeline._handle_command("stop")
        assert pipeline._state == "IDLE"

    def test_empty_transcription_returns_to_idle(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.return_value = _make_mock_transcription_result(text="")
        pipeline._handle_command("stop")
        assert pipeline._state == "IDLE"
        mocks["outputter"].paste.assert_not_called()

    def test_no_audio_after_stop_returns_to_idle(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        pipeline._handle_command("stop")
        assert pipeline._state == "IDLE"

    def test_successful_session_saved_to_history(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.return_value = _make_mock_transcription_result()
        mocks["outputter"].paste.return_value = 11
        mocks["outputter"].get_active_window_class.return_value = "test-app"
        pipeline._handle_command("stop")
        mocks["history"].save_session.assert_called_once()

    def test_history_disabled_skips_save(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._history = None
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.return_value = _make_mock_transcription_result()
        mocks["outputter"].paste.return_value = 11
        mocks["outputter"].get_active_window_class.return_value = "test-app"
        pipeline._handle_command("stop")


class TestLLMReplacement:
    def test_llm_thread_spawned_when_enabled(self) -> None:
        from tests.conftest import make_llm_settings

        pipeline, mocks = _make_pipeline(
            settings_overrides={"llm": make_llm_settings(enabled=True)}
        )
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.return_value = _make_mock_transcription_result()
        mocks["outputter"].paste.return_value = 11
        mocks["outputter"].get_active_window_class.return_value = "test-app"
        pipeline._handle_command("stop")
        time.sleep(0.2)
        assert mocks["llm_cleaner"].clean.call_count == 1

    def test_llm_thread_not_spawned_when_disabled(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.return_value = _make_mock_transcription_result()
        mocks["outputter"].paste.return_value = 11
        mocks["outputter"].get_active_window_class.return_value = "test-app"
        active_before = threading.active_count()
        pipeline._handle_command("stop")
        time.sleep(0.1)
        active_after = threading.active_count()
        assert active_after == active_before

    def test_llm_replace_applies_cleaned_text(self) -> None:
        pipeline, mocks = _make_pipeline()
        mocks["llm_cleaner"].clean.return_value = "Cleaned text."
        pipeline._llm_replace("raw text", 8, "test-app", time.monotonic())
        mocks["outputter"].replace.assert_called_once_with(
            "Cleaned text.",
            8,
            "test-app",
            pipeline._outputter.replace.call_args[0][3]
            if hasattr(pipeline._outputter, "replace")
            else 0,
        )

    def test_llm_replace_skips_when_same(self) -> None:
        pipeline, mocks = _make_pipeline()
        mocks["llm_cleaner"].clean.return_value = "raw text"
        pipeline._llm_replace("raw text", 8, "test-app", time.monotonic())
        mocks["outputter"].replace.assert_not_called()

    def test_llm_replace_skips_when_none(self) -> None:
        pipeline, mocks = _make_pipeline()
        mocks["llm_cleaner"].clean.return_value = None
        pipeline._llm_replace("raw text", 8, "test-app", time.monotonic())
        mocks["outputter"].replace.assert_not_called()


class TestIndicatorsAndSounds:
    def test_show_on_start(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._handle_command("start")
        mocks["indicator"].show.assert_called_once()
        mocks["sound_cues"].play_start.assert_called_once()

    def test_hide_on_stop(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        audio = np.zeros(16000, dtype=np.float32)
        pipeline._audio_queue.put(audio)
        mocks["transcriber"].transcribe.return_value = _make_mock_transcription_result(
            text="hello",
        )
        mocks["outputter"].paste.return_value = 5
        mocks["outputter"].get_active_window_class.return_value = "test-app"
        pipeline._handle_command("stop")
        mocks["indicator"].hide.assert_called_once()
        mocks["sound_cues"].play_stop.assert_called_once()


class TestShutdown:
    def test_shutdown_stops_hotkey_and_indicator(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._shutdown_threads()
        mocks["hotkey"].stop.assert_called_once()
        mocks["indicator"].stop.assert_called_once()

    def test_shutdown_closes_history(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._shutdown_threads()
        mocks["history"].close.assert_called_once()

    def test_shutdown_stops_recorder_when_recording(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "RECORDING"
        pipeline._shutdown_threads()
        mocks["recorder"].stop_recording.assert_called_once()

    def test_shutdown_skips_recorder_when_idle(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._state = "IDLE"
        pipeline._shutdown_threads()
        mocks["recorder"].stop_recording.assert_not_called()
