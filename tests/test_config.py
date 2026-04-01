"""Tests for config loading, parsing, and settings construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vox.config import (
    AudioSettings,
    HistorySettings,
    HotkeySettings,
    IndicatorSettings,
    LLMSettings,
    OutputSettings,
    SoundsSettings,
    TranscriptionSettings,
    _build_settings,
    _load,
    _parse_indicator_width,
    get_settings,
)


def _write_toml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content)
    return path


class TestLoad:
    def test_load_default_config(self) -> None:
        result = _load()
        assert "hotkey" in result
        assert "audio" in result
        assert "transcription" in result
        assert "llm" in result
        assert "output" in result
        assert "indicator" in result
        assert "history" in result
        assert "sounds" in result

    def test_load_custom_path(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, '[hotkey]\ntrigger = "f10"\nmode = "toggle"\n')
        result = _load(path)
        assert result["hotkey"]["trigger"] == "f10"


class TestBuildSettingsDefaults:
    def test_all_defaults_from_empty(self) -> None:
        settings = _build_settings({})
        assert settings.hotkey.trigger == "f9"
        assert settings.hotkey.mode == "toggle"
        assert settings.audio.sample_rate == 16000
        assert settings.audio.channels == 1
        assert settings.audio.device is None
        assert settings.transcription.model == "large-v3"
        assert settings.transcription.device == "cuda"
        assert settings.transcription.compute_type == "int8"
        assert settings.transcription.language is None
        assert settings.transcription.vad is True
        assert settings.llm.enabled is True
        assert settings.llm.base_url == "http://localhost:11434/v1"
        assert settings.llm.api_key == ""
        assert settings.llm.model == "phi3-mini"
        assert settings.llm.timeout_seconds == 10
        assert "Clean up" in settings.llm.prompt
        assert settings.output.method == "xdotool"
        assert settings.output.fallback_to_clipboard is True
        assert settings.output.notify_on_paste is False
        assert settings.output.notify_on_fallback is True
        assert settings.output.replace_strategy == "select"
        assert settings.output.replace_timeout_seconds == 5
        assert settings.output.replace_blacklist == []
        assert settings.indicator.style == "bar"
        assert settings.indicator.position == "top"
        assert settings.indicator.height == 4
        assert settings.indicator.width == "full"
        assert settings.indicator.color == "#ff4444"
        assert settings.indicator.opacity == 0.9
        assert settings.history.enabled is True
        assert "history.db" in settings.history.db_path
        assert settings.history.max_entries == 10000
        assert settings.sounds.enabled is True
        assert "start.wav" in settings.sounds.start_sound
        assert "stop.wav" in settings.sounds.stop_sound
        assert settings.sounds.volume == 0.7

    def test_full_config_mapping(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """
[hotkey]
trigger = "ctrl+shift+space"
mode = "push_to_talk"

[audio]
device = 2
sample_rate = 48000
channels = 2

[transcription]
model = "small"
device = "cpu"
compute_type = "float32"
language = "de"
vad = false

[llm]
enabled = false
base_url = "https://api.openai.com/v1"
api_key = "sk-test"
model = "gpt-4"
timeout_seconds = 30
prompt = "Fix this."

[output]
method = "xdotool"
fallback_to_clipboard = false
notify_on_paste = true
notify_on_fallback = false
replace_strategy = "append"
replace_timeout_seconds = 10
replace_blacklist = ["Vim", "nvim"]

[indicator]
style = "bar"
position = "bottom"
height = 6
width = 400
color = "#00ff00"
opacity = 0.5

[history]
enabled = false
db_path = "/tmp/test.db"
max_entries = 500

[sounds]
enabled = false
start_sound = "/tmp/start.wav"
stop_sound = "/tmp/stop.wav"
volume = 0.3
""",
        )
        raw = _load(path)
        settings = _build_settings(raw)

        assert settings.hotkey.trigger == "ctrl+shift+space"
        assert settings.hotkey.mode == "push_to_talk"
        assert settings.audio.device == 2
        assert settings.audio.sample_rate == 48000
        assert settings.audio.channels == 2
        assert settings.transcription.model == "small"
        assert settings.transcription.device == "cpu"
        assert settings.transcription.compute_type == "float32"
        assert settings.transcription.language == "de"
        assert settings.transcription.vad is False
        assert settings.llm.enabled is False
        assert settings.llm.base_url == "https://api.openai.com/v1"
        assert settings.llm.api_key == "sk-test"
        assert settings.llm.model == "gpt-4"
        assert settings.llm.timeout_seconds == 30
        assert settings.llm.prompt == "Fix this."
        assert settings.output.fallback_to_clipboard is False
        assert settings.output.notify_on_paste is True
        assert settings.output.notify_on_fallback is False
        assert settings.output.replace_strategy == "append"
        assert settings.output.replace_timeout_seconds == 10
        assert settings.output.replace_blacklist == ["Vim", "nvim"]
        assert settings.indicator.position == "bottom"
        assert settings.indicator.height == 6
        assert settings.indicator.width == 400
        assert settings.indicator.color == "#00ff00"
        assert settings.indicator.opacity == 0.5
        assert settings.history.enabled is False
        assert settings.history.db_path == "/tmp/test.db"
        assert settings.history.max_entries == 500
        assert settings.sounds.enabled is False
        assert settings.sounds.volume == 0.3

    def test_missing_sections(self) -> None:
        settings = _build_settings({})
        assert isinstance(settings.hotkey, HotkeySettings)
        assert isinstance(settings.audio, AudioSettings)
        assert isinstance(settings.transcription, TranscriptionSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.output, OutputSettings)
        assert isinstance(settings.indicator, IndicatorSettings)
        assert isinstance(settings.history, HistorySettings)
        assert isinstance(settings.sounds, SoundsSettings)


class TestBuildSettingsEdgeCases:
    def test_empty_string_language_is_none(self, tmp_path: Path) -> None:
        toml = (
            "[transcription]\n"
            'model = "large-v3"\n'
            'device = "cuda"\n'
            'compute_type = "int8"\n'
            'language = ""\n'
            "vad = true\n"
        )
        path = _write_toml(tmp_path, toml)
        raw = _load(path)
        settings = _build_settings(raw)
        assert settings.transcription.language is None

    def test_empty_string_device_is_none(self, tmp_path: Path) -> None:
        toml = '[audio]\nsample_rate = 16000\nchannels = 1\ndevice = ""\n'
        path = _write_toml(tmp_path, toml)
        raw = _load(path)
        settings = _build_settings(raw)
        assert settings.audio.device is None

    def test_device_int_vs_string(self, tmp_path: Path) -> None:
        toml = "[audio]\nsample_rate = 16000\nchannels = 1\ndevice = 3\n"
        path = _write_toml(tmp_path, toml)
        raw = _load(path)
        settings = _build_settings(raw)
        assert settings.audio.device == 3
        assert isinstance(settings.audio.device, int)

    def test_empty_api_key_stays_empty(self, tmp_path: Path) -> None:
        toml = (
            "[llm]\n"
            "enabled = true\n"
            'base_url = "http://localhost:11434/v1"\n'
            'api_key = ""\n'
            'model = "phi3"\n'
            "timeout_seconds = 10\n"
            'prompt = "ok"\n'
        )
        path = _write_toml(tmp_path, toml)
        raw = _load(path)
        settings = _build_settings(raw)
        assert settings.llm.api_key == ""

    def test_empty_blacklist_is_empty_list(self) -> None:
        settings = _build_settings({"output": {"replace_blacklist": []}})
        assert settings.output.replace_blacklist == []

    def test_none_blacklist_is_empty_list(self) -> None:
        settings = _build_settings({"output": {"replace_blacklist": None}})
        assert settings.output.replace_blacklist == []


class TestEnvVarApiKeyOverride:
    def test_env_var_overrides_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VOX_LLM_API_KEY", "env-secret-key")
        toml = (
            "[llm]\n"
            "enabled = true\n"
            'base_url = "http://localhost:11434/v1"\n'
            'api_key = "config-key"\n'
            'model = "phi3"\n'
            "timeout_seconds = 10\n"
            'prompt = "ok"\n'
        )
        path = _write_toml(tmp_path, toml)
        raw = _load(path)
        settings = _build_settings(raw)
        assert settings.llm.api_key == "env-secret-key"

    def test_env_var_used_when_config_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VOX_LLM_API_KEY", "env-secret-key")
        toml = (
            "[llm]\n"
            "enabled = true\n"
            'base_url = "http://localhost:11434/v1"\n'
            'api_key = ""\n'
            'model = "phi3"\n'
            "timeout_seconds = 10\n"
            'prompt = "ok"\n'
        )
        path = _write_toml(tmp_path, toml)
        raw = _load(path)
        settings = _build_settings(raw)
        assert settings.llm.api_key == "env-secret-key"

    def test_no_env_no_config_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VOX_LLM_API_KEY", raising=False)
        toml = (
            "[llm]\n"
            "enabled = true\n"
            'base_url = "http://localhost:11434/v1"\n'
            'api_key = ""\n'
            'model = "phi3"\n'
            "timeout_seconds = 10\n"
            'prompt = "ok"\n'
        )
        path = _write_toml(tmp_path, toml)
        raw = _load(path)
        settings = _build_settings(raw)
        assert settings.llm.api_key == ""


class TestParseIndicatorWidth:
    def test_full_string(self) -> None:
        assert _parse_indicator_width("full") == "full"

    def test_full_string_case_insensitive(self) -> None:
        assert _parse_indicator_width("FULL") == "full"
        assert _parse_indicator_width("Full") == "full"

    def test_pixel_string(self) -> None:
        assert _parse_indicator_width("200") == 200

    def test_pixel_int(self) -> None:
        assert _parse_indicator_width(300) == 300

    def test_zero_falls_back(self) -> None:
        assert _parse_indicator_width(0) == "full"

    def test_negative_falls_back(self) -> None:
        assert _parse_indicator_width(-10) == "full"

    def test_garbage_string_falls_back(self) -> None:
        assert _parse_indicator_width("abc") == "full"

    def test_empty_string_falls_back(self) -> None:
        assert _parse_indicator_width("") == "full"


class TestGetSettings:
    def test_singleton(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_db_path_expands_tilde(self) -> None:
        settings = _build_settings({})
        assert "~" not in settings.history.db_path
        assert Path(settings.history.db_path).is_absolute()


class TestFrozenDataclasses:
    def test_hotkey_settings_immutable(self) -> None:
        s = HotkeySettings(trigger="f9", mode="toggle")
        with pytest.raises(FrozenInstanceError):
            s.trigger = "f10"

    def test_audio_settings_immutable(self) -> None:
        s = AudioSettings(device=None, sample_rate=16000, channels=1)
        with pytest.raises(FrozenInstanceError):
            s.sample_rate = 48000

    def test_transcription_settings_immutable(self) -> None:
        s = TranscriptionSettings(
            model="large-v3", device="cuda", compute_type="int8", language=None, vad=True
        )
        with pytest.raises(FrozenInstanceError):
            s.model = "small"

    def test_llm_settings_immutable(self) -> None:
        s = LLMSettings(
            enabled=True,
            base_url="http://localhost:11434/v1",
            api_key="",
            model="phi3",
            timeout_seconds=10,
            prompt="ok",
        )
        with pytest.raises(FrozenInstanceError):
            s.enabled = False

    def test_output_settings_immutable(self) -> None:
        s = OutputSettings(
            method="xdotool",
            fallback_to_clipboard=True,
            notify_on_paste=False,
            notify_on_fallback=True,
            replace_strategy="select",
            replace_timeout_seconds=5,
            replace_blacklist=[],
        )
        with pytest.raises(FrozenInstanceError):
            s.replace_strategy = "append"

    def test_indicator_settings_immutable(self) -> None:
        s = IndicatorSettings(
            style="bar", position="top", height=4, width="full", color="#ff4444", opacity=0.9
        )
        with pytest.raises(FrozenInstanceError):
            s.height = 10

    def test_history_settings_immutable(self) -> None:
        s = HistorySettings(enabled=True, db_path="/tmp/test.db", max_entries=10000)
        with pytest.raises(FrozenInstanceError):
            s.max_entries = 5000

    def test_sounds_settings_immutable(self) -> None:
        s = SoundsSettings(enabled=True, start_sound="start.wav", stop_sound="stop.wav", volume=0.7)
        with pytest.raises(FrozenInstanceError):
            s.volume = 0.5

    def test_settings_immutable(self) -> None:
        s = _build_settings({})
        with pytest.raises(FrozenInstanceError):
            s.hotkey = HotkeySettings(trigger="f1", mode="toggle")
