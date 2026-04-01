"""Shared fixtures for all Vox tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from vox.config import (
    AudioSettings,
    HistorySettings,
    HotkeySettings,
    IndicatorSettings,
    LLMSettings,
    OutputSettings,
    Settings,
    SoundsSettings,
    TranscriptionSettings,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--benchmark-samples",
        default=100,
        type=int,
        help="Number of samples per dataset for benchmark tests",
    )
    parser.addoption(
        "--benchmark-seed",
        default=42,
        type=int,
        help="Random seed for reproducible benchmark sample selection",
    )
    parser.addoption(
        "--benchmark-tier",
        default="quick",
        type=str,
        choices=["quick", "full"],
        help="Benchmark tier: quick (4 models) or full (12 models)",
    )
    parser.addoption(
        "--benchmark-datasets",
        default="clean,other",
        type=str,
        help="Comma-separated dataset names: clean,other",
    )


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Write a temporary TOML config file and return its path."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("""
[hotkey]
trigger = "f9"
mode = "toggle"

[audio]
device = 0
sample_rate = 16000
channels = 1

[transcription]
model = "large-v3"
device = "cuda"
compute_type = "int8"
language = "en"
vad = true

[llm]
enabled = true
base_url = "http://localhost:11434/v1"
api_key = ""
model = "phi3-mini"
timeout_seconds = 10
prompt = "Clean up the text."

[output]
method = "xdotool"
fallback_to_clipboard = true
notify_on_paste = false
notify_on_fallback = true
replace_strategy = "select"
replace_timeout_seconds = 5
replace_blacklist = ["Alacritty"]

[indicator]
style = "bar"
position = "top"
height = 4
width = "full"
color = "#ff4444"
opacity = 0.9

[history]
enabled = true
db_path = "~/.local/share/vox/history.db"
max_entries = 10000

[sounds]
enabled = true
start_sound = "assets/start.wav"
stop_sound = "assets/stop.wav"
volume = 0.7
""")
    return config_path


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Return a temporary SQLite DB path (auto-cleaned)."""
    return str(tmp_path / "test_history.db")


@pytest.fixture
def mock_subprocess():
    """Patch subprocess.run for xdotool/notification tests."""
    with patch("subprocess.run") as mock:
        mock.return_value.returncode = 0
        mock.return_value.stdout = ""
        mock.return_value.stderr = ""
        yield mock


def make_hotkey_settings(**kwargs: Any) -> HotkeySettings:
    defaults: dict[str, Any] = {
        "trigger": "f9",
        "mode": "toggle",
    }
    defaults.update(kwargs)
    return HotkeySettings(**defaults)


def make_audio_settings(**kwargs: Any) -> AudioSettings:
    defaults: dict[str, Any] = {
        "device": None,
        "sample_rate": 16000,
        "channels": 1,
    }
    defaults.update(kwargs)
    return AudioSettings(**defaults)


def make_transcription_settings(**kwargs: Any) -> TranscriptionSettings:
    defaults: dict[str, Any] = {
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "int8",
        "language": None,
        "vad": True,
    }
    defaults.update(kwargs)
    return TranscriptionSettings(**defaults)


def make_llm_settings(**kwargs: Any) -> LLMSettings:
    defaults: dict[str, Any] = {
        "enabled": False,
        "base_url": "http://localhost:11434/v1",
        "api_key": "",
        "model": "phi3-mini",
        "timeout_seconds": 10,
        "prompt": "Clean up the text.",
    }
    defaults.update(kwargs)
    return LLMSettings(**defaults)


def make_output_settings(**kwargs: Any) -> OutputSettings:
    defaults: dict[str, Any] = {
        "method": "xdotool",
        "fallback_to_clipboard": True,
        "notify_on_paste": False,
        "notify_on_fallback": True,
        "replace_strategy": "select",
        "replace_timeout_seconds": 5,
        "replace_blacklist": [],
    }
    defaults.update(kwargs)
    return OutputSettings(**defaults)


def make_indicator_settings(**kwargs: Any) -> IndicatorSettings:
    defaults: dict[str, Any] = {
        "style": "bar",
        "position": "top",
        "height": 4,
        "width": "full",
        "color": "#ff4444",
        "opacity": 0.9,
    }
    defaults.update(kwargs)
    return IndicatorSettings(**defaults)


def make_history_settings(**kwargs: Any) -> HistorySettings:
    defaults: dict[str, Any] = {
        "enabled": True,
        "db_path": ":memory:",
        "max_entries": 10000,
    }
    defaults.update(kwargs)
    return HistorySettings(**defaults)


def make_sounds_settings(**kwargs: Any) -> SoundsSettings:
    defaults: dict[str, Any] = {
        "enabled": True,
        "start_sound": "assets/start.wav",
        "stop_sound": "assets/stop.wav",
        "volume": 0.7,
    }
    defaults.update(kwargs)
    return SoundsSettings(**defaults)


def make_full_settings(**kwargs: Any) -> Settings:
    return Settings(
        hotkey=kwargs.get("hotkey", make_hotkey_settings()),
        audio=kwargs.get("audio", make_audio_settings()),
        transcription=kwargs.get("transcription", make_transcription_settings()),
        llm=kwargs.get("llm", make_llm_settings()),
        output=kwargs.get("output", make_output_settings()),
        indicator=kwargs.get("indicator", make_indicator_settings()),
        history=kwargs.get("history", make_history_settings()),
        sounds=kwargs.get("sounds", make_sounds_settings()),
    )
