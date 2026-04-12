"""Load config.toml and expose typed settings."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass(frozen=True)
class HotkeySettings:
    trigger: str
    mode: str  # "toggle" or "push_to_talk"


@dataclass(frozen=True)
class AudioSettings:
    device: int | None
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class TranscriptionSettings:
    model: str
    device: str
    compute_type: str
    language: str | None
    vad: bool


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    prompt: str


@dataclass(frozen=True)
class OutputSettings:
    method: str
    fallback_to_clipboard: bool
    notify_on_paste: bool
    notify_on_fallback: bool
    replace_strategy: str  # "select", "skip", "append"
    replace_timeout_seconds: int
    replace_blacklist: list[str]


@dataclass(frozen=True)
class IndicatorSettings:
    style: str
    position: str
    height: int
    width: str | int  # "full" or pixel value
    color: str
    opacity: float


@dataclass(frozen=True)
class HistorySettings:
    enabled: bool
    db_path: str
    max_entries: int


@dataclass(frozen=True)
class DictionarySettings:
    enabled: bool
    replacements: dict[str, str]


@dataclass(frozen=True)
class SoundsSettings:
    enabled: bool
    start_sound: str
    stop_sound: str
    volume: float


@dataclass(frozen=True)
class Settings:
    hotkey: HotkeySettings
    audio: AudioSettings
    transcription: TranscriptionSettings
    llm: LLMSettings
    output: OutputSettings
    indicator: IndicatorSettings
    history: HistorySettings
    sounds: SoundsSettings
    dictionary: DictionarySettings


def _load(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _parse_indicator_width(raw: str | int) -> str | int:
    """Parse indicator width config value.

    Returns "full" (string) or a positive integer pixel value.
    Invalid values fall back to "full" with a warning.
    """
    if isinstance(raw, str):
        if raw.lower() == "full":
            return "full"
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    elif raw > 0:
        return raw

    import logging

    logging.getLogger(__name__).warning("Invalid indicator width '%s', falling back to 'full'", raw)
    return "full"


def _build_settings(raw: dict[str, Any]) -> Settings:
    hotkey_raw = raw.get("hotkey", {})
    audio_raw = raw.get("audio", {})
    transcription_raw = raw.get("transcription", {})
    llm_raw = raw.get("llm", {})
    output_raw = raw.get("output", {})
    indicator_raw = raw.get("indicator", {})
    history_raw = raw.get("history", {})
    sounds_raw = raw.get("sounds", {})
    dictionary_raw = raw.get("dictionary", {})

    api_key = llm_raw.get("api_key", "") or ""
    env_key = os.environ.get("VOX_LLM_API_KEY")
    if env_key:
        api_key = env_key

    db_path = history_raw.get("db_path", "~/.local/share/vox/history.db")
    db_path = str(Path(db_path).expanduser())

    return Settings(
        hotkey=HotkeySettings(
            trigger=hotkey_raw.get("trigger", "f9"),
            mode=hotkey_raw.get("mode", "toggle"),
        ),
        audio=AudioSettings(
            device=audio_raw.get("device") if isinstance(audio_raw.get("device"), int) else None,
            sample_rate=audio_raw.get("sample_rate", 16000),
            channels=audio_raw.get("channels", 1),
        ),
        transcription=TranscriptionSettings(
            model=transcription_raw.get("model", "large-v3"),
            device=transcription_raw.get("device", "cuda"),
            compute_type=transcription_raw.get("compute_type", "int8"),
            language=transcription_raw.get("language") or None,
            vad=transcription_raw.get("vad", True),
        ),
        llm=LLMSettings(
            enabled=llm_raw.get("enabled", True),
            base_url=llm_raw.get("base_url", "http://localhost:11434/v1"),
            api_key=api_key,
            model=llm_raw.get("model", "phi3-mini"),
            timeout_seconds=llm_raw.get("timeout_seconds", 10),
            prompt=llm_raw.get(
                "prompt",
                "Clean up the following transcribed speech. Fix grammar, punctuation, "
                "and sentence structure. Remove any remaining filler words. "
                "Keep the meaning and tone exactly the same. "
                "Return only the cleaned text, nothing else.",
            ),
        ),
        output=OutputSettings(
            method=output_raw.get("method", "xdotool"),
            fallback_to_clipboard=output_raw.get("fallback_to_clipboard", True),
            notify_on_paste=output_raw.get("notify_on_paste", False),
            notify_on_fallback=output_raw.get("notify_on_fallback", True),
            replace_strategy=output_raw.get("replace_strategy", "select"),
            replace_timeout_seconds=output_raw.get("replace_timeout_seconds", 5),
            replace_blacklist=output_raw.get("replace_blacklist") or [],
        ),
        indicator=IndicatorSettings(
            style=indicator_raw.get("style", "bar"),
            position=indicator_raw.get("position", "top"),
            height=indicator_raw.get("height", 4),
            width=_parse_indicator_width(indicator_raw.get("width", "full")),
            color=indicator_raw.get("color", "#ff4444"),
            opacity=indicator_raw.get("opacity", 0.9),
        ),
        history=HistorySettings(
            enabled=history_raw.get("enabled", True),
            db_path=db_path,
            max_entries=history_raw.get("max_entries", 10000),
        ),
        sounds=SoundsSettings(
            enabled=sounds_raw.get("enabled", True),
            start_sound=sounds_raw.get("start_sound", "assets/start.wav"),
            stop_sound=sounds_raw.get("stop_sound", "assets/stop.wav"),
            volume=sounds_raw.get("volume", 0.7),
        ),
        dictionary=DictionarySettings(
            enabled=dictionary_raw.get("enabled", True),
            replacements=dictionary_raw.get("replacements") or {},
        ),
    )


_settings: Settings | None = None


def get_settings(path: Path | None = None) -> Settings:
    global _settings
    if _settings is None:
        raw = _load(path)
        _settings = _build_settings(raw)
    return _settings
