"""Tests for hotkey — pynput key parsing and state machine."""

from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock

import pytest
from pynput import keyboard

from vox.config import HotkeySettings
from vox.hotkey import HotkeyListener, _parse_trigger


def _make_hotkey_settings(**kwargs: object) -> HotkeySettings:
    defaults: dict[str, object] = {
        "trigger": "f9",
        "mode": "toggle",
    }
    defaults.update(kwargs)
    return HotkeySettings(**defaults)  # type: ignore[arg-type]


class TestParseTrigger:
    def test_single_function_key(self) -> None:
        result = _parse_trigger("f9")
        assert result == {keyboard.Key.f9}

    def test_all_function_keys(self) -> None:
        for i in range(1, 21):
            result = _parse_trigger(f"f{i}")
            expected = getattr(keyboard.Key, f"f{i}")
            assert result == {expected}

    def test_modifier(self) -> None:
        assert _parse_trigger("ctrl") == {keyboard.Key.ctrl}
        assert _parse_trigger("alt") == {keyboard.Key.alt}
        assert _parse_trigger("shift") == {keyboard.Key.shift}

    def test_special_key(self) -> None:
        assert _parse_trigger("space") == {keyboard.Key.space}
        assert _parse_trigger("esc") == {keyboard.Key.esc}
        assert _parse_trigger("enter") == {keyboard.Key.enter}

    def test_single_char(self) -> None:
        result = _parse_trigger("a")
        assert result == {keyboard.KeyCode.from_char("a")}

    def test_combo_two_keys(self) -> None:
        result = _parse_trigger("ctrl+shift")
        assert result == {keyboard.Key.ctrl, keyboard.Key.shift}

    def test_combo_three_keys(self) -> None:
        result = _parse_trigger("ctrl+shift+space")
        assert result == {
            keyboard.Key.ctrl,
            keyboard.Key.shift,
            keyboard.Key.space,
        }

    def test_combo_with_function(self) -> None:
        result = _parse_trigger("ctrl+f9")
        assert result == {keyboard.Key.ctrl, keyboard.Key.f9}

    def test_case_insensitive(self) -> None:
        upper = _parse_trigger("F9")
        lower = _parse_trigger("f9")
        assert upper == lower

    def test_whitespace_handling(self) -> None:
        result = _parse_trigger("ctrl + f9")
        assert result == {keyboard.Key.ctrl, keyboard.Key.f9}

    def test_unknown_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_trigger("f21")
        assert "Unknown key" in caplog.text

    def test_all_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_trigger("zzz")


class TestHotkeyListenerToggle:
    def _make_listener(self, **kwargs: object) -> tuple[HotkeyListener, queue.Queue[str]]:
        q: queue.Queue[str] = queue.Queue()
        config = _make_hotkey_settings(mode="toggle", **kwargs)
        return HotkeyListener(config, q), q

    def test_first_press_starts(self) -> None:
        listener, q = self._make_listener()
        key = keyboard.Key.f9
        listener._on_press(key)
        assert q.get_nowait() == "start"
        assert listener._is_recording is True

    def test_second_press_stops(self) -> None:
        listener, q = self._make_listener()
        key = keyboard.Key.f9
        listener._on_press(key)
        q.get_nowait()
        listener._last_toggle_time = 0.0
        listener._on_press(key)
        assert q.get_nowait() == "stop"
        assert listener._is_recording is False

    def test_third_press_restarts(self) -> None:
        listener, q = self._make_listener()
        key = keyboard.Key.f9
        listener._on_press(key)
        q.get_nowait()
        listener._last_toggle_time = 0.0
        listener._on_press(key)
        q.get_nowait()
        listener._last_toggle_time = 0.0
        listener._on_press(key)
        assert q.get_nowait() == "start"
        assert listener._is_recording is True

    def test_debounce_ignores_rapid(self) -> None:
        listener, q = self._make_listener()
        key = keyboard.Key.f9
        listener._on_press(key)
        q.get_nowait()
        listener._last_toggle_time = time.monotonic()
        listener._on_press(key)
        with pytest.raises(queue.Empty):
            q.get_nowait()


class TestHotkeyListenerPushToTalk:
    def _make_listener(self, **kwargs: object) -> tuple[HotkeyListener, queue.Queue[str]]:
        q: queue.Queue[str] = queue.Queue()
        config = _make_hotkey_settings(mode="push_to_talk", **kwargs)
        return HotkeyListener(config, q), q

    def test_press_starts(self) -> None:
        listener, q = self._make_listener()
        key = keyboard.Key.f9
        listener._on_press(key)
        assert q.get_nowait() == "start"
        assert listener._is_recording is True

    def test_release_stops(self) -> None:
        listener, q = self._make_listener()
        key = keyboard.Key.f9
        listener._on_press(key)
        q.get_nowait()
        listener._on_release(key)
        assert q.get_nowait() == "stop"
        assert listener._is_recording is False

    def test_release_without_press_no_stop(self) -> None:
        listener, q = self._make_listener()
        key = keyboard.Key.f9
        listener._on_release(key)
        assert q.empty()


class TestHotkeyListenerEdgeCases:
    def test_none_key_on_press(self) -> None:
        listener, _ = self._make_listener()
        listener._on_press(None)
        assert listener._is_recording is False

    def test_none_key_on_release(self) -> None:
        listener, _ = self._make_listener()
        listener._on_release(None)

    def test_stop_listener(self) -> None:
        listener, _ = self._make_listener()
        mock_listener = MagicMock()
        listener._listener = mock_listener
        listener.stop()
        mock_listener.stop.assert_called_once()

    def test_stop_before_start(self) -> None:
        listener, _ = self._make_listener()
        listener.stop()

    def _make_listener(self, **kwargs: object) -> tuple[HotkeyListener, queue.Queue[str]]:
        q: queue.Queue[str] = queue.Queue()
        config = _make_hotkey_settings(**kwargs)
        return HotkeyListener(config, q), q
