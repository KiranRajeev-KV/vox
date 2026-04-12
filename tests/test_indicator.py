"""Tests for indicator — tkinter recording indicator."""

from __future__ import annotations

import queue
import subprocess
from unittest.mock import MagicMock, patch

from vox.config import IndicatorSettings
from vox.indicator import Indicator, _make_non_interactive


def _make_indicator_settings(**kwargs: object) -> IndicatorSettings:
    defaults: dict[str, object] = {
        "style": "bar",
        "position": "top",
        "height": 4,
        "width": "full",
        "color": "#ff4444",
        "opacity": 0.9,
    }
    defaults.update(kwargs)
    return IndicatorSettings(**defaults)  # type: ignore[arg-type]


class TestIndicatorInit:
    def test_available_when_tkinter_exists(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        assert indicator._available is True

    def test_unavailable_when_tkinter_none(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator._available = False
        assert indicator._available is False


class TestShowHide:
    def test_show_puts_in_queue(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator.show()
        assert indicator._queue.get_nowait() == "show"

    def test_hide_puts_in_queue(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator.hide()
        assert indicator._queue.get_nowait() == "hide"

    def test_show_noop_when_unavailable(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator._available = False
        indicator.show()
        assert indicator._queue.empty()

    def test_hide_noop_when_unavailable(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator._available = False
        indicator.hide()
        assert indicator._queue.empty()

    def test_queue_full_does_not_crash(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator._queue = queue.Queue(maxsize=1)
        indicator._queue.put_nowait("show")
        indicator.show()


class TestMakeNonInteractive:
    @patch("vox.indicator.subprocess.run")
    def test_calls_xprop(self, mock_run: MagicMock) -> None:
        _make_non_interactive("12345")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "xprop"
        assert "12345" in args
        assert "_NET_WM_WINDOW_TYPE_DOCK" in args

    @patch("vox.indicator.subprocess.run")
    def test_file_not_found_logged(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        _make_non_interactive("12345")

    @patch("vox.indicator.subprocess.run")
    def test_called_process_error_logged(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "xprop")
        _make_non_interactive("12345")


class TestStop:
    def test_stop_closes_root(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        mock_root = MagicMock()
        indicator._root = mock_root
        indicator.stop()
        mock_root.quit.assert_called_once()
        mock_root.destroy.assert_called_once()
        assert indicator._root is None

    def test_stop_before_run_no_crash(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator.stop()


class TestShowText:
    def test_show_text_puts_in_queue(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator.show_text("hello world")
        cmd = indicator._queue.get_nowait()
        assert cmd.startswith("show_text:")
        assert cmd.endswith("hello world")

    def test_show_text_truncates_long_text(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        long_text = "x" * 200
        indicator.show_text(long_text)
        cmd = indicator._queue.get_nowait()
        # Prefix "show_text:" + 120 chars + "..."
        assert len(cmd) == len("show_text:") + Indicator._MAX_TEXT_LENGTH + 3
        assert cmd.endswith("...")

    def test_show_text_noop_when_unavailable(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator._available = False
        indicator.show_text("hello")
        assert indicator._queue.empty()

    def test_show_text_queue_full_does_not_crash(self) -> None:
        config = _make_indicator_settings()
        indicator = Indicator(config)
        indicator._queue = queue.Queue(maxsize=1)
        indicator._queue.put_nowait("show")
        indicator.show_text("overflow")  # Should silently drop
