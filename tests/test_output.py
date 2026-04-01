"""Tests for output — xdotool paste, clipboard fallback, replace strategy."""

from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from vox.config import OutputSettings
from vox.output import OutputBackend, OutputError, Outputter, XdotoolBackend


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


class TestXdotoolBackend:
    @patch("vox.output.subprocess.run")
    def test_type_text_success(self, mock_run: MagicMock) -> None:
        backend = XdotoolBackend()
        result = backend.type_text("hello")
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "xdotool"
        assert args[1] == "type"
        assert "--" in args

    @patch("vox.output.subprocess.run")
    def test_type_text_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "xdotool")
        backend = XdotoolBackend()
        result = backend.type_text("hello")
        assert result is False

    @patch("vox.output.subprocess.run")
    def test_type_text_file_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        backend = XdotoolBackend()
        result = backend.type_text("hello")
        assert result is False

    @patch("vox.output.subprocess.run")
    def test_type_text_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired("xdotool", 10)
        backend = XdotoolBackend()
        result = backend.type_text("hello")
        assert result is False

    @patch("vox.output.subprocess.run")
    def test_type_text_with_dash_prefix(self, mock_run: MagicMock) -> None:
        backend = XdotoolBackend()
        backend.type_text("-test")
        args = mock_run.call_args[0][0]
        assert "--" in args
        assert "-test" in args

    @patch("vox.output.subprocess.run")
    def test_select_left_success(self, mock_run: MagicMock) -> None:
        backend = XdotoolBackend()
        result = backend.select_left(5)
        assert result is True
        args = mock_run.call_args[0][0]
        assert args[0] == "xdotool"
        assert args[1] == "key"
        assert "shift+Left" in args

    @patch("vox.output.subprocess.run")
    def test_select_left_zero(self, mock_run: MagicMock) -> None:
        backend = XdotoolBackend()
        result = backend.select_left(0)
        assert result is True
        mock_run.assert_not_called()

    @patch("vox.output.subprocess.run")
    def test_select_left_negative(self, mock_run: MagicMock) -> None:
        backend = XdotoolBackend()
        result = backend.select_left(-1)
        assert result is True
        mock_run.assert_not_called()

    @patch("vox.output.subprocess.run")
    def test_select_left_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "xdotool")
        backend = XdotoolBackend()
        result = backend.select_left(5)
        assert result is False

    @patch("vox.output.subprocess.run")
    def test_get_active_window_class(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="firefox\n")
        backend = XdotoolBackend()
        result = backend.get_active_window_class()
        assert result == "firefox"

    @patch("vox.output.subprocess.run")
    def test_get_active_window_class_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        backend = XdotoolBackend()
        result = backend.get_active_window_class()
        assert result == ""


class TestOutputterPaste:
    @patch("vox.output.pyperclip")
    @patch("vox.output.subprocess.run")
    def test_paste_success(self, mock_run: MagicMock, mock_clip: MagicMock) -> None:
        config = _make_output_settings()
        backend = _MockBackend()
        outputter = Outputter(config, backend=backend)
        result = outputter.paste("hello world")
        assert result == 11
        mock_clip.copy.assert_called()
        mock_run.assert_called_once()

    @patch("vox.output._notify")
    @patch("vox.output.pyperclip")
    @patch("vox.output.subprocess.run")
    def test_paste_with_notify(
        self, mock_run: MagicMock, mock_clip: MagicMock, mock_notify: MagicMock
    ) -> None:
        config = _make_output_settings(notify_on_paste=True)
        backend = _MockBackend()
        outputter = Outputter(config, backend=backend)
        outputter.paste("text")
        mock_notify.assert_called_once_with("Vox", "Text pasted")

    @patch("vox.output._notify")
    @patch("vox.output.pyperclip")
    @patch("vox.output.subprocess.run")
    def test_paste_without_notify(
        self, mock_run: MagicMock, mock_clip: MagicMock, mock_notify: MagicMock
    ) -> None:
        config = _make_output_settings(notify_on_paste=False)
        backend = _MockBackend()
        outputter = Outputter(config, backend=backend)
        outputter.paste("text")
        mock_notify.assert_not_called()

    @patch("vox.output.pyperclip")
    @patch("vox.output.subprocess.run")
    def test_paste_xdotool_fallback_clipboard(
        self, mock_run: MagicMock, mock_clip: MagicMock
    ) -> None:
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "xdotool"),
            MagicMock(),
        ]
        config = _make_output_settings(fallback_to_clipboard=True)
        backend = _MockBackend()
        outputter = Outputter(config, backend=backend)
        result = outputter.paste("text")
        assert result == 4
        assert mock_clip.copy.call_count == 2

    @patch("vox.output._notify")
    @patch("vox.output.pyperclip")
    @patch("vox.output.subprocess.run")
    def test_paste_fallback_notify(
        self, mock_run: MagicMock, mock_clip: MagicMock, mock_notify: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "xdotool")
        config = _make_output_settings(fallback_to_clipboard=True, notify_on_fallback=True)
        backend = _MockBackend()
        outputter = Outputter(config, backend=backend)
        outputter.paste("text")
        mock_notify.assert_called_once_with("Vox", "Text copied to clipboard")

    @patch("vox.output.pyperclip")
    @patch("vox.output.subprocess.run")
    def test_paste_no_fallback_raises(self, mock_run: MagicMock, mock_clip: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "xdotool")
        config = _make_output_settings(fallback_to_clipboard=False)
        backend = _MockBackend()
        outputter = Outputter(config, backend=backend)
        with pytest.raises(OutputError):
            outputter.paste("text")


class TestOutputterReplace:
    def _make_outputter(
        self, backend: _MockBackend | None = None, **kwargs: object
    ) -> tuple[Outputter, _MockBackend]:
        mock = backend or _MockBackend()
        config = _make_output_settings(**kwargs)
        return Outputter(config, backend=mock), mock

    def test_replace_select_success(self) -> None:
        outputter, backend = self._make_outputter(replace_strategy="select")
        backend._window_class = "same-app"
        outputter.replace("cleaned", 7, "same-app", time.monotonic())
        assert ("select_left", (7,)) in backend.calls
        assert ("type_text", ("cleaned",)) in backend.calls

    def test_replace_select_fallback_to_append(self) -> None:
        outputter, backend = self._make_outputter(replace_strategy="select")
        backend._window_class = "same-app"
        backend._select_left_result = False
        outputter.replace("cleaned", 7, "same-app", time.monotonic())
        assert ("type_text", ("\ncleaned",)) in backend.calls

    def test_replace_append(self) -> None:
        outputter, backend = self._make_outputter(replace_strategy="append")
        backend._window_class = "same-app"
        outputter.replace("cleaned", 7, "same-app", time.monotonic())
        assert ("type_text", ("\ncleaned",)) in backend.calls

    def test_replace_skip(self) -> None:
        outputter, backend = self._make_outputter(replace_strategy="skip")
        backend._window_class = "same-app"
        outputter.replace("cleaned", 7, "same-app", time.monotonic())
        assert len(backend.calls) == 1
        assert backend.calls[0][0] == "get_active_window_class"

    @patch("vox.output.pyperclip")
    def test_replace_stale_timeout(self, mock_clip: MagicMock) -> None:
        outputter, backend = self._make_outputter(replace_strategy="select")
        backend._window_class = "same-app"
        stale_time = time.monotonic() - 10
        outputter.replace("cleaned", 7, "same-app", stale_time)
        mock_clip.copy.assert_called_once_with("cleaned")
        assert ("type_text", ("cleaned",)) not in backend.calls

    @patch("vox.output.pyperclip")
    def test_replace_window_changed(self, mock_clip: MagicMock) -> None:
        outputter, backend = self._make_outputter(replace_strategy="select")
        backend._window_class = "different-app"
        outputter.replace("cleaned", 7, "original-app", time.monotonic())
        mock_clip.copy.assert_called_once_with("cleaned")
        assert ("type_text", ("cleaned",)) not in backend.calls

    @patch("vox.output.pyperclip")
    def test_replace_blacklisted_window(self, mock_clip: MagicMock) -> None:
        outputter, backend = self._make_outputter(
            replace_strategy="select", replace_blacklist=["Vim"]
        )
        backend._window_class = "Vim"
        outputter.replace("cleaned", 7, "Vim", time.monotonic())
        mock_clip.copy.assert_called_once_with("cleaned")
        assert ("type_text", ("cleaned",)) not in backend.calls

    def test_replace_unknown_strategy(self, caplog: pytest.LogCaptureFixture) -> None:
        outputter, backend = self._make_outputter(replace_strategy="nonexistent")
        backend._window_class = "same-app"
        outputter.replace("cleaned", 7, "same-app", time.monotonic())
        assert "Unknown replace strategy" in caplog.text


class TestOutputterWindowClass:
    def test_get_active_window_class_delegates(self) -> None:
        backend = _MockBackend()
        backend._window_class = "my-app"
        config = _make_output_settings()
        outputter = Outputter(config, backend=backend)
        assert outputter.get_active_window_class() == "my-app"
