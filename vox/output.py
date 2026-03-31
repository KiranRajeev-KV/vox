"""Output backend — xdotool paste, clipboard fallback, replace strategy."""

from __future__ import annotations

import logging
import subprocess
import time

import pyperclip

from vox.config import OutputSettings

logger = logging.getLogger(__name__)


class OutputError(Exception):
    """Raised when both xdotool and clipboard fallback fail."""


class OutputBackend:
    """Abstract interface for text output into the active window."""

    def type_text(self, text: str) -> bool:
        """Type text into the active window.

        Returns True on success, False on failure.
        """
        raise NotImplementedError

    def select_left(self, n_chars: int) -> bool:
        """Select n_chars to the left of the cursor.

        Returns True on success, False on failure.
        """
        raise NotImplementedError

    def get_active_window_class(self) -> str:
        """Return the WM_CLASS of the currently active window."""
        raise NotImplementedError


class XdotoolBackend(OutputBackend):
    """X11 implementation using xdotool subprocess calls."""

    def type_text(self, text: str) -> bool:
        try:
            subprocess.run(
                ["xdotool", "type", "--delay", "0", "--clearmodifiers", "--", text],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.exception("xdotool type failed")
            return False

    def select_left(self, n_chars: int) -> bool:
        if n_chars <= 0:
            return True
        try:
            keys = ["shift+Left"] * n_chars
            subprocess.run(
                ["xdotool", "key", "--delay", "3", *keys],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.exception("xdotool select_left failed")
            return False

    def get_active_window_class(self) -> str:
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowclassname"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.exception("xdotool get_active_window_class failed")
            return ""


def _notify(title: str, body: str) -> None:
    """Send a desktop notification via notify-send."""
    try:
        subprocess.run(
            ["notify-send", title, body],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.debug("notify-send not available")


class Outputter:
    """High-level paste and replace strategy orchestrator.

    Wraps an OutputBackend with clipboard fallback, notify-send,
    and replace strategy routing.
    """

    def __init__(
        self,
        config: OutputSettings,
        backend: OutputBackend | None = None,
    ) -> None:
        self._config = config
        self._backend = backend or XdotoolBackend()

    def get_active_window_class(self) -> str:
        """Return the WM_CLASS of the currently active window."""
        return self._backend.get_active_window_class()

    def paste(self, text: str) -> int:
        """Paste text into the active window.

        Uses pyperclip.copy() + xdotool ctrl+v for speed and reliability.
        Falls back to pyperclip.copy() only if xdotool fails.

        Returns the number of characters pasted.
        """
        pyperclip.copy(text)
        time.sleep(0.1)

        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            if self._config.notify_on_paste:
                _notify("Vox", "Text pasted")
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("xdotool paste failed, falling back to clipboard")
            if self._config.fallback_to_clipboard:
                pyperclip.copy(text)
                if self._config.notify_on_fallback:
                    _notify("Vox", "Text copied to clipboard")
            else:
                msg = "Paste failed: xdotool unavailable and clipboard fallback disabled"
                raise OutputError(msg) from None

        return len(text)

    def replace(
        self,
        clean_text: str,
        raw_len: int,
        original_window: str,
        paste_time: float,
    ) -> None:
        """Replace previously pasted text with the LLM-cleaned version.

        Args:
            clean_text: The cleaned text from the LLM.
            raw_len: Character count of the originally pasted raw text.
            original_window: WM_CLASS of the window where raw text was pasted.
            paste_time: time.monotonic() value when the raw text was pasted.
        """
        elapsed = time.monotonic() - paste_time
        if elapsed > self._config.replace_timeout_seconds:
            pyperclip.copy(clean_text)
            logger.debug(
                "Replace skipped: stale (%.1fs elapsed), clean text copied to clipboard",
                elapsed,
            )
            return

        current_window = self._backend.get_active_window_class()

        if current_window != original_window:
            pyperclip.copy(clean_text)
            logger.debug(
                "Replace skipped: window changed (was '%s', now '%s'), "
                "clean text copied to clipboard",
                original_window,
                current_window,
            )
            return

        if current_window in self._config.replace_blacklist:
            pyperclip.copy(clean_text)
            logger.debug(
                "Replace skipped: window '%s' is blacklisted, clean text copied to clipboard",
                current_window,
            )
            return

        strategy = self._config.replace_strategy

        if strategy == "skip":
            logger.debug("Replace skipped: strategy is 'skip'")
            return

        if strategy == "append":
            self._backend.type_text("\n" + clean_text)
            logger.info("Replace: appended cleaned text")
            return

        if strategy == "select":
            if self._backend.select_left(raw_len):
                self._backend.type_text(clean_text)
                logger.info("Replace: selected %d chars and overwrote with cleaned text", raw_len)
            else:
                logger.warning("Replace: select_left failed, falling back to append")
                self._backend.type_text("\n" + clean_text)
            return

        logger.warning("Unknown replace strategy: '%s'", strategy)
