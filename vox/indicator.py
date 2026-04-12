"""Recording indicator — thin bar overlay via tkinter."""

from __future__ import annotations

import logging
import queue
import subprocess

from vox.config import IndicatorSettings

logger = logging.getLogger(__name__)


def _make_non_interactive(window_id: str) -> None:
    """Set X11 window type to DOCK so clicks pass through."""
    try:
        subprocess.run(
            [
                "xprop",
                "-id",
                window_id,
                "-f",
                "_NET_WM_WINDOW_TYPE",
                "32a",
                "-set",
                "_NET_WM_WINDOW_TYPE",
                "_NET_WM_WINDOW_TYPE_DOCK",
            ],
            capture_output=True,
            check=True,
        )
        logger.debug("Set indicator window to DOCK type (non-interactive)")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.debug("xprop not available or failed; indicator may be clickable")


try:
    import tkinter as tk
except ImportError:
    tk = None  # type: ignore[assignment]


class Indicator:
    """Thin bar overlay shown during recording.

    Thread-safe: show() and hide() may be called from any thread.
    Actual tkinter operations run in the main loop via a command queue.
    """

    _MAX_TEXT_LENGTH = 120  # Truncate long partial transcriptions
    _TEXT_HEIGHT = 40  # Height in pixels when showing streaming text

    def __init__(self, config: IndicatorSettings) -> None:
        self._config = config
        self._queue: queue.Queue[str] = queue.Queue()
        self._root: tk.Tk | None = None  # type: ignore[assignment]
        self._label: tk.Label | None = None  # type: ignore[assignment]
        self._available = tk is not None
        self._is_expanded = False  # Track if indicator is expanded for text

        if not self._available:
            logger.warning("tkinter not available; indicator disabled")

    def show(self) -> None:
        """Show the indicator bar. Safe to call from any thread."""
        if not self._available:
            return
        try:
            self._queue.put_nowait("show")
        except queue.Full:
            pass

    def hide(self) -> None:
        """Hide the indicator bar and clear any partial text."""
        if not self._available:
            return
        try:
            self._queue.put_nowait("hide")
        except queue.Full:
            pass

    def show_text(self, text: str) -> None:
        """Show the indicator bar with partial transcription text.

        The text is displayed as an overlay on the bar. If the bar
        is not yet visible, it is shown automatically.

        Args:
            text: Partial transcription text to display.
        """
        if not self._available:
            return
        try:
            truncated = text[: self._MAX_TEXT_LENGTH]
            if len(text) > self._MAX_TEXT_LENGTH:
                truncated += "..."
            self._queue.put_nowait(f"show_text:{truncated}")
        except queue.Full:
            pass

    def _process_queue(self) -> None:
        """Process pending commands from the queue."""
        while True:
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                break
            if cmd == "show":
                assert self._root is not None
                self._root.deiconify()
            elif cmd == "hide":
                assert self._root is not None
                self._is_expanded = False
                # Clear text label on hide
                if self._label is not None:
                    self._label.config(text="")
                # Restore original geometry
                screen_width = self._root.winfo_screenwidth()
                if self._config.width == "full":
                    bar_width = screen_width
                    x = 0
                else:
                    bar_width = int(self._config.width)
                    x = (screen_width - bar_width) // 2
                y = (
                    0
                    if self._config.position == "top"
                    else self._root.winfo_screenheight() - self._config.height
                )
                self._root.geometry(f"{bar_width}x{self._config.height}+{x}+{y}")
                self._root.withdraw()
            elif cmd.startswith("show_text:"):
                assert self._root is not None
                text = cmd[len("show_text:") :]
                self._show_text_impl(text)
        assert self._root is not None
        self._root.after(50, self._process_queue)

    def _show_text_impl(self, text: str) -> None:
        """Show the indicator bar with text overlay.

        Expands the bar to _TEXT_HEIGHT so text is readable.
        """
        assert self._root is not None
        # Show the bar if it's hidden
        if not self._root.winfo_viewable():
            self._root.deiconify()

        # Expand the bar if not already expanded
        if not self._is_expanded:
            self._is_expanded = True
            screen_width = self._root.winfo_screenwidth()
            if self._config.width == "full":
                bar_width = screen_width
                x = 0
            else:
                bar_width = int(self._config.width)
                x = (screen_width - bar_width) // 2
            y = (
                0
                if self._config.position == "top"
                else self._root.winfo_screenheight() - self._TEXT_HEIGHT
            )
            self._root.geometry(f"{bar_width}x{self._TEXT_HEIGHT}+{x}+{y}")

        if self._label is None:
            assert tk is not None
            self._label = tk.Label(  # type: ignore[call-arg, arg-type, reportUnknownArgumentType]
                self._root,  # type: ignore[arg-type]
                text=text,
                fg="white",
                bg=self._config.color,
                font=("sans-serif", 12),
                anchor="w",
                padx=8,
            )
            self._label.place(relx=0, rely=0, relwidth=1, relheight=1)  # type: ignore[union-attr]
        else:
            self._label.config(text=text)

    def run(self) -> None:
        """Create the indicator window and enter the main loop."""
        if not self._available:
            return

        try:
            assert tk is not None
            root = tk.Tk()
            self._root = root
            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", self._config.opacity)

            screen_width = root.winfo_screenwidth()
            if self._config.width == "full":
                bar_width = screen_width
                x = 0
            else:
                bar_width = int(self._config.width)
                x = (screen_width - bar_width) // 2
            y = (
                0
                if self._config.position == "top"
                else root.winfo_screenheight() - self._config.height
            )

            root.geometry(f"{bar_width}x{self._config.height}+{x}+{y}")
            root.configure(bg=self._config.color)

            root.update_idletasks()
            window_id = root.winfo_id()
            _make_non_interactive(str(window_id))

            root.after(50, self._process_queue)
            logger.info("Indicator window created")
            root.mainloop()
        except Exception:
            logger.exception("Failed to create indicator window")
            self._available = False

    def stop(self) -> None:
        """Shut down the indicator window."""
        if self._root is not None:
            self._root.quit()
            self._root.destroy()
            self._root = None
            logger.info("Indicator window closed")
