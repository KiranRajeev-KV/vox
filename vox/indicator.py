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

    def __init__(self, config: IndicatorSettings) -> None:
        self._config = config
        self._queue: queue.Queue[str] = queue.Queue()
        self._root: tk.Tk | None = None  # type: ignore[assignment]
        self._available = tk is not None

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
        """Hide the indicator bar. Safe to call from any thread."""
        if not self._available:
            return
        try:
            self._queue.put_nowait("hide")
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
                self._root.withdraw()
        assert self._root is not None
        self._root.after(50, self._process_queue)

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
