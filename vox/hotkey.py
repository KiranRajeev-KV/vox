"""Global hotkey listener using pynput."""

from __future__ import annotations

import logging
import queue
import threading
import time

from pynput import keyboard

from vox.config import HotkeySettings

logger = logging.getLogger(__name__)

_MODIFIER_MAP: dict[str, keyboard.Key] = {
    "ctrl": keyboard.Key.ctrl,
    "alt": keyboard.Key.alt,
    "shift": keyboard.Key.shift,
    "cmd": keyboard.Key.cmd,
    "super": keyboard.Key.cmd,
    "win": keyboard.Key.cmd,
}

_FUNCTION_MAP: dict[str, keyboard.Key | None] = {
    f"f{i}": getattr(keyboard.Key, f"f{i}", None) for i in range(1, 21)
}

_SPECIAL_MAP: dict[str, keyboard.Key] = {
    "space": keyboard.Key.space,
    "esc": keyboard.Key.esc,
    "enter": keyboard.Key.enter,
    "return": keyboard.Key.enter,
    "tab": keyboard.Key.tab,
    "backspace": keyboard.Key.backspace,
    "delete": keyboard.Key.delete,
    "insert": keyboard.Key.insert,
    "home": keyboard.Key.home,
    "end": keyboard.Key.end,
    "page_up": keyboard.Key.page_up,
    "page_down": keyboard.Key.page_down,
    "up": keyboard.Key.up,
    "down": keyboard.Key.down,
    "left": keyboard.Key.left,
    "right": keyboard.Key.right,
    "pause": keyboard.Key.pause,
    "caps_lock": keyboard.Key.caps_lock,
    "num_lock": keyboard.Key.num_lock,
    "scroll_lock": keyboard.Key.scroll_lock,
    "print_screen": keyboard.Key.print_screen,
}


def _parse_trigger(
    trigger: str,
) -> set[keyboard.Key | keyboard.KeyCode]:
    """Parse a trigger string into a set of pynput keys.

    Supports single keys (e.g. "f9") and combinations
    (e.g. "ctrl+shift+space").
    """
    parts = trigger.lower().split("+")
    keys: set[keyboard.Key | keyboard.KeyCode] = set()

    for part in parts:
        part = part.strip()
        if part in _MODIFIER_MAP:
            keys.add(_MODIFIER_MAP[part])
        elif part in _FUNCTION_MAP:
            key = _FUNCTION_MAP[part]
            if key is not None:
                keys.add(key)
        elif part in _SPECIAL_MAP:
            keys.add(_SPECIAL_MAP[part])
        elif len(part) == 1:
            keys.add(keyboard.KeyCode.from_char(part))
        else:
            logger.warning("Unknown key in trigger '%s': '%s'", trigger, part)

    if not keys:
        raise ValueError(f"Could not parse trigger: '{trigger}'")

    return keys


class HotkeyListener:
    """Listen for global hotkey and send start/stop to control_queue.

    Supports two modes:
    - "toggle": press trigger to start, press again to stop.
    - "push_to_talk": hold trigger to record, release to stop.
    """

    def __init__(
        self,
        config: HotkeySettings,
        control_queue: queue.Queue[str],
    ) -> None:
        self._config = config
        self._control_queue = control_queue
        self._keys = _parse_trigger(config.trigger)
        self._pressed: set[keyboard.Key | keyboard.KeyCode] = set()
        self._is_recording = False
        self._listener: keyboard.Listener | None = None
        self._lock = threading.Lock()
        self._last_toggle_time = 0.0

        mode_label = "push-to-talk" if config.mode == "push_to_talk" else "toggle"
        logger.info(
            "Hotkey configured: %s mode, trigger=%s",
            mode_label,
            config.trigger,
        )

    def _all_pressed(self) -> bool:
        return self._keys.issubset(self._pressed)

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        with self._lock:
            if key is not None:
                self._pressed.add(key)
            if not self._all_pressed():
                return

            # Debounce: ignore toggles within 200ms of last event
            now = time.monotonic()
            if now - self._last_toggle_time < 0.2:
                return
            self._last_toggle_time = now

            if self._config.mode == "toggle":
                if self._is_recording:
                    self._is_recording = False
                    self._control_queue.put("stop")
                    logger.debug("Hotkey: toggle stop")
                else:
                    self._is_recording = True
                    self._control_queue.put("start")
                    logger.debug("Hotkey: toggle start")
            else:
                if not self._is_recording:
                    self._is_recording = True
                    self._control_queue.put("start")
                    logger.debug("Hotkey: push-to-talk start")

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        with self._lock:
            if key is not None:
                self._pressed.discard(key)

            if self._config.mode == "push_to_talk":
                if self._is_recording and not self._all_pressed():
                    self._is_recording = False
                    self._control_queue.put("stop")
                    logger.debug("Hotkey: push-to-talk stop")

    def run(self) -> None:
        """Start the listener and block until stopped."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        logger.info("Hotkey listener started, waiting for %s", self._config.trigger)
        self._listener.start()
        self._listener.wait()
        self._listener.join()
        logger.info("Hotkey listener stopped")

    def stop(self) -> None:
        """Stop the listener."""
        if self._listener is not None:
            self._listener.stop()
