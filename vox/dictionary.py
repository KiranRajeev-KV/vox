"""Post-transcription vocabulary correction."""

from __future__ import annotations

import logging
import re

from vox.config import DictionarySettings

logger = logging.getLogger(__name__)


class VocabularyCorrector:
    """Applies user-defined word mappings to transcribed text.

    All replacements are case-insensitive substring replacements applied
    in a single pass. Keys are sorted longest-first so that longer patterns
    (e.g. "voice text engine") match before shorter overlapping ones (e.g.
    "voice").
    """

    def __init__(self, config: DictionarySettings) -> None:
        self._enabled = config.enabled
        self._replacements: dict[str, str] = config.replacements

        # Compile all replacements into a single regex for one-pass efficiency.
        sorted_keys = sorted(config.replacements.keys(), key=len, reverse=True)
        self._pattern: re.Pattern[str] | None = None

        if sorted_keys:
            escaped = [re.escape(k) for k in sorted_keys]
            self._pattern = re.compile("|".join(escaped), re.IGNORECASE)
            # Build a lowercase lookup map since the regex is case-insensitive
            # but matched text preserves original casing.
            self._key_map: dict[str, str] = {k.lower(): k for k in sorted_keys}

    def correct(self, text: str) -> str:
        """Apply vocabulary corrections to text.

        Args:
            text: The raw transcribed text.

        Returns:
            Text with all dictionary corrections applied.
            Returns the original text unchanged if disabled or no mappings exist.
        """
        if not self._enabled or not self._pattern or not text.strip():
            return text

        result = self._pattern.sub(
            lambda m: self._replacements[self._key_map[m.group(0).lower()]],
            text,
        )

        if result != text:
            logger.info("Dictionary corrected: %r -> %r", text, result)

        return result
