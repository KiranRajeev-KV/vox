"""Text processing — LLM cleanup."""

from __future__ import annotations

import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from vox.config import LLMSettings

logger = logging.getLogger(__name__)


class LLMCleaner:
    """Grammar cleanup via any OpenAI-compatible endpoint.

    Initializes the OpenAI client lazily on first clean() call so that
    an unreachable LLM endpoint does not block startup.
    """

    def __init__(self, config: LLMSettings) -> None:
        self._config = config
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key or "none",
                max_retries=0,
            )
        return self._client

    def clean(self, text: str) -> str | None:
        """Send text to the LLM for grammar cleanup.

        Args:
            text: The transcribed text.

        Returns:
            Cleaned text from the LLM, or None if the call failed.
        """
        if not self._config.enabled:
            return None

        if not text.strip():
            return None

        try:
            client = self._get_client()
            is_local = "localhost" in self._config.base_url
            extra_body = {"keep_alive": 0} if is_local else None

            prompt = self._config.prompt + "\n\n" + text

            response = client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._config.timeout_seconds,
                extra_body=extra_body,
            )

            content = response.choices[0].message.content
            if content:
                return content.strip()

            logger.warning("LLM returned empty response")
            return None

        except APITimeoutError:
            logger.error("LLM timeout after %ds", self._config.timeout_seconds)
            return None
        except APIConnectionError:
            logger.error("LLM connection failed: %s", self._config.base_url)
            return None
        except APIStatusError as exc:
            logger.error("LLM API error %s: %s", exc.status_code, exc.message)
            return None
        except Exception:
            logger.exception("Unexpected LLM error")
            return None
