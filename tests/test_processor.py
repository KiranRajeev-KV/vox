"""Tests for processor — LLM cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openai import APIConnectionError, APIStatusError, APITimeoutError

from vox.config import LLMSettings
from vox.processor import LLMCleaner


def _make_llm_settings(**kwargs: object) -> LLMSettings:
    defaults: dict[str, object] = {
        "enabled": True,
        "base_url": "http://localhost:11434/v1",
        "api_key": "",
        "model": "phi3-mini",
        "timeout_seconds": 10,
        "prompt": "Clean up the text.",
    }
    defaults.update(kwargs)
    return LLMSettings(**defaults)  # type: ignore[arg-type]


class TestLLMCleanerInit:
    def test_stores_config(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        assert cleaner._config is config
        assert cleaner._client is None


class TestLLMCleanerClean:
    def test_returns_cleaned_text(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Cleaned text."
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get.return_value = mock_client
            result = cleaner.clean("raw text")
        assert result == "Cleaned text."

    def test_returns_none_when_disabled(self) -> None:
        config = _make_llm_settings(enabled=False)
        cleaner = LLMCleaner(config)
        assert cleaner.clean("text") is None

    def test_returns_none_for_empty_input(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        assert cleaner.clean("") is None
        assert cleaner.clean("   ") is None

    def test_returns_none_for_empty_llm_response(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get.return_value = mock_client
            result = cleaner.clean("text")
        assert result is None

    def test_returns_none_on_timeout(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
            mock_get.return_value = mock_client
            result = cleaner.clean("text")
        assert result is None

    def test_returns_none_on_connection_error(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            mock_get.return_value = mock_client
            result = cleaner.clean("text")
        assert result is None

    def test_returns_none_on_api_status_error(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APIStatusError(
                "error", response=MagicMock(), body=None
            )
            mock_get.return_value = mock_client
            result = cleaner.clean("text")
        assert result is None

    def test_returns_none_on_unexpected_error(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = ValueError("unexpected")
            mock_get.return_value = mock_client
            result = cleaner.clean("text")
        assert result is None

    def test_uses_localhost_keep_alive(self) -> None:
        config = _make_llm_settings(base_url="http://localhost:11434/v1")
        cleaner = LLMCleaner(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "cleaned"
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get.return_value = mock_client
            cleaner.clean("text")
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["extra_body"] == {"keep_alive": 0}

    def test_no_keep_alive_for_cloud(self) -> None:
        config = _make_llm_settings(base_url="https://api.openai.com/v1")
        cleaner = LLMCleaner(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "cleaned"
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get.return_value = mock_client
            cleaner.clean("text")
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["extra_body"] is None

    def test_prompt_concatenation(self) -> None:
        config = _make_llm_settings(prompt="Fix grammar.")
        cleaner = LLMCleaner(config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "done"
        with patch.object(cleaner, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get.return_value = mock_client
            cleaner.clean("hello world")
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["messages"][0]["content"] == "Fix grammar.\n\nhello world"

    def test_client_lazy_init(self) -> None:
        config = _make_llm_settings()
        cleaner = LLMCleaner(config)
        assert cleaner._client is None
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "cleaned"
        with patch("vox.processor.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            cleaner.clean("text")
            assert cleaner._client is mock_client
            mock_openai.assert_called_once()
            cleaner.clean("text2")
            assert mock_openai.call_count == 1
