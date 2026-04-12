"""Tests for dictionary — vocabulary correction."""

from __future__ import annotations

import logging

import pytest

from vox.config import DictionarySettings
from vox.dictionary import VocabularyCorrector


def _make_dict_settings(**kwargs: object) -> DictionarySettings:
    defaults: dict[str, object] = {
        "enabled": True,
        "replacements": {},
    }
    defaults.update(kwargs)
    return DictionarySettings(**defaults)  # type: ignore[arg-type]


class TestVocabularyCorrectorInit:
    def test_stores_config_enabled(self) -> None:
        replacements = {"foo": "bar"}
        config = _make_dict_settings(replacements=replacements)
        corrector = VocabularyCorrector(config)
        assert corrector._enabled is True
        assert corrector._replacements == replacements

    def test_stores_config_disabled(self) -> None:
        replacements = {"foo": "bar"}
        config = _make_dict_settings(enabled=False, replacements=replacements)
        corrector = VocabularyCorrector(config)
        assert corrector._enabled is False

    def test_pattern_compiled_with_replacements(self) -> None:
        config = _make_dict_settings(replacements={"foo": "bar"})
        corrector = VocabularyCorrector(config)
        assert corrector._pattern is not None

    def test_pattern_none_with_empty_replacements(self) -> None:
        config = _make_dict_settings(replacements={})
        corrector = VocabularyCorrector(config)
        assert corrector._pattern is None


class TestVocabularyCorrectorCorrect:
    def test_basic_replacement(self) -> None:
        config = _make_dict_settings(replacements={"pie torch": "PyTorch"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("I use pie torch for ML") == "I use PyTorch for ML"

    def test_case_insensitive_match(self) -> None:
        config = _make_dict_settings(replacements={"pie torch": "PyTorch"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("I use Pie Torch for ML") == "I use PyTorch for ML"

    def test_case_insensitive_all_caps(self) -> None:
        config = _make_dict_settings(replacements={"pie torch": "PyTorch"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("I use PIE TORCH for ML") == "I use PyTorch for ML"

    def test_longest_match_first(self) -> None:
        """Longer keys should match before shorter ones to prevent partial consumption."""
        replacements = {"voice text engine": "Vox", "voice": "Voice"}
        config = _make_dict_settings(replacements=replacements)
        corrector = VocabularyCorrector(config)
        assert corrector.correct("voice text engine is great") == "Vox is great"

    def test_disabled_returns_unchanged(self) -> None:
        config = _make_dict_settings(
            enabled=False,
            replacements={"foo": "bar"},
        )
        corrector = VocabularyCorrector(config)
        assert corrector.correct("foo bar") == "foo bar"

    def test_empty_replacements_returns_unchanged(self) -> None:
        config = _make_dict_settings(replacements={})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("hello world") == "hello world"

    def test_empty_text_returns_unchanged(self) -> None:
        config = _make_dict_settings(replacements={"foo": "bar"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("") == ""

    def test_whitespace_only_returns_unchanged(self) -> None:
        config = _make_dict_settings(replacements={"foo": "bar"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("   ") == "   "

    def test_multiple_replacements_in_one_text(self) -> None:
        replacements = {"co pilot": "Copilot", "pie torch": "PyTorch"}
        config = _make_dict_settings(replacements=replacements)
        corrector = VocabularyCorrector(config)
        result = corrector.correct("I use co pilot and pie torch")
        assert result == "I use Copilot and PyTorch"

    def test_replacement_preserves_surrounding_text(self) -> None:
        config = _make_dict_settings(replacements={"arch linux": "Arch Linux"})
        corrector = VocabularyCorrector(config)
        result = corrector.correct("I run arch linux on my machine and it works")
        assert result == "I run Arch Linux on my machine and it works"

    def test_no_match_returns_unchanged(self) -> None:
        config = _make_dict_settings(replacements={"foo": "bar"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("hello world") == "hello world"

    def test_regex_metacharacters_in_key_are_escaped(self) -> None:
        """Keys containing regex special characters should be treated as literal strings."""
        config = _make_dict_settings(replacements={"c++": "C++"})
        corrector = VocabularyCorrector(config)
        result = corrector.correct("I program in c++ daily")
        assert result == "I program in C++ daily"

    def test_pipe_character_in_key_is_escaped(self) -> None:
        """The pipe character | must be escaped so it doesn't break the regex alternation."""
        config = _make_dict_settings(replacements={"a|b": "AB"})
        corrector = VocabularyCorrector(config)
        result = corrector.correct("replace a|b here")
        assert result == "replace AB here"

    def test_replacement_preserves_case_of_surrounding_text(self) -> None:
        config = _make_dict_settings(replacements={"foo": "FOO"})
        corrector = VocabularyCorrector(config)
        result = corrector.correct("lower foo and UPPER FOO")
        assert result == "lower FOO and UPPER FOO"

    def test_same_replacement_applied_multiple_times(self) -> None:
        config = _make_dict_settings(replacements={"foo": "bar"})
        corrector = VocabularyCorrector(config)
        result = corrector.correct("foo foo foo")
        assert result == "bar bar bar"

    def test_overlapping_keys_longest_wins(self) -> None:
        """When keys overlap, the longest key should match first."""
        replacements = {"hello world": "HW", "hello": "H"}
        config = _make_dict_settings(replacements=replacements)
        corrector = VocabularyCorrector(config)
        result = corrector.correct("hello world there")
        assert result == "HW there"

    def test_replacement_value_not_matched_against_other_keys(self) -> None:
        """After a replacement, the replacement value should not be re-scanned."""
        # "pie" -> "PyTorch" would then match "torch" -> "Fire" in a naive loop.
        # With regex sub, this is a single pass so it doesn't happen.
        replacements = {"pie": "PyTorch", "torch": "Fire"}
        config = _make_dict_settings(replacements=replacements)
        corrector = VocabularyCorrector(config)
        result = corrector.correct("I use pie")
        assert result == "I use PyTorch"

    def test_punctuation_adjacent_to_match(self) -> None:
        """Matches should work even when adjacent to punctuation."""
        config = _make_dict_settings(replacements={"co pilot": "Copilot"})
        result = VocabularyCorrector(config).correct("I love co pilot, it is great.")
        assert result == "I love Copilot, it is great."

    def test_match_at_start_of_text(self) -> None:
        config = _make_dict_settings(replacements={"hello world": "Hi"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("hello world there") == "Hi there"

    def test_match_at_end_of_text(self) -> None:
        config = _make_dict_settings(replacements={"hello world": "Hi"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("say hello world") == "say Hi"

    def test_match_is_entire_text(self) -> None:
        config = _make_dict_settings(replacements={"hello world": "Hi"})
        corrector = VocabularyCorrector(config)
        assert corrector.correct("hello world") == "Hi"


class TestVocabularyCorrectorLogging:
    def test_logs_when_corrections_applied(self, caplog: pytest.LogCaptureFixture) -> None:
        config = _make_dict_settings(replacements={"foo": "bar"})
        corrector = VocabularyCorrector(config)
        with caplog.at_level(logging.INFO, logger="vox.dictionary"):
            result = corrector.correct("hello foo world")
        assert result == "hello bar world"
        assert any("corrected" in record.message for record in caplog.records)

    def test_no_log_when_no_corrections(self, caplog: pytest.LogCaptureFixture) -> None:
        config = _make_dict_settings(replacements={"foo": "bar"})
        corrector = VocabularyCorrector(config)
        with caplog.at_level(logging.INFO, logger="vox.dictionary"):
            result = corrector.correct("hello world")
        assert result == "hello world"
        assert not any("corrected" in record.message for record in caplog.records)

    def test_no_log_when_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        config = _make_dict_settings(
            enabled=False,
            replacements={"foo": "bar"},
        )
        corrector = VocabularyCorrector(config)
        with caplog.at_level(logging.INFO, logger="vox.dictionary"):
            result = corrector.correct("hello foo world")
        assert result == "hello foo world"
        assert not any("corrected" in record.message for record in caplog.records)
