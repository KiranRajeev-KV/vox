"""Tests for history — SQLite storage with FTS5 full-text search."""

from __future__ import annotations

from pathlib import Path

from vox.config import HistorySettings
from vox.history import (
    History,
    SessionRecord,
    _row_to_record,
)


def _make_session(**kwargs: object) -> SessionRecord:
    defaults: dict[str, object] = {
        "id": None,
        "created_at": "",
        "raw_text": "hello world",
        "clean_text": "Hello world.",
        "duration_ms": 3000,
        "duration_after_vad_ms": 2500,
        "word_count": 2,
        "app_context": "firefox",
        "language": "en",
        "model_used": "large-v3",
        "transcription_latency_ms": 800,
        "full_pipeline_latency_ms": 1200,
    }
    defaults.update(kwargs)
    return SessionRecord(**defaults)  # type: ignore[arg-type]


def _make_history(tmp_path: Path, **kwargs: object) -> History:
    db_path = str(tmp_path / "test.db")
    defaults: dict[str, object] = {
        "enabled": True,
        "db_path": db_path,
        "max_entries": 10000,
    }
    defaults.update(kwargs)
    config = HistorySettings(**defaults)  # type: ignore[arg-type]
    return History(config)


class TestInit:
    def test_creates_tables(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            tables = history._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {t[0] for t in tables}
            assert "sessions" in names
            assert "sessions_fts" in names
            assert "stats" in names
        finally:
            history.close()

    def test_creates_triggers(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            triggers = history._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
            names = {t[0] for t in triggers}
            assert "sessions_ai" in names
            assert "sessions_ad" in names
            assert "sessions_au" in names
        finally:
            history.close()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "test.db"
        config = HistorySettings(
            enabled=True,
            db_path=str(nested),
            max_entries=10000,
        )
        history = History(config)
        try:
            assert nested.exists()
        finally:
            history.close()


class TestSaveSession:
    def test_save_and_return_row_id(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            session = _make_session(raw_text="test")
            row_id = history.save_session(session)
            assert row_id > 0
        finally:
            history.close()

    def test_save_retrieves_correctly(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            session = _make_session(
                raw_text="original text",
                clean_text="cleaned text",
                word_count=5,
                app_context="code",
                language="de",
            )
            history.save_session(session)
            recent = history.get_recent()
            assert len(recent) == 1
            assert recent[0].raw_text == "original text"
            assert recent[0].clean_text == "cleaned text"
            assert recent[0].word_count == 5
            assert recent[0].app_context == "code"
            assert recent[0].language == "de"
        finally:
            history.close()

    def test_save_updates_stats(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            session = _make_session(word_count=10, duration_ms=5000)
            history.save_session(session)
            stats = history.get_stats()
            assert stats.total_sessions == 1
            assert stats.total_words == 10
            assert stats.total_duration_ms == 5000
        finally:
            history.close()

    def test_save_multiple_updates_stats(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session(word_count=5, duration_ms=1000))
            history.save_session(_make_session(word_count=15, duration_ms=2000))
            history.save_session(_make_session(word_count=10, duration_ms=3000))
            stats = history.get_stats()
            assert stats.total_sessions == 3
            assert stats.total_words == 30
            assert stats.total_duration_ms == 6000
        finally:
            history.close()

    def test_save_with_null_clean_text(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            session = _make_session(clean_text=None)
            row_id = history.save_session(session)
            assert row_id > 0
            recent = history.get_recent()
            assert recent[0].clean_text is None
        finally:
            history.close()


class TestGetRecent:
    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            assert history.get_recent() == []
        finally:
            history.close()

    def test_returns_most_recent_first(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history._conn.execute(
                "INSERT INTO sessions (raw_text, created_at) VALUES (?, '2026-01-01 00:00:01')",
                ("first",),
            )
            history._conn.execute(
                "INSERT INTO sessions (raw_text, created_at) VALUES (?, '2026-01-01 00:00:02')",
                ("second",),
            )
            history._conn.execute(
                "INSERT INTO sessions (raw_text, created_at) VALUES (?, '2026-01-01 00:00:03')",
                ("third",),
            )
            history._conn.commit()
            recent = history.get_recent()
            assert recent[0].raw_text == "third"
            assert recent[1].raw_text == "second"
            assert recent[2].raw_text == "first"
        finally:
            history.close()

    def test_respects_limit(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            for i in range(5):
                ts = f"2026-01-01 00:00:{i + 1:02d}"
                history._conn.execute(
                    "INSERT INTO sessions (raw_text, created_at) VALUES (?, ?)",
                    (f"session {i}", ts),
                )
            history._conn.commit()
            recent = history.get_recent(limit=2)
            assert len(recent) == 2
            assert recent[0].raw_text == "session 4"
        finally:
            history.close()


class TestSearch:
    def test_search_raw_text(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session(raw_text="the quick brown fox"))
            results = history.search("fox")
            assert len(results) == 1
            assert results[0].raw_text == "the quick brown fox"
        finally:
            history.close()

    def test_search_clean_text(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session(raw_text="um uh hello", clean_text="Hello there."))
            results = history.search("there")
            assert len(results) == 1
        finally:
            history.close()

    def test_search_no_match(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session(raw_text="hello world"))
            results = history.search("nonexistent")
            assert results == []
        finally:
            history.close()

    def test_search_respects_limit(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            for i in range(5):
                history.save_session(_make_session(raw_text=f"test word {i}"))
            results = history.search("test", limit=2)
            assert len(results) == 2
        finally:
            history.close()

    def test_search_multiple_matches(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history._conn.execute(
                "INSERT INTO sessions (raw_text, created_at) VALUES (?, '2026-01-01 00:00:01')",
                ("meeting notes monday",),
            )
            history._conn.execute(
                "INSERT INTO sessions (raw_text, created_at) VALUES (?, '2026-01-01 00:00:02')",
                ("meeting notes tuesday",),
            )
            history._conn.execute(
                "INSERT INTO sessions (raw_text, created_at) VALUES (?, '2026-01-01 00:00:03')",
                ("grocery list",),
            )
            history._conn.commit()
            results = history.search("meeting")
            assert len(results) == 2
            assert results[0].raw_text == "meeting notes tuesday"
            assert results[1].raw_text == "meeting notes monday"
        finally:
            history.close()


class TestGetStats:
    def test_empty_stats(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            stats = history.get_stats()
            assert stats.total_sessions == 0
            assert stats.total_words == 0
            assert stats.total_duration_ms == 0
        finally:
            history.close()

    def test_stats_after_saves(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session(word_count=8, duration_ms=4000))
            history.save_session(_make_session(word_count=12, duration_ms=6000))
            stats = history.get_stats()
            assert stats.total_sessions == 2
            assert stats.total_words == 20
            assert stats.total_duration_ms == 10000
            assert stats.first_used_at != ""
            assert stats.last_used_at != ""
        finally:
            history.close()


class TestGetCount:
    def test_empty_count(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            assert history.get_count() == 0
        finally:
            history.close()

    def test_count_after_saves(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            for _ in range(7):
                history.save_session(_make_session())
            assert history.get_count() == 7
        finally:
            history.close()


class TestGetAvgLatency:
    def test_empty_returns_zero(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            assert history.get_avg_latency() == 0.0
        finally:
            history.close()

    def test_correct_average(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session(full_pipeline_latency_ms=1000))
            history.save_session(_make_session(full_pipeline_latency_ms=2000))
            history.save_session(_make_session(full_pipeline_latency_ms=3000))
            assert history.get_avg_latency() == 2000.0
        finally:
            history.close()

    def test_ignores_zero_latency(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session(full_pipeline_latency_ms=0))
            history.save_session(_make_session(full_pipeline_latency_ms=1000))
            assert history.get_avg_latency() == 1000.0
        finally:
            history.close()

    def test_ignores_null_latency(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            session = _make_session(full_pipeline_latency_ms=0)
            history.save_session(session)
            history.save_session(_make_session(full_pipeline_latency_ms=500))
            assert history.get_avg_latency() == 500.0
        finally:
            history.close()


class TestPagination:
    def test_first_page(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            for i in range(5):
                ts = f"2026-01-01 00:00:{i + 1:02d}"
                history._conn.execute(
                    "INSERT INTO sessions (raw_text, created_at) VALUES (?, ?)",
                    (f"item {i}", ts),
                )
            history._conn.commit()
            page = history.get_recent_paginated(limit=3, offset=0)
            assert len(page) == 3
            assert page[0].raw_text == "item 4"
        finally:
            history.close()

    def test_second_page(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            for i in range(6):
                ts = f"2026-01-01 00:00:{i + 1:02d}"
                history._conn.execute(
                    "INSERT INTO sessions (raw_text, created_at) VALUES (?, ?)",
                    (f"item {i}", ts),
                )
            history._conn.commit()
            page = history.get_recent_paginated(limit=3, offset=3)
            assert len(page) == 3
            assert page[0].raw_text == "item 2"
        finally:
            history.close()

    def test_partial_last_page(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            for i in range(4):
                ts = f"2026-01-01 00:00:{i + 1:02d}"
                history._conn.execute(
                    "INSERT INTO sessions (raw_text, created_at) VALUES (?, ?)",
                    (f"item {i}", ts),
                )
            history._conn.commit()
            page = history.get_recent_paginated(limit=3, offset=3)
            assert len(page) == 1
            assert page[0].raw_text == "item 0"
        finally:
            history.close()

    def test_offset_beyond_count(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        try:
            history.save_session(_make_session())
            page = history.get_recent_paginated(limit=10, offset=100)
            assert page == []
        finally:
            history.close()


class TestMaxEntries:
    def test_enforces_max(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path, max_entries=3)
        try:
            for i in range(6):
                ts = f"2026-01-01 00:00:{i + 1:02d}"
                history._conn.execute(
                    "INSERT INTO sessions (raw_text, created_at) VALUES (?, ?)",
                    (f"session {i}", ts),
                )
                history._conn.commit()
                history._conn.execute(
                    "DELETE FROM sessions WHERE id NOT IN ("
                    "SELECT id FROM sessions ORDER BY created_at DESC LIMIT 3)"
                )
                history._conn.commit()
            count = history.get_count()
            assert count == 3
            recent = history.get_recent()
            assert recent[0].raw_text == "session 5"
            assert recent[2].raw_text == "session 3"
        finally:
            history.close()

    def test_no_max_when_zero(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path, max_entries=0)
        try:
            for _ in range(10):
                history.save_session(_make_session())
            assert history.get_count() == 10
        finally:
            history.close()


class TestClose:
    def test_close_sets_conn_none(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        history.close()
        assert history._conn is None

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        history = _make_history(tmp_path)
        history.close()
        history.close()


class TestRowToRecord:
    def test_null_fields_defaults(self) -> None:
        row: tuple[object, ...] = (
            1,
            "2026-01-01",
            "raw",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        record = _row_to_record(row)
        assert record.id == 1
        assert record.raw_text == "raw"
        assert record.clean_text is None
        assert record.duration_ms == 0
        assert record.duration_after_vad_ms == 0
        assert record.word_count == 0
        assert record.app_context == ""
        assert record.language == ""
        assert record.model_used == ""
        assert record.transcription_latency_ms == 0
        assert record.full_pipeline_latency_ms == 0
