"""Transcription history — SQLite storage with FTS5 full-text search."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from vox.config import HistorySettings

logger = logging.getLogger(__name__)

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at                DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_text                  TEXT NOT NULL,
    clean_text                TEXT,
    duration_ms               INTEGER,
    duration_after_vad_ms     INTEGER,
    word_count                INTEGER,
    app_context               TEXT,
    language                  TEXT,
    model_used                TEXT,
    transcription_latency_ms  INTEGER,
    full_pipeline_latency_ms  INTEGER
);
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    raw_text,
    clean_text,
    content='sessions',
    content_rowid='id'
);
"""

_CREATE_STATS = """
CREATE TABLE IF NOT EXISTS stats (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    total_sessions    INTEGER DEFAULT 0,
    total_words       INTEGER DEFAULT 0,
    total_duration_ms INTEGER DEFAULT 0,
    first_used_at     DATETIME,
    last_used_at      DATETIME
);
"""

_CREATE_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
        INSERT INTO sessions_fts(rowid, raw_text, clean_text)
        VALUES (new.id, new.raw_text, new.clean_text);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
        INSERT INTO sessions_fts(sessions_fts, rowid, raw_text, clean_text)
        VALUES ('delete', old.id, old.raw_text, old.clean_text);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
        INSERT INTO sessions_fts(sessions_fts, rowid, raw_text, clean_text)
        VALUES ('delete', old.id, old.raw_text, old.clean_text);
        INSERT INTO sessions_fts(rowid, raw_text, clean_text)
        VALUES (new.id, new.raw_text, new.clean_text);
    END;
    """,
]

_INSERT_SESSION = """
INSERT INTO sessions (
    raw_text, clean_text, duration_ms, duration_after_vad_ms,
    word_count, app_context, language, model_used,
    transcription_latency_ms, full_pipeline_latency_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_UPDATE_STATS = """
INSERT INTO stats (id, total_sessions, total_words, total_duration_ms,
                   first_used_at, last_used_at)
VALUES (1, 1, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(id) DO UPDATE SET
    total_sessions = total_sessions + 1,
    total_words = total_words + excluded.total_words,
    total_duration_ms = total_duration_ms + excluded.total_duration_ms,
    last_used_at = CURRENT_TIMESTAMP;
"""

_ENFORCE_MAX = """
DELETE FROM sessions WHERE id NOT IN (
    SELECT id FROM sessions ORDER BY created_at DESC LIMIT ?
);
"""

_SEARCH = """
SELECT s.id, s.created_at, s.raw_text, s.clean_text,
       s.duration_ms, s.duration_after_vad_ms, s.word_count,
       s.app_context, s.language, s.model_used,
       s.transcription_latency_ms, s.full_pipeline_latency_ms
FROM sessions s
JOIN sessions_fts f ON s.id = f.rowid
WHERE sessions_fts MATCH ?
ORDER BY s.created_at DESC
LIMIT ?;
"""

_GET_RECENT = """
SELECT id, created_at, raw_text, clean_text,
       duration_ms, duration_after_vad_ms, word_count,
       app_context, language, model_used,
       transcription_latency_ms, full_pipeline_latency_ms
FROM sessions
ORDER BY created_at DESC
LIMIT ?;
"""

_GET_STATS = """
SELECT total_sessions, total_words, total_duration_ms,
       first_used_at, last_used_at
FROM stats WHERE id = 1;
"""

_GET_COUNT = """
SELECT COUNT(*) FROM sessions;
"""

_GET_AVG_LATENCY = """
SELECT AVG(full_pipeline_latency_ms) FROM sessions
WHERE full_pipeline_latency_ms IS NOT NULL AND full_pipeline_latency_ms > 0;
"""

_GET_RECENT_PAGINATED = """
SELECT id, created_at, raw_text, clean_text,
       duration_ms, duration_after_vad_ms, word_count,
       app_context, language, model_used,
       transcription_latency_ms, full_pipeline_latency_ms
FROM sessions
ORDER BY created_at DESC
LIMIT ? OFFSET ?;
"""


@dataclass
class SessionRecord:
    """A single transcription session."""

    id: int | None
    created_at: str
    raw_text: str
    clean_text: str | None
    duration_ms: int
    duration_after_vad_ms: int
    word_count: int
    app_context: str
    language: str
    model_used: str
    transcription_latency_ms: int
    full_pipeline_latency_ms: int


@dataclass
class StatsRecord:
    """Aggregate usage statistics."""

    total_sessions: int
    total_words: int
    total_duration_ms: int
    first_used_at: str
    last_used_at: str


class History:
    """SQLite storage for transcription sessions with FTS5 full-text search.

    Every session is saved with raw text, cleaned text, duration, word count,
    active window class, detected language, and model used. FTS5 indexes raw
    and clean text for fast keyword search.
    """

    def __init__(self, config: HistorySettings) -> None:
        self._config = config
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = sqlite3.connect(str(db_path))  # type: ignore[assignment]
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

        logger.info("History database opened: %s", db_path)

    def _init_schema(self) -> None:
        """Create tables, FTS index, and triggers if they don't exist."""
        assert self._conn is not None
        self._conn.execute(_CREATE_SESSIONS)
        self._conn.execute(_CREATE_FTS)
        self._conn.execute(_CREATE_STATS)
        for trigger in _CREATE_TRIGGERS:
            self._conn.execute(trigger)
        self._conn.commit()

    def save_session(self, session: SessionRecord) -> int:
        """Insert a session record and update aggregate stats.

        Args:
            session: The session data to persist.

        Returns:
            The database row ID of the new session, or 0 on failure.
        """
        try:
            assert self._conn is not None
            cursor = self._conn.execute(
                _INSERT_SESSION,
                (
                    session.raw_text,
                    session.clean_text,
                    session.duration_ms,
                    session.duration_after_vad_ms,
                    session.word_count,
                    session.app_context,
                    session.language,
                    session.model_used,
                    session.transcription_latency_ms,
                    session.full_pipeline_latency_ms,
                ),
            )
            row_id = cursor.lastrowid

            self._conn.execute(_UPDATE_STATS, (session.word_count, session.duration_ms))

            if self._config.max_entries > 0:
                self._conn.execute(_ENFORCE_MAX, (self._config.max_entries,))

            self._conn.commit()
            return row_id or 0
        except Exception:
            logger.exception("Failed to save session to history")
            return 0

    def search(self, query: str, limit: int = 50) -> list[SessionRecord]:
        """Full-text search across raw and clean text.

        Args:
            query: FTS5 search query (supports boolean operators).
            limit: Maximum number of results to return.

        Returns:
            Matching sessions ordered by most recent first, or empty list on failure.
        """
        try:
            assert self._conn is not None
            rows = self._conn.execute(_SEARCH, (query, limit)).fetchall()
            return [_row_to_record(r) for r in rows]
        except Exception:
            logger.exception("Failed to search history")
            return []

    def get_recent(self, limit: int = 20) -> list[SessionRecord]:
        """Get the most recent transcription sessions.

        Args:
            limit: Maximum number of results to return.

        Returns:
            Sessions ordered by creation date descending, or empty list on failure.
        """
        try:
            assert self._conn is not None
            rows = self._conn.execute(_GET_RECENT, (limit,)).fetchall()
            return [_row_to_record(r) for r in rows]
        except Exception:
            logger.exception("Failed to get recent sessions")
            return []

    def get_stats(self) -> StatsRecord:
        """Get aggregate usage statistics.

        Returns:
            StatsRecord with totals, or zeroes if no sessions exist or on failure.
        """
        try:
            assert self._conn is not None
            row = self._conn.execute(_GET_STATS).fetchone()
            if row is None:
                return StatsRecord(0, 0, 0, "", "")
            return StatsRecord(*row)
        except Exception:
            logger.exception("Failed to get stats")
            return StatsRecord(0, 0, 0, "", "")

    def get_count(self) -> int:
        """Get total number of sessions."""
        try:
            assert self._conn is not None
            row = self._conn.execute(_GET_COUNT).fetchone()
            return row[0] if row else 0
        except Exception:
            logger.exception("Failed to get session count")
            return 0

    def get_avg_latency(self) -> float:
        """Get average full-pipeline latency in ms."""
        try:
            assert self._conn is not None
            row = self._conn.execute(_GET_AVG_LATENCY).fetchone()
            return row[0] if row and row[0] is not None else 0.0
        except Exception:
            logger.exception("Failed to get average latency")
            return 0.0

    def get_recent_paginated(self, limit: int = 20, offset: int = 0) -> list[SessionRecord]:
        """Get sessions with proper OFFSET/LIMIT pagination.

        Args:
            limit: Maximum number of results to return.
            offset: Number of rows to skip.

        Returns:
            Sessions ordered by creation date descending, or empty list on failure.
        """
        try:
            assert self._conn is not None
            rows = self._conn.execute(_GET_RECENT_PAGINATED, (limit, offset)).fetchall()
            return [_row_to_record(r) for r in rows]
        except Exception:
            logger.exception("Failed to get paginated sessions")
            return []

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
            logger.info("History database closed")


def _row_to_record(row: tuple[object, ...]) -> SessionRecord:
    """Convert a database row to a SessionRecord."""
    return SessionRecord(
        id=int(row[0]) if row[0] is not None else 0,  # type: ignore[arg-type]
        created_at=str(row[1]),
        raw_text=str(row[2]),
        clean_text=str(row[3]) if row[3] is not None else None,
        duration_ms=int(row[4]) if row[4] is not None else 0,  # type: ignore[arg-type]
        duration_after_vad_ms=int(row[5]) if row[5] is not None else 0,  # type: ignore[arg-type]
        word_count=int(row[6]) if row[6] is not None else 0,  # type: ignore[arg-type]
        app_context=str(row[7]) if row[7] is not None else "",
        language=str(row[8]) if row[8] is not None else "",
        model_used=str(row[9]) if row[9] is not None else "",
        transcription_latency_ms=int(row[10]) if row[10] is not None else 0,  # type: ignore[arg-type]
        full_pipeline_latency_ms=int(row[11]) if row[11] is not None else 0,  # type: ignore[arg-type]
    )
