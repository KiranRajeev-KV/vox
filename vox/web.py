"""Web UI for Vox transcription history."""

from __future__ import annotations

import logging

from flask import Flask, Response, jsonify, render_template, request

from vox.config import get_settings
from vox.history import History, SessionRecord

logger = logging.getLogger(__name__)

app = Flask(__name__)


def _get_history() -> History:
    """Create a fresh History instance per request for thread safety."""
    return History(get_settings().history)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/stats")
def api_stats() -> Response | tuple[Response, int]:
    try:
        history = _get_history()
        stats = history.get_stats()
        avg_latency = history.get_avg_latency()
        history.close()
        return jsonify(
            {
                "total_sessions": stats.total_sessions,
                "total_words": stats.total_words,
                "total_duration_ms": stats.total_duration_ms,
                "first_used_at": stats.first_used_at,
                "last_used_at": stats.last_used_at,
                "avg_latency_ms": round(avg_latency, 1),
            }
        )
    except Exception:
        logger.exception("Failed to fetch stats")
        return jsonify({"error": "Failed to fetch stats"}), 500


@app.route("/api/sessions")
def api_sessions() -> Response | tuple[Response, int]:
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        per_page = min(per_page, 100)

        history = _get_history()
        total = history.get_count()
        page_sessions = history.get_recent_paginated(per_page, (page - 1) * per_page)
        history.close()

        return jsonify(
            {
                "sessions": [_session_to_dict(s) for s in page_sessions],
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": max(1, (total + per_page - 1) // per_page),
            }
        )
    except Exception:
        logger.exception("Failed to fetch sessions")
        return jsonify({"error": "Failed to fetch sessions"}), 500


@app.route("/api/search")
def api_search() -> Response | tuple[Response, int]:
    try:
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"sessions": [], "total": 0})

        limit = request.args.get("limit", 50, type=int)
        history = _get_history()
        sessions = history.search(q, limit)
        history.close()

        return jsonify(
            {
                "sessions": [_session_to_dict(s) for s in sessions],
                "total": len(sessions),
                "query": q,
            }
        )
    except Exception:
        logger.exception("Failed to search sessions")
        return jsonify({"error": "Failed to search sessions"}), 500


@app.route("/api/session/<int:session_id>")
def api_session(session_id: int) -> Response | tuple[Response, int]:
    try:
        history = _get_history()
        sessions = history.get_recent(10000)
        history.close()
        session = next((s for s in sessions if s.id == session_id), None)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(_session_to_dict(session))
    except Exception:
        logger.exception("Failed to fetch session %d", session_id)
        return jsonify({"error": "Failed to fetch session"}), 500


def _session_to_dict(session: SessionRecord) -> dict[str, object]:
    return {
        "id": session.id,
        "created_at": session.created_at,
        "raw_text": session.raw_text,
        "clean_text": session.clean_text,
        "duration_ms": session.duration_ms,
        "duration_after_vad_ms": session.duration_after_vad_ms,
        "word_count": session.word_count,
        "app_context": session.app_context,
        "language": session.language,
        "model_used": session.model_used,
        "transcription_latency_ms": session.transcription_latency_ms,
        "full_pipeline_latency_ms": session.full_pipeline_latency_ms,
    }
