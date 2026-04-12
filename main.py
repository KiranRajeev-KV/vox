"""Vox — local, offline, GPU-accelerated voice-to-text."""

from __future__ import annotations

import argparse
import logging
import signal
import sqlite3
import sys
from pathlib import Path

import vox
from vox.config import get_settings
from vox.history import History, SessionRecord
from vox.pipeline import Pipeline

LOG_DIR = Path.home() / ".local" / "share" / "vox"
LOG_FILE = LOG_DIR / "vox.log"

LOG_FMT = "%(asctime)s %(levelname)-8s [%(threadName)-16s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_TEXT_COL_WIDTH = 60
_APP_COL_WIDTH = 12


def setup_logging(debug: bool) -> None:
    """Configure logging to file and optionally to stderr."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
    root_logger.addHandler(file_handler)

    if debug:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
        root_logger.addHandler(console_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vox — voice-to-text")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Default: run the voice pipeline
    subparsers.add_parser("run", help="Run the voice pipeline (default)")

    # Web UI
    web_parser = subparsers.add_parser("web", help="Launch the history web UI")
    web_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve the web UI on (default: 8080)",
    )

    # History browser
    history_parser = subparsers.add_parser("history", help="Browse transcription history")
    history_parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Full-text search query",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Results per page (default: 20)",
    )
    history_parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number (default: 1)",
    )

    # Quick search shortcut
    search_parser = subparsers.add_parser("search", help="Search transcription history")
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max results (default: 20)",
    )

    # Usage statistics
    subparsers.add_parser("stats", help="Show usage statistics")

    # Personal dictionary
    dict_parser = subparsers.add_parser("dict", help="Manage personal vocabulary")
    dict_sub = dict_parser.add_subparsers(dest="dict_command", help="Dictionary commands")

    add_parser = dict_sub.add_parser("add", help="Add a word correction")
    add_parser.add_argument("word", type=str, help="Word as Whisper mishears it")
    add_parser.add_argument("correction", type=str, help="Correct spelling")

    remove_parser = dict_sub.add_parser("remove", help="Remove a word correction")
    remove_parser.add_argument("word", type=str, help="Word to remove")

    dict_sub.add_parser("list", help="List all word corrections")

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to stderr",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.version:
        print(f"Vox {vox.__version__}")
        sys.exit(0)

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Suppress noisy OpenAI client debug logs (request payloads, retries)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)

    if args.command == "web":
        _run_web(args.port)
        return

    if args.command == "history":
        _run_history(args.config, args.search, args.limit, args.page)
        return

    if args.command == "search":
        _run_history(args.config, args.query, args.limit, 1)
        return

    if args.command == "stats":
        _run_stats(args.config)
        return

    if args.command == "dict":
        _run_dict(args.config, args.dict_command, args)
        return

    logger.info("Vox %s starting", vox.__version__)

    settings = get_settings(args.config)
    pipeline = Pipeline(settings)

    def handle_signal(signum: int, frame: object) -> None:
        logger.info("Received signal %s, shutting down", signum)
        pipeline.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error in pipeline")
        sys.exit(1)


def _open_history(config_path: Path | None) -> History | None:
    """Open the history database for CLI commands.

    Returns None if history is disabled or DB doesn't exist.
    Prints error messages to stderr.
    """
    settings = get_settings(config_path)
    if not settings.history.enabled:
        print("History is disabled in config.toml", file=sys.stderr)
        sys.exit(1)

    db_path = Path(settings.history.db_path)
    if not db_path.exists():
        print("No history database found. Run Vox first.", file=sys.stderr)
        sys.exit(1)

    try:
        return History(settings.history)
    except sqlite3.Error as e:
        print(f"Failed to open history database: {e}", file=sys.stderr)
        sys.exit(1)


def _run_history(
    config_path: Path | None,
    search: str | None,
    limit: int,
    page: int,
) -> None:
    """Browse and search transcription history."""
    history = _open_history(config_path)
    if history is None:
        return

    try:
        if search:
            if not search.strip():
                print("Search query cannot be empty.", file=sys.stderr)
                sys.exit(1)

            results = history.search(search.strip(), limit=limit)
            print(f'Search: "{search}" ({len(results)} result{"s" if len(results) != 1 else ""})')
            print()
            _print_sessions(results)
        else:
            offset = (page - 1) * limit
            results = history.get_recent_paginated(limit=limit, offset=offset)
            if not results:
                print("No sessions found.")
                return
            _print_sessions(results)
            total = history.get_count()
            total_pages = (total + limit - 1) // limit
            if total_pages > 1:
                print()
                print(f"Page {page} of {total_pages} (use --page to navigate)")
    finally:
        history.close()


def _run_stats(config_path: Path | None) -> None:
    """Show aggregate usage statistics."""
    history = _open_history(config_path)
    if history is None:
        return

    try:
        stats = history.get_stats()
        if stats.total_sessions == 0:
            print("Vox Usage Statistics")
            print("─" * 30)
            print("No transcription history yet.")
            return

        avg_latency = history.get_avg_latency()

        print("Vox Usage Statistics")
        print("─" * 30)
        print(f"Sessions:          {_format_number(stats.total_sessions)}")
        print(f"Total words:       {_format_number(stats.total_words)}")
        print(f"Total duration:    {_format_duration(stats.total_duration_ms)}")
        print(f"Avg latency:       {_format_latency(avg_latency)}")
        if stats.first_used_at:
            print(f"First used:        {stats.first_used_at}")
        if stats.last_used_at:
            print(f"Last used:         {stats.last_used_at}")
    finally:
        history.close()


def _print_sessions(sessions: list[SessionRecord]) -> None:
    """Print sessions as a formatted table."""
    if not sessions:
        print("No sessions found.")
        return

    header = f"{'ID':>4}  {'Date':<19}  {'Words':>5}  {'App':<{_APP_COL_WIDTH}}  {'Text'}"
    print(header)
    print("-" * len(header))

    for s in sessions:
        text_preview = s.raw_text[:_TEXT_COL_WIDTH]
        if len(s.raw_text) > _TEXT_COL_WIDTH:
            text_preview += "..."
        app = s.app_context[:_APP_COL_WIDTH]
        date = s.created_at[:19] if s.created_at else ""
        print(f"{s.id:>4}  {date:<19}  {s.word_count:>5}  {app:<{_APP_COL_WIDTH}}  {text_preview}")


def _format_duration(ms: int) -> str:
    """Format milliseconds to human-readable duration."""
    if ms < 0:
        return "0s"
    if ms < 1000:
        return f"{ms}ms"

    total_seconds = ms // 1000
    if total_seconds < 60:
        return f"{total_seconds}s"

    total_minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60

    if total_minutes < 60:
        if remaining_seconds > 0:
            return f"{total_minutes}m {remaining_seconds}s"
        return f"{total_minutes}m"

    total_hours = total_minutes // 60
    remaining_minutes = total_minutes % 60

    if total_hours < 24:
        parts = [f"{total_hours}h"]
        if remaining_minutes > 0:
            parts.append(f"{remaining_minutes}m")
        return " ".join(parts)

    total_days = total_hours // 24
    remaining_hours = total_hours % 24
    parts = [f"{total_days}d"]
    if remaining_hours > 0:
        parts.append(f"{remaining_hours}h")
    return " ".join(parts)


def _format_latency(ms: float) -> str:
    """Format latency in ms to human-readable string."""
    if ms <= 0:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _format_number(n: int) -> str:
    """Format integer with comma separators."""
    return f"{n:,}"


def _run_web(port: int) -> None:
    """Launch the Flask-based history web UI."""
    from vox.web import app

    logger = logging.getLogger(__name__)
    logger.info("Starting Vox web UI on http://localhost:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)


def _resolve_config(config_path: Path | None) -> Path:
    """Return the path to config.toml, using default if not specified."""
    if config_path:
        return config_path
    # Check working directory first, then fall back to project root
    for candidate in [Path("config.toml"), Path(__file__).parent / "config.toml"]:
        if candidate.exists():
            return candidate.resolve()
    print("config.toml not found. Create one from config.toml.example.", file=sys.stderr)
    sys.exit(1)


def _read_dict_section(path: Path) -> dict[str, str]:
    """Parse only the [dictionary] replacements from config.toml as text.

    Returns a dict of {misheard: correction}.
    """
    replacements: dict[str, str] = {}
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return replacements

    in_replacements = False

    for line in lines:
        stripped = line.strip()

        # Detect section boundaries
        if stripped.startswith("[") and not stripped.startswith("[["):
            if stripped == "[dictionary]":
                in_replacements = False
                continue
            elif stripped == "[dictionary.replacements]":
                in_replacements = True
                continue
            else:
                in_replacements = False
                continue

        # Parse replacements
        if in_replacements and "=" in stripped:
            # Format: "misheard" = "correction"
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                key = parts[0].strip().strip('"')
                value = parts[1].strip().strip('"')
                replacements[key] = value

    return replacements


def _update_dict_section(path: Path, replacements: dict[str, str]) -> None:
    """Update the [dictionary] replacements in config.toml, preserving all other content.

    If the section doesn't exist, appends it to the end of the file.
    """
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Failed to read config.toml: {e}", file=sys.stderr)
        sys.exit(1)

    # Find where the replacements section is (or should go)
    in_replacements = False
    replace_start = -1
    replace_end = -1

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("[") and not stripped.startswith("[["):
            if stripped == "[dictionary.replacements]":
                in_replacements = True
                replace_start = i + 1
                continue
            elif stripped == "[dictionary]":
                in_replacements = False
                continue
            else:
                if in_replacements:
                    replace_end = i
                    in_replacements = False
                continue

        if in_replacements and "=" in stripped:
            # This is a replacement line — mark for removal
            if replace_end == -1:
                replace_start = i  # first replacement line
            replace_end = i + 1  # one past last replacement

    # Build new replacements block
    if replacements:
        new_block = ["[dictionary.replacements]\n"]
        for key, value in sorted(replacements.items()):
            new_block.append(f'"{key}" = "{value}"\n')
    else:
        new_block = []

    if replace_start >= 0 and replace_end >= 0:
        # Replace existing section
        lines[replace_start:replace_end] = new_block
        # If no replacements, remove the entire [dictionary.replacements] header too
        if not new_block:
            # Find and remove the [dictionary.replacements] line
            for i in range(replace_start - 1, -1, -1):
                if lines[i].strip() == "[dictionary.replacements]":
                    lines.pop(i)
                    # Also remove blank lines before it
                    while i > 0 and lines[i - 1].strip() == "":
                        lines.pop(i - 1)
                        i -= 1
                    break
    elif replace_start >= 0 and replace_end < 0:
        # Section header exists but no content — append after header
        lines[replace_start:replace_start] = new_block
    else:
        # No dictionary section at all — append to end
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        # Only write [dictionary] header if we have replacements
        if replacements:
            lines.append("[dictionary]\n")
            lines.extend(new_block)

    try:
        with open(path, "w") as f:
            f.writelines(lines)
    except OSError as e:
        print(f"Failed to write config.toml: {e}", file=sys.stderr)
        sys.exit(1)


def _run_dict(config_path: Path | None, command: str | None, args: argparse.Namespace) -> None:
    """Manage personal vocabulary corrections in config.toml."""
    if not command:
        print("Usage: vox dict <add|remove|list>", file=sys.stderr)
        sys.exit(1)

    config_file = _resolve_config(config_path)
    replacements = _read_dict_section(config_file)

    if command == "add":
        word = args.word.strip()
        correction = args.correction.strip()
        if not word or not correction:
            print("Error: word and correction cannot be empty.", file=sys.stderr)
            sys.exit(1)
        if word in replacements:
            old = replacements[word]
            print(f"Updated: {word!r} → {correction!r} (was: {old!r})")
        else:
            print(f"Added: {word!r} → {correction!r}")
        replacements[word] = correction
        _update_dict_section(config_file, replacements)

    elif command == "remove":
        word = args.word.strip()
        if not word:
            print("Error: word cannot be empty.", file=sys.stderr)
            sys.exit(1)
        if word in replacements:
            del replacements[word]
            _update_dict_section(config_file, replacements)
            print(f"Removed: {word!r}")
        else:
            print(f"Not found: {word!r}")
            sys.exit(1)

    elif command == "list":
        if not replacements:
            print("No vocabulary corrections defined.")
            return
        print(f"{'Misheard':<20}  {'Correction'}")
        print("-" * 40)
        for word, correction in sorted(replacements.items()):
            print(f"{word:<20}  {correction}")
        print(f"\n{len(replacements)} correction{'s' if len(replacements) != 1 else ''} defined.")

    else:
        print(f"Unknown dictionary command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
