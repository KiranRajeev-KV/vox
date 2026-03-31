"""Vox — local, offline, GPU-accelerated voice-to-text."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import vox
from vox.config import get_settings
from vox.pipeline import Pipeline

LOG_DIR = Path.home() / ".local" / "share" / "vox"
LOG_FILE = LOG_DIR / "vox.log"

LOG_FMT = "%(asctime)s %(levelname)-8s [%(threadName)-16s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


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


def _run_web(port: int) -> None:
    """Launch the Flask-based history web UI."""
    from vox.web import app

    logger = logging.getLogger(__name__)
    logger.info("Starting Vox web UI on http://localhost:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
