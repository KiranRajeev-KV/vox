# Vox

> Press a key. Speak. Text appears.

A local, offline, GPU-accelerated voice-to-text tool for Linux. Audio never leaves your machine. LLM cleanup runs locally by default, or via any OpenAI-compatible cloud provider you configure.

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#development">Development</a>
</p>

---

## What Is Vox?

Vox is a personal voice-to-text tool for Linux. Press a hotkey, speak naturally, and your words are transcribed and pasted directly into whatever window is focused — your editor, browser, terminal, notes app, anything.

It runs Whisper locally on your GPU, cleans up your speech with an LLM of your choice, and gets out of your way. No accounts, no subscriptions, no telemetry. Inspired by [Wispr Flow](https://wisprflow.ai/) and [Superwhisper](https://superwhisper.com/).

---

## Features

- **Global hotkey** — configurable single keys (`f9`, `space`) and modifier combos (`ctrl+shift+space`), works across all apps
- **Two recording modes** — toggle (press once to start, press again to stop) or push-to-talk (hold to record)
- **Local transcription** — faster-whisper with Whisper large-v3, runs on your GPU via CUDA
- **Voice Activity Detection** — Silero VAD v6 bundled in faster-whisper, no extra dependency
- **Smart cleanup** — filler word removal and grammar cleanup via any OpenAI-compatible LLM (Ollama, Groq, OpenRouter, OpenAI, etc.)
- **Optimistic paste** — raw text appears immediately, LLM cleanup runs silently in the background and replaces the text when ready
- **Context-aware replace** — skips LLM text replacement in terminals and editors where synthetic keyboard input is unsafe, with configurable blacklist and stale timeout
- **Clipboard fallback** — if direct paste fails, copies to clipboard and notifies you
- **Recording indicator** — thin bar at the top of the screen while the mic is live (tkinter, click-through DOCK window)
- **Sound cues** — subtle start/stop beeps with configurable volume
- **Full history** — every transcription saved to SQLite with FTS5 full-text search
- **Personal dictionary** — correct Whisper's persistent misrecognitions (say "vox dict add <word> <correction>")
- **Web UI** — browse and search your transcription history in a minimal Flask interface
- **CLI tools** — `vox history`, `vox search`, `vox stats`, `vox dict` from the command line
- **Real-time streaming** — optional CarelessWhisper backend shows partial text on the indicator as you speak
- **Privacy-first** — audio never leaves your machine. Transcribed text stays local when using a local LLM; sent only to the configured endpoint if using a cloud provider

---

## Requirements

### System

- Arch Linux (or any systemd-based Linux)
- X11 (Wayland not supported — xdotool requires X11)
- NVIDIA GPU with CUDA (recommended — CPU fallback works but is slower)
- [uv](https://docs.astral.sh/uv/) for Python environment management

### System Packages

```bash
sudo pacman -S xdotool libnotify xprop
```

### LLM Provider (Optional)

Vox works without an LLM — you'll get raw Whisper output with filler word removal. For grammar cleanup, configure any OpenAI-compatible endpoint:

**Local (Ollama):**
```bash
ollama pull phi3-mini
```

**Cloud providers** — no local install needed, just configure `base_url` + `model` in `config.toml`:

| Provider | base_url | Example model |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `phi3-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| OpenRouter | `https://openrouter.ai/api/v1` | `google/gemini-flash-1.5` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |

---

## Streaming Transcription (Optional)

Vox supports real-time streaming transcription via [CarelessWhisper](https://huggingface.co/MLSpeech/CarelessWhisper-Streaming), showing partial text on the indicator bar as you speak. This requires an NVIDIA GPU with CUDA 13 support and ~2.5GB VRAM for the `small` model.

```bash
just setup-streaming   # clone WhisperRT-Streaming, sync deps, login to HF
```

Then enable in `config.toml`:

```toml
[streaming]
enabled = true
model = "small"           # base, small, large-v2
chunk_size_ms = 300       # decode frequency
beam_size = 0             # 0 = greedy, 1+ = beam search
```

When streaming is enabled, the indicator bar expands to 40px with 12px font to display live partial transcriptions. The indicator collapses back to normal height when idle.

**Offline fallback:** If streaming fails to start, Vox falls back to the standard faster-whisper offline transcription automatically.

---

## Installation

```bash
git clone https://github.com/KiranRajeev-KV/vox.git
cd vox
uv sync
uv sync --group dev          # install dev tools (optional)
cp config.toml.example config.toml
just install-hooks           # set up pre-commit hooks
```

Edit `config.toml` to configure your hotkey, transcription model, and LLM provider.

---

## Usage

### Start Vox

```bash
just run

# Or directly
uv run main.py
uv run main.py --debug       # verbose logging to stderr
uv run main.py --version     # print version and exit
```

### Record

1. Press `F9` (or your configured hotkey) to start recording
2. Speak naturally
3. Press `F9` again to stop (toggle mode) or release the key (push-to-talk)
4. Text appears in the focused window

A thin red bar at the top of your screen indicates the mic is live. You'll hear a subtle beep on start and stop.

### History & Search

```bash
uv run main.py history                    # browse all transcriptions
uv run main.py history --search "query"   # full-text search (FTS5)
uv run main.py stats                      # usage statistics
```

### Web UI

```bash
uv run main.py web                        # launch Flask UI on port 8080
```

Browse your transcription history, search, and view individual sessions in a minimal web interface.

---

## Configuration

All settings live in `config.toml`:

```toml
[hotkey]
trigger = "f9"                    # f1-f20, ctrl, alt, shift, space, etc.
mode = "toggle"                   # "toggle" or "push_to_talk"

[audio]
device = ""                       # "" = default, or integer index
sample_rate = 16000
channels = 1

[transcription]
model = "large-v3"                # large-v3, large-v3-turbo, medium, small, tiny
device = "cuda"                   # "cuda" or "cpu"
compute_type = "int8"             # "int8", "float16", "float32"
language = ""                     # "" = auto-detect, or "en", "ml", "hi"
vad = true

[llm]
enabled = false                   # set true to enable LLM cleanup
base_url = "http://localhost:11434/v1"
api_key = ""                      # or set VOX_LLM_API_KEY env var
model = "phi3-mini"
timeout_seconds = 10

[output]
method = "xdotool"
fallback_to_clipboard = true
replace_strategy = "select"       # "select" (overwrite), "append", "skip"
replace_timeout_seconds = 5
replace_blacklist = ["Alacritty", "kitty", "st", "Vim", "nvim", "Code"]

[indicator]
style = "bar"
position = "top"
height = 4
width = "200"                     # "full" or pixel value
color = "#ff4444"
opacity = 0.9

[history]
enabled = true
db_path = "~/.local/share/vox/history.db"
max_entries = 10000

[sounds]
enabled = true
start_sound = "assets/start.wav"
stop_sound = "assets/stop.wav"
volume = 0.7
```

Full config reference: see [`config.toml.example`](config.toml.example).

---

## GPU / VRAM Notes

Tested on RTX 4050 (6GB VRAM):

| Component | VRAM | When Active |
|---|---|---|
| Whisper large-v3 (int8) | ~2.5GB | During transcription |
| Local LLM (e.g. phi3-mini) | ~2.3GB | During LLM cleanup only |
| Both simultaneously | Never | Pipeline is sequential |

When using a **local provider** (Ollama), Vox automatically passes `keep_alive=0` so the model unloads from VRAM immediately after inference, preventing contention with Whisper. **Cloud providers** use no local VRAM for the LLM.

**Low on VRAM?** Use a smaller Whisper model:

```toml
[transcription]
model = "medium"
```

Or disable LLM entirely:

```toml
[llm]
enabled = false
```

---

## Architecture

<img width="1440" height="1652" alt="image" src="https://github.com/user-attachments/assets/9e9161e6-f700-4180-93a9-80233d3bcfe6" />

**Threading model:** The critical path (transcribe → process → paste) runs inline on the Main thread for offline transcription. With streaming enabled, a dedicated `CWWorkerThread` handles GPU decode to avoid blocking the sounddevice callback.

Daemon threads:

- **HotkeyThread** — pynput listener, sends `start`/`stop`/`shutdown` to `control_queue`
- **IndicatorThread** — tkinter mainloop, polls `indicator_queue` every 50ms
- **LLMThread** — spawned per-session after paste, runs cleanup, dies on completion
- **StreamerThread** — polls transcriber for partial updates, drives indicator text
- **CWWorkerThread** — CarelessWhisper GPU decode loop (streaming only)

**State machine:** `IDLE → RECORDING → TRANSCRIBING → PROCESSING → PASTING → IDLE`

LLM cleanup runs asynchronously alongside `IDLE` — a new recording can start while cleanup is in progress.

**Streaming state machine:** `IDLE → RECORDING (streaming) → PASTING → IDLE` — transcription runs inline via the worker thread, bypassing TRANSCRIBING and PROCESSING states for lower latency.

---

## Benchmarks

Vox includes a WER (Word Error Rate) regression benchmark suite that tests multiple Whisper models against LibriSpeech test-clean samples. Results are streamed via HuggingFace — no full datasets downloaded.

### Running Benchmarks

```bash
# Quick benchmark (4 models)
just benchmark

# Full benchmark (12 models)
just benchmark-full

# Specific dataset
just benchmark-single clean
```

### Results

<img width="1366" height="789" alt="image" src="https://github.com/user-attachments/assets/9193493d-c499-4060-b586-933c0a51929c" />


The benchmark suite tests 12 Whisper models including `large-v3`, `large-v3-turbo`, `medium`, `small`, `tiny`, `base`, `large-v1`, `large-v2`, and distilled variants. Metrics tracked:

| Metric | Description |
|---|---|
| WER (mean/median/p95) | Word Error Rate across samples |
| Latency (mean/median/p95) | Transcription time per sample |
| LLM delta | WER(raw) − WER(cleaned) — positive = LLM helped |
| Paste success rate | Logged in SQLite stats table |

Results are appended to `tests/benchmark_results.jsonl` and charts are generated in `tests/benchmark_charts/`.

---

## Project Structure

```
vox/
├── config.toml              # Active configuration
├── config.toml.example      # Full config reference with docs
├── pyproject.toml           # Dependencies and metadata
├── justfile                 # Task runner commands
├── pyrightconfig.json       # Strict type checking config
├── .pre-commit-config.yaml  # ruff + pyright hooks
├── main.py                  # CLI entry point, logging, history/stats/web
│
├── vox/
│   ├── __init__.py          # __version__ = "0.1.0"
│   ├── config.py            # TOML loader, frozen dataclass settings
│   ├── pipeline.py          # State machine, thread wiring, main loop
│   ├── hotkey.py            # pynput listener, toggle + push-to-talk
│   ├── recorder.py          # sounddevice audio capture
│   ├── transcriber.py       # Factory function, delegates to backends
│   ├── transcriber_base.py  # ABC: shared interface and types
│   ├── transcriber_fw.py    # Faster-whisper backend (offline)
│   ├── transcriber_cw.py    # CarelessWhisper backend (streaming)
│   ├── transcriber_cw_decode.py  # Decoding options for CW
│   ├── processor.py         # Filler words + LLM cleanup
│   ├── output.py            # OutputBackend abstraction, xdotool + clipboard
│   ├── indicator.py         # tkinter thin bar overlay + live text
│   ├── sounds.py            # WAV loading, playback, tone generation
│   ├── history.py           # SQLite + FTS5 full-text search
│   ├── dictionary.py        # Personal vocabulary corrections
│   └── web.py               # Flask web UI for history
│
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_transcriber.py  # Factory function tests
│   ├── test_transcriber_base.py  # ABC contract tests
│   ├── test_transcriber_fw.py   # Faster-whisper backend tests
│   ├── test_transcriber_cw.py   # CarelessWhisper streaming tests
│   ├── test_pipeline.py     # Pipeline state machine tests
│   ├── test_indicator.py    # Indicator window tests
│   ├── test_integration.py  # End-to-end pipeline tests
│   ├── test_benchmark.py    # WER regression benchmarks
│   └── benchmark_charts/    # Generated PNG charts
│
├── third_party/
│   └── WhisperRT-Streaming/  # CarelessWhisper model (git submodule)
│
└── assets/
    ├── start.wav            # Recording start sound cue
    └── stop.wav             # Recording stop sound cue
```

---

## Development

### Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + Whisper large-v3 |
| Audio capture | [sounddevice](https://python-sounddevice.readthedocs.io/) |
| Hotkeys | [pynput](https://pynput.readthedocs.io/) |
| LLM client | [openai](https://github.com/openai/openai-python) (universal OpenAI-compatible client) |
| Clipboard | [pyperclip](https://github.com/asweigart/pyperclip) |
| Web UI | [Flask](https://flask.palletsprojects.com/) |
| Database | SQLite + FTS5 |
| Linting/formatting | [ruff](https://docs.astral.sh/ruff/) |
| Type checking | [pyright](https://github.com/microsoft/pyright) |
| Testing | [pytest](https://docs.pytest.org/) + [jiwer](https://github.com/jitsi/jiwer) |
| Task runner | [just](https://just.systems/) |

### Common Commands

```bash
just run          # start Vox
just debug        # start with verbose logging
just test         # run all tests
just lint         # ruff check
just fmt          # ruff format (auto-fix)
just typecheck    # pyright strict check
just benchmark    # quick WER regression test
just deadcode     # vulture dead code detection
just web          # launch Flask history UI
just setup-streaming  # install CarelessWhisper streaming deps
```

### Pre-commit Hooks

Hooks run `ruff` and `pyright` automatically on every commit. A commit that fails either check is rejected.

```bash
just install-hooks   # first-time setup
```

### Testing

```bash
uv sync --group dev              # install dev dependencies

uv run pytest                    # run all tests
uv run pytest tests/ -v --tb=short  # verbose
uv run pytest tests/test_transcriber.py  # single module

# Marked tests
uv run pytest -m slow            # integration tests (real audio)
uv run pytest -m benchmark       # benchmark suite
```

Test layers:

- **Unit tests** — each module in isolation, mocked audio/GPU/xdotool (~3,000+ lines)
- **Integration tests** — real WhisperModel with real LibriSpeech audio streaming
- **Benchmark tests** — WER regression across 12 Whisper models with chart generation

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Hotkey not working | Make sure Vox is running as your user, not root. Check for hotkey conflicts with your WM. |
| Paste not working in some apps | Electron apps intercept keyboard events. Vox falls back to clipboard automatically — press `Ctrl+V`. |
| Text replacement misfires in terminal/Vim | Add the window class to `replace_blacklist` in `config.toml`. Find class: `xdotool getactivewindow getwindowclassname` |
| LLM is slow on first use (local) | Vox sends a warmup ping on startup to pre-load the model. Wait a few seconds after launch before your first recording. Cloud providers don't have this issue. |
| Out of VRAM | Use `model = "medium"` or `model = "small"`, or set `[llm] enabled = false` |
| Mic not found | Check `arecord -l` for available devices. Set `device = <index>` in `[audio]` section. |

---

## Roadmap

See [`PLAN.md`](PLAN.md) for the full feature roadmap and build order.

### Planned

- [ ] Idea expansion mode (LLM expands rough voice notes into structured writeups)
- [ ] Custom vocabulary / personal dictionary (post-correct Whisper misheard words)
- [ ] Snippets / voice shortcuts (say a cue, expand to full text)
- [ ] Transcription search UI (minimal GUI or rofi integration)
- [ ] Per-app profiles (different post-processing per active window)
- [ ] Multilingual support (Malayalam, Hindi — Whisper handles this natively)
- [ ] WhisperX two-mode architecture (faster-whisper for quick dictation, WhisperX for long-form with word-level timestamps)

---

## License

MIT
