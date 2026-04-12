# PLAN.md — Vox

A local, offline, GPU-accelerated voice-to-text tool for Arch Linux.
Press a key. Speak. Text appears wherever your cursor is.

---

## Feature Set

### Phase 1 — Core (get it working) — ✅ COMPLETE

- [x] Global hotkey listener (default: F9, configurable)
- [x] Both toggle mode and push-to-talk mode (config flag)
- [x] Audio capture from microphone (sounddevice)
- [x] Transcription via faster-whisper (large-v3, int8, CUDA)
- [x] Auto-paste into active window via xdotool (OutputBackend)
- [x] Clipboard fallback if paste fails (pyperclip)
- [x] Sound cues on start and stop
- [x] Thin bar overlay indicator while recording (tkinter, top of screen)

### Phase 2 — Quality — ✅ COMPLETE

- [x] Filler word removal (regex, configurable word list in config.toml)
- [x] Active window class detection (xdotool getwindowclassname)
- [x] Replace strategy: `replace_strategy` config sets default (`select` or `append`); blacklist overrides to `skip`; stale timeout (>5s) also skips
- [x] SQLite transcription history (every session saved with timestamp, app context, word count)
- [x] FTS5 full-text search over history
- [x] VAD via faster-whisper's bundled Silero VAD (`vad_filter=True` in transcribe call)
- [x] Desktop notifications via notify-send (fallback events, errors)

### Phase 3 — AI Layer — ✅ COMPLETE

- [x] Grammar cleanup via any OpenAI-compatible LLM endpoint (async, `openai` client)
- [x] Optimistic paste — paste raw text immediately, replace with cleaned text when LLM finishes
- [x] Replace respects stale timeout (>5s since paste → skip), then blacklist check, then executes strategy
- [x] Warmup ping on app start when `base_url` is localhost (with `keep_alive=0`)

### Phase 4 — Polish & Testing — ✅ COMPLETE

- [x] Unit tests for all modules (pytest, mocked audio + xdotool)
- [x] Integration tests — wav file through pipeline, assert WER within threshold
- [x] Regression benchmarks — LibriSpeech test-clean, 50 samples, WER baseline stored
- [x] Personal voice fixture recordings in tests/fixtures/
- [x] History viewer (CLI: search, list recent, word count stats)
- [x] Usage stats (total words dictated, sessions, estimated time saved)
- [x] Graceful shutdown (drain queues before exit)
- [x] Full error handling pass

### Phase 5 — Dual-Backend Transcription — ✅ COMPLETE

- [x] Factory-based transcriber architecture (`make_transcriber()`)
- [x] `TranscriberBase` ABC with shared interface and types
- [x] `FasterWhisperTranscriber` — offline backend (large-v3, int8, CUDA)
- [x] `CarelessWhisperTranscriber` — streaming backend (causal, real-time)
- [x] Real-time partial text display on indicator bar (40px expanded)
- [x] Thread-safe audio buffer with worker thread for GPU decode
- [x] Streaming config section ([streaming]) with model, chunk_size_ms, beam_size
- [x] Setup script for CarelessWhisper dependencies and HF auth
- [x] Offline fallback when streaming fails to start
- [x] Personal dictionary for vocabulary correction

### Maybe Later

- [ ] Idea expansion mode (LLM expands rough voice notes into structured writeups)
- [ ] Snippets / voice shortcuts (say a cue, expand to full text)
- [ ] Transcription search UI (minimal GUI or rofi integration)
- [ ] Per-app profiles (different post-processing per active window)
- [ ] Multilingual support (Malayalam, Hindi — Whisper handles this natively)
- [ ] WhisperX two-mode architecture — faster-whisper for quick dictation, WhisperX
      for long-form capture (journalling, idea dumps) with word-level timestamps,
      natural pause detection, and richer LLM structuring

---

## Build Order

### Phase 1 Steps

1. Project structure + `uv init`, `pyproject.toml`
2. `justfile` with common commands (run, test, lint, fmt, typecheck, benchmark)
3. `pyrightconfig.json` — strict mode
4. `.pre-commit-config.yaml` — ruff + pyright hooks
5. `config.toml` with full defaults
6. `vox/__init__.py` with `__version__ = "0.1.0"`
7. `vox/config.py` — toml loader, settings object
8. `vox/recorder.py` — mic capture into numpy buffer
9. `vox/transcriber.py` — faster-whisper wrapper
10. `vox/hotkey.py` — pynput listener, both modes
11. `vox/sounds.py` — start/stop cues
12. `vox/indicator.py` — thin bar overlay
13. `vox/output.py` — OutputBackend interface, XdotoolBackend, clipboard fallback
14. `vox/pipeline.py` — thread wiring, queues, state machine
15. `main.py` — entry point, logging setup (thread-name format, file + stderr)

### Phase 2 Steps

16. `vox/processor.py` — filler word removal (regex)
17. Window class detection in `output.py`, replace strategy routing
18. `vox/history.py` — SQLite schema, write on each session
19. Enable VAD in `vox/transcriber.py` via `vad_filter=True` in faster-whisper transcribe call
20. notify-send integration into output.py

### Phase 3 Steps

21. OpenAI-compatible client in `vox/processor.py`, conditional `keep_alive=0` for localhost
22. Async LLM call + optimistic paste replace logic in output.py
23. Stale timeout check + blacklist check before replace execution in output.py
24. Warmup ping in main.py (with `keep_alive=0`)

### Phase 4 Steps

25. `tests/` structure, pytest setup in `pyproject.toml`
26. Unit tests — processor, transcriber, output, history
27. Integration test — full pipeline with wav fixture
28. Regression benchmark script — LibriSpeech test-clean streaming, WER computation
29. Personal voice fixtures recorded and committed to tests/fixtures/
30. CLI history viewer (argparse subcommands)
31. Stats commands
32. Graceful shutdown handler (signal.SIGINT)
33. Full error handling pass

---

## Testing Strategy

### Datasets

Never download full datasets. Always stream and take a fixed slice.

| Dataset | Use | How to access |
|---|---|---|
| LibriSpeech test-clean | WER baseline, clean studio speech, standard benchmark | HuggingFace streaming, 50 samples |
| Common Voice (Mozilla) | Accent diversity, more natural speech | HuggingFace streaming, filter by language |
| FLEURS (Google) | Multilingual — English, Malayalam, Hindi | HuggingFace streaming |
| Personal voice recordings | Real-world regression, your actual voice and environment | Record manually, commit to tests/fixtures/ |

Personal recordings are the most valuable for catching real issues. Record 20–30
sentences covering: slow deliberate dictation, fast casual speech, technical terms,
mixed language, background noise conditions.

### Metrics

| Metric | Description | Tool |
|---|---|---|
| WER (raw) | Whisper output vs ground truth | jiwer |
| WER (post-LLM) | Cleaned output vs ground truth | jiwer |
| LLM delta | WER(raw) - WER(clean). Positive = LLM helped, negative = LLM hurt | computed |
| Whisper latency | Inference time for transcriber.transcribe() | time.perf_counter() |
| Full pipeline latency | Hotkey trigger → paste complete | time.perf_counter() |
| Paste success rate | xdotool success vs clipboard fallback ratio | SQLite stats table |

### Test layers

**Unit tests** (`tests/test_*.py`)
Each module tested in isolation. No real GPU, no real audio, no real xdotool.
- `test_processor.py` — filler word removal with known inputs/outputs
- `test_output.py` — replace strategy routing with mocked window class names
- `test_history.py` — SQLite writes, FTS5 search queries
- `test_transcriber.py` — load a fixture wav, assert transcript is non-empty

**Integration tests**
Feed a real wav file through the full pipeline (transcriber → processor → output).
No hotkey, no real paste. Assert WER on known fixture is within threshold (e.g. <10%).

**Regression / benchmark**
Stream 50 samples from LibriSpeech test-clean. Run through Vox transcriber.
Compute WER. Compare against stored baseline in `tests/baseline_wer.json`.
Fail if WER degrades by more than 2 percentage points.
Run this after any change to Whisper model config.

### pytest setup (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[dependency-groups]
dev = ["pytest", "jiwer", "datasets", "ruff", "pyright", "pre-commit"]
```

Install dev dependencies: `uv sync --group dev`

### justfile

```makefile
run:
    uv run main.py

debug:
    uv run main.py --debug

test:
    uv run pytest

lint:
    uv run ruff check vox/ tests/

fmt:
    uv run ruff format vox/ tests/

typecheck:
    uv run pyright .

benchmark:
    uv run pytest tests/test_benchmark.py -v

deadcode:
    uv run vulture vox/ tests/ --min-confidence 80

setup-streaming:
    bash scripts/setup_streaming.sh

install-hooks:
    uv run pre-commit install
```

### pyrightconfig.json

```json
{
  "pythonVersion": "3.12",
  "typeCheckingMode": "strict",
  "venvPath": ".venv",
  "include": ["vox", "main.py"],
  "exclude": ["tests"],
  "reportMissingTypeStubs": false,
  "reportUnknownMemberType": false,
  "reportUnknownParameterType": false,
  "reportUnknownVariableType": false
}
```

---

## Config Schema (config.toml)

```toml
[hotkey]
trigger = "f9"
mode = "toggle"              # "toggle" or "push_to_talk"

[audio]
device = ""              # "" = system default
sample_rate = 16000
channels = 1

[transcription]
model = "large-v3"
device = "cuda"
compute_type = "int8"
language = ""            # "" = auto-detect, or "en", "ml", "hi"
vad = true

[processing]
filler_words = ["uh", "um", "like", "you know", "basically", "literally", "actually", "right"]

[llm]
enabled = true
base_url = "http://localhost:11434/v1"   # any OpenAI-compatible endpoint
api_key = ""                              # or set VOX_LLM_API_KEY env var
model = "phi3-mini"                       # whatever the provider calls it
timeout_seconds = 10
prompt = """
Clean up the following transcribed speech. Fix grammar, punctuation, and sentence structure.
Remove any remaining filler words. Keep the meaning and tone exactly the same.
Return only the cleaned text, nothing else.
"""

# Example providers:
# Ollama (local):   base_url = "http://localhost:11434/v1", model = "phi3-mini"
# Groq:             base_url = "https://api.groq.com/openai/v1", model = "llama-3.1-8b-instant"
# OpenRouter:       base_url = "https://openrouter.ai/api/v1", model = "google/gemini-flash-1.5"
# OpenAI:           base_url = "https://api.openai.com/v1", model = "gpt-4o-mini"

[indicator]
style = "bar"
position = "top"
height = 4
width = "full"           # "full" or pixel value e.g. 200
color = "#ff4444"
opacity = 0.9

[output]
method = "xdotool"           # X11 only
fallback_to_clipboard = true
notify_on_paste = false
notify_on_fallback = true
replace_strategy = "select"  # "select" or "append" — overridden to "skip" if window in blacklist
replace_timeout_seconds = 5  # skip replace if LLM takes longer than this after paste
replace_blacklist = [
    "Alacritty", "kitty", "st", "urxvt", "xterm",
    "Vim", "nvim", "Code", "Emacs"
]

[history]
enabled = true
db_path = "~/.local/share/vox/history.db"
max_entries = 10000

[sounds]
enabled = true
start_sound = "assets/start.wav"
stop_sound = "assets/stop.wav"
volume = 0.7

[streaming]
enabled = false
model = "small"           # base, small, large-v2
chunk_size_ms = 300       # decode frequency
beam_size = 0             # 0 = greedy, 1+ = beam search

[dictionary]
db_path = "~/.local/share/vox/dictionary.db"
```

---

## SQLite Schema

```sql
CREATE TABLE sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_text      TEXT NOT NULL,
    clean_text    TEXT,
    duration_ms   INTEGER,
    word_count    INTEGER,
    app_context   TEXT,
    language      TEXT,
    model_used    TEXT
);

CREATE VIRTUAL TABLE sessions_fts USING fts5(
    raw_text,
    clean_text,
    content='sessions',
    content_rowid='id'
);

CREATE TABLE stats (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    total_sessions    INTEGER DEFAULT 0,
    total_words       INTEGER DEFAULT 0,
    total_duration_ms INTEGER DEFAULT 0,
    first_used_at     DATETIME,
    last_used_at      DATETIME
);
```

---

## State Machine

### Offline mode:

```
IDLE
  │  (hotkey pressed)
  ▼
RECORDING
  │  (hotkey pressed again / released)
  ▼
TRANSCRIBING
  │
  ▼
PROCESSING  ──────────────────────────────────┐
  │                                           │
  ▼                                     LLM_CLEANING (async)
PASTING                                       │
  │                                    (check blacklist)
  ▼                                           │
IDLE ◄──────── replace / skip / append ───────┘
```

### Streaming mode:

```
IDLE
  │  (hotkey pressed)
  ▼
RECORDING ◄──── feed_chunk() from sounddevice callback
  │              (CWWorkerThread decodes in background)
  │              (partial text shown on indicator)
  │  (hotkey pressed again)
  ▼
PASTING
  │
  ▼
IDLE
```

---

## OutputBackend Interface

```python
class OutputBackend:
    def type_text(self, text: str) -> bool: ...
    def select_left(self, n_chars: int) -> bool: ...
    def get_active_window_class(self) -> str: ...

class XdotoolBackend(OutputBackend): ...   # X11, only backend
```

---

## System Dependencies

```bash
# Required
sudo pacman -S xdotool libnotify
```

## Python Setup (uv)

```bash
uv init vox
cd vox
uv add faster-whisper sounddevice pynput pyperclip openai
uv add --group dev pytest jiwer datasets ruff pyright pre-commit
uv add --extra streaming   # optional: CarelessWhisper streaming support
uv run main.py
```

### Streaming Setup (Optional)

```bash
git submodule update --init   # fetch WhisperRT-Streaming
just setup-streaming          # install CUDA 13 deps, login to HF
```
