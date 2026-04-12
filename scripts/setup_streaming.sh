#!/usr/bin/env bash
# setup_streaming.sh — Set up CarelessWhisper streaming dependencies for Vox.
#
# Usage:
#   bash scripts/setup_streaming.sh
#
# What it does:
#   1. Clones the WhisperRT-Streaming repo into third_party/
#   2. Installs streaming Python deps via uv
#   3. Prompts for Hugging Face login (required for model weights)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

echo "=== Vox Streaming Setup ==="
echo ""

# ── Step 1: Clone WhisperRT-Streaming ─────────────────────────────────
WHISPER_RT_DIR="$PROJECT_DIR/third_party/WhisperRT-Streaming"

if [ -d "$WHISPER_RT_DIR" ]; then
    log_info "whisper_rt already exists at third_party/WhisperRT-Streaming"
    read -rp "Update existing clone? (y/N) " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        cd "$WHISPER_RT_DIR"
        git pull --ff-only
        cd "$PROJECT_DIR"
        log_info "Updated whisper_rt"
    fi
else
    echo "Cloning WhisperRT-Streaming into third_party/..."
    mkdir -p "$PROJECT_DIR/third_party"
    git clone https://github.com/tomer9080/WhisperRT-Streaming "$WHISPER_RT_DIR"
    log_info "Cloned whisper_rt"
fi

# ── Step 2: Install Python deps ───────────────────────────────────────
echo ""
echo "Installing streaming Python dependencies via uv..."
uv sync --extra streaming
log_info "Installed torch, pytorch-lightning, huggingface-hub, pyaudio, tiktoken"

# ── Step 3: Hugging Face login ────────────────────────────────────────
echo ""
echo "The CarelessWhisper model weights are hosted on Hugging Face"
echo "and require authentication."
echo ""
if command -v hf &>/dev/null; then
    HF_CMD="hf auth login"
elif command -v huggingface-cli &>/dev/null; then
    HF_CMD="huggingface-cli login"
else
    log_warn "Neither 'hf' nor 'huggingface-cli' found. Install huggingface-hub first."
    exit 1
fi

echo "Run this command to log in:"
echo "  $HF_CMD"
echo ""
read -rp "Log in now? (y/N) " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    $HF_CMD
    log_info "Hugging Face login complete"
else
    log_warn "Skipped login. Run '$HF_CMD' before using streaming mode."
fi

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "To enable streaming:"
echo "  1. Set streaming.enabled = true in config.toml"
echo "  2. Run: just run"
echo ""
echo "In ~/.local/share/vox/vox.log, look for:"
echo "  'CarelessWhisper streaming session started'"
echo ""
