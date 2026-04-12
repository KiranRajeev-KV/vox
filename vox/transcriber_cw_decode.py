"""Decoding options for CarelessWhisper streaming.

Mirrors whisper_rt/streaming_decoding.DecodingOptions from the original repo.
See: https://github.com/tomer9080/WhisperRT-Streaming/blob/main/whisper_rt/streaming_decoding.py
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class DecodingOptions:
    """Options passed to CarelessWhisper model.decode()."""

    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"
    # language that the audio is in; uses detected language if None
    language: str | None = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: int | None = None
    best_of: int | None = None
    beam_size: int | None = None
    patience: float | None = None
    length_penalty: float | None = None

    prompt: str | list[int] | None = None
    prefix: str | list[int] | None = None

    suppress_tokens: str | Iterable[int] | None = "-1"
    suppress_blank: bool = True

    without_timestamps: bool = True
    max_initial_timestamp: float | None = 1.0

    fp16: bool = False

    # Advisor & Streaming params
    advised: bool = False
    attentive_advisor: bool = False
    use_sa: bool = False
    ctx: int = 3
    gran: int = 15
    pad_audio_features: bool = True
    single_frame_mel: bool = True
    pad_input: bool = False
    look_ahead_blocks: int = 0
    maximal_seconds_context: int = 30

    use_kv_cache: bool = False
    use_ca_kv_cache: bool = False

    # streaming decoding args
    stream_decode: bool = True
    tokens_per_frame: int = 2
    n_tokens_look_back: int = 2
    streaming_timestamps: bool = True
    wait_for_all: bool = False
    force_first_tokens_timestamps: bool = False

    # localagreement params
    localagreement: bool = False
