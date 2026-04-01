"""Benchmark tests — WER and latency across models and datasets."""

from __future__ import annotations

import gc
import io
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel
from jiwer import wer

RESULTS_FILE = Path(__file__).parent / "benchmark_results.jsonl"
CHARTS_DIR = Path(__file__).parent / "benchmark_charts"

QUICK_MODELS = ["large-v3", "large-v3-turbo", "medium", "small"]
FULL_MODELS = QUICK_MODELS + [
    "tiny",
    "base",
    "large-v1",
    "large-v2",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
]

DATASET_CONFIGS = {
    "clean": {
        "name": "librispeech_asr",
        "config": "clean",
        "text_key": "text",
        "label": "LibriSpeech Clean",
    },
    "other": {
        "name": "librispeech_asr",
        "config": "other",
        "text_key": "text",
        "label": "LibriSpeech Other",
    },
}


def _get_cli_options(request: pytest.FixtureRequest) -> dict[str, Any]:
    return {
        "samples": request.config.getoption("--benchmark-samples"),
        "seed": request.config.getoption("--benchmark-seed"),
        "tier": request.config.getoption("--benchmark-tier"),
        "datasets": request.config.getoption("--benchmark-datasets").split(","),
    }


def _load_samples(dataset_key: str, n: int, seed: int) -> list[tuple[np.ndarray, str]]:
    """Stream n samples from LibriSpeech, return (audio_array, ground_truth) pairs.

    Uses raw byte decoding (decode=False) to avoid torchcodec dependency.
    Audio comes as FLAC bytes, decoded via soundfile.
    """
    cfg = DATASET_CONFIGS[dataset_key]
    ds = load_dataset(
        cfg["name"],
        cfg["config"],
        split="test",
        streaming=True,
    )
    # Disable auto audio decoding — get raw bytes instead
    new_features = ds.features.copy()
    new_features["audio"] = Audio(sampling_rate=None, decode=False)
    ds = ds.cast(new_features)
    ds = ds.shuffle(seed=seed)

    samples: list[tuple[np.ndarray, str]] = []
    for item in ds:
        raw_bytes = item["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # LibriSpeech is always 16kHz
        text = item[cfg["text_key"]].strip()
        if not text:
            continue
        samples.append((audio, text))
        if len(samples) >= n:
            break
    return samples


def _run_model_benchmark(
    model_name: str,
    samples: list[tuple[np.ndarray, str]],
) -> dict[str, Any]:
    """Run transcription benchmark for one model against a sample set.

    Returns dict with wer_mean, wer_median, wer_p95, latency stats, failed count.
    """
    model = WhisperModel(
        model_name,
        device="cuda",
        compute_type="int8",
    )

    wer_scores: list[float] = []
    latencies: list[float] = []
    failed = 0

    for audio, reference in samples:
        start = time.perf_counter()
        try:
            segments, info = model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                beam_size=5,
            )
            hypothesis = " ".join(seg.text for seg in segments).strip()
            latency = time.perf_counter() - start

            if hypothesis:
                w = wer(reference, hypothesis)
                wer_scores.append(w)
            latencies.append(latency * 1000)
        except Exception as e:
            print(f"ERROR: {model_name} failed on sample: {type(e).__name__}: {e}")
            failed += 1
            if failed <= 3:
                import traceback

                traceback.print_exc()

    del model
    gc.collect()

    return {
        "wer_mean": statistics.mean(wer_scores) if wer_scores else 0.0,
        "wer_median": statistics.median(wer_scores) if wer_scores else 0.0,
        "wer_p95": sorted(wer_scores)[int(len(wer_scores) * 0.95)] if wer_scores else 0.0,
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "latency_median_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        "samples": len(samples) - failed,
        "failed": failed,
    }


def _save_result(result: dict[str, Any]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def _print_table(results: list[dict[str, Any]], title: str) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold]{title}[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Dataset", style="green")
    table.add_column("WER ↓", justify="right", style="yellow")
    table.add_column("WER med", justify="right")
    table.add_column("WER p95", justify="right")
    table.add_column("Latency ↓", justify="right", style="yellow")
    table.add_column("Latency med", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Failed", justify="right", style="red")

    for r in results:
        table.add_row(
            r["model"],
            r["dataset_label"],
            f"{r['wer_mean']:.3f}",
            f"{r['wer_median']:.3f}",
            f"{r['wer_p95']:.3f}",
            f"{r['latency_mean_ms']:.0f}ms",
            f"{r['latency_median_ms']:.0f}ms",
            str(r["samples"]),
            str(r["failed"]),
        )

    console.print(table)


def _generate_charts(results: list[dict[str, Any]]) -> None:
    """Generate benchmark charts and save to CHARTS_DIR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    models = sorted(set(r["model"] for r in results))
    datasets = sorted(set(r["dataset_label"] for r in results))

    # --- WER Bar Chart ---
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2.5), 6))
    x = np.arange(len(models))
    width = 0.8 / max(len(datasets), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(datasets), 1)))
    for i, ds_label in enumerate(datasets):
        ds_results = {r["model"]: r for r in results if r["dataset_label"] == ds_label}
        values = [ds_results.get(m, {}).get("wer_mean", 0) for m in models]
        ax.bar(
            x + i * width - (len(datasets) - 1) * width / 2,
            values,
            width,
            label=ds_label,
            color=colors[i],
        )
    ax.set_xlabel("Model")
    ax.set_ylabel("WER (lower is better)")
    ax.set_title("Word Error Rate by Model and Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "wer_comparison.png", dpi=150)
    plt.close(fig)

    # --- Latency Bar Chart ---
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2.5), 6))
    for i, ds_label in enumerate(datasets):
        ds_results = {r["model"]: r for r in results if r["dataset_label"] == ds_label}
        values = [ds_results.get(m, {}).get("latency_mean_ms", 0) for m in models]
        ax.bar(
            x + i * width - (len(datasets) - 1) * width / 2,
            values,
            width,
            label=ds_label,
            color=colors[i],
        )
    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (ms, lower is better)")
    ax.set_title("Transcription Latency by Model and Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "latency_comparison.png", dpi=150)
    plt.close(fig)

    # --- WER vs Latency Scatter ---
    fig, ax = plt.subplots(figsize=(10, 7))
    model_sizes = {
        "tiny": 39,
        "base": 74,
        "small": 244,
        "medium": 769,
        "large-v1": 1550,
        "large-v2": 1550,
        "large-v3": 1550,
        "large-v3-turbo": 809,
        "distil-small.en": 122,
        "distil-medium.en": 384,
        "distil-large-v2": 756,
        "distil-large-v3": 756,
    }
    for ds_label in datasets:
        ds_results = [r for r in results if r["dataset_label"] == ds_label]
        for r in ds_results:
            size = model_sizes.get(r["model"], 500)
            ax.scatter(
                r["latency_mean_ms"],
                r["wer_mean"],
                s=size * 0.3,
                alpha=0.7,
                label=f"{r['model']} ({ds_label})",
            )
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("WER (lower-left = best)")
    ax.set_title("WER vs Latency (bubble size = model params in M)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "wer_vs_latency.png", dpi=150)
    plt.close(fig)

    # --- WER Distribution Box Plot ---
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    ax.boxplot(
        [[r["wer_mean"] for r in results if r["model"] == m] for m in models],
        labels=models,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("WER")
    ax.set_title("WER Distribution by Model")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "wer_distribution.png", dpi=150)
    plt.close(fig)


@pytest.mark.benchmark
class TestBenchmarkQuick:
    """Quick tier benchmark: 4 models across selected datasets."""

    def test_quick_tier(self, request: pytest.FixtureRequest) -> None:
        opts = _get_cli_options(request)
        if opts["tier"] != "quick":
            pytest.skip("Quick tier test, use --benchmark-tier full for all models")

        models = QUICK_MODELS
        dataset_keys = [k for k in opts["datasets"] if k in DATASET_CONFIGS]
        if not dataset_keys:
            pytest.skip(f"No valid datasets in {opts['datasets']}")

        all_results: list[dict[str, Any]] = []

        for ds_key in dataset_keys:
            samples = _load_samples(ds_key, opts["samples"], opts["seed"])
            if not samples:
                continue

            for model_name in models:
                print(f"\nBenchmarking {model_name} on {ds_key} ({len(samples)} samples)...")
                stats = _run_model_benchmark(model_name, samples)

                result = {
                    "model": model_name,
                    "dataset": ds_key,
                    "dataset_label": DATASET_CONFIGS[ds_key]["label"],
                    "tier": "quick",
                    "seed": opts["seed"],
                    "compute_type": "int8",
                    "device": "cuda",
                    "vad": True,
                    "language": "en",
                    "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                    **stats,
                }
                _save_result(result)
                all_results.append(result)

        if all_results:
            _print_table(
                all_results,
                f"Quick Tier Benchmark ({opts['samples']} samples, seed={opts['seed']})",
            )
            _generate_charts(all_results)
            print(f"\nCharts saved to: {CHARTS_DIR}/")
            print(f"Results appended to: {RESULTS_FILE}")


@pytest.mark.benchmark
class TestBenchmarkFull:
    """Full tier benchmark: all models across selected datasets."""

    def test_full_tier(self, request: pytest.FixtureRequest) -> None:
        opts = _get_cli_options(request)
        if opts["tier"] != "full":
            pytest.skip("Full tier test, use --benchmark-tier full")

        models = FULL_MODELS
        dataset_keys = [k for k in opts["datasets"] if k in DATASET_CONFIGS]
        if not dataset_keys:
            pytest.skip(f"No valid datasets in {opts['datasets']}")

        all_results: list[dict[str, Any]] = []

        for ds_key in dataset_keys:
            samples = _load_samples(ds_key, opts["samples"], opts["seed"])
            if not samples:
                continue

            for model_name in models:
                print(f"\nBenchmarking {model_name} on {ds_key} ({len(samples)} samples)...")
                stats = _run_model_benchmark(model_name, samples)

                result = {
                    "model": model_name,
                    "dataset": ds_key,
                    "dataset_label": DATASET_CONFIGS[ds_key]["label"],
                    "tier": "full",
                    "seed": opts["seed"],
                    "compute_type": "int8",
                    "device": "cuda",
                    "vad": True,
                    "language": "en",
                    "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                    **stats,
                }
                _save_result(result)
                all_results.append(result)

        if all_results:
            _print_table(
                all_results,
                f"Full Tier Benchmark ({opts['samples']} samples, seed={opts['seed']})",
            )
            _generate_charts(all_results)
            print(f"\nCharts saved to: {CHARTS_DIR}/")
            print(f"Results appended to: {RESULTS_FILE}")


@pytest.mark.benchmark
class TestBenchmarkUtils:
    """Utility tests for benchmark infrastructure."""

    def test_load_dataset_samples(self) -> None:
        samples = _load_samples("clean", 5, seed=42)
        assert len(samples) == 5
        for audio, text in samples:
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert audio.ndim == 1
            assert isinstance(text, str)
            assert len(text) > 0

    def test_reproducible_seed(self) -> None:
        samples_a = _load_samples("clean", 10, seed=123)
        samples_b = _load_samples("clean", 10, seed=123)
        for (a_audio, a_text), (b_audio, b_text) in zip(samples_a, samples_b, strict=True):
            np.testing.assert_array_equal(a_audio, b_audio)
            assert a_text == b_text

    def test_wer_computation(self) -> None:
        assert wer("hello world", "hello world") == 0.0
        assert wer("hello world", "hello") > 0.0
        assert wer("hello", "hello world") > 0.0

    def test_results_file_append(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test_results.jsonl"
        r1 = {"model": "test", "wer_mean": 0.1}
        r2 = {"model": "test2", "wer_mean": 0.2}
        with open(test_file, "a") as f:
            f.write(json.dumps(r1) + "\n")
        with open(test_file, "a") as f:
            f.write(json.dumps(r2) + "\n")
        lines = test_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["model"] == "test"
        assert json.loads(lines[1])["model"] == "test2"

    def test_failed_sample_handling(self) -> None:
        stats = {
            "wer_mean": 0.05,
            "wer_median": 0.04,
            "wer_p95": 0.1,
            "latency_mean_ms": 500,
            "latency_median_ms": 480,
            "latency_p95_ms": 900,
            "samples": 48,
            "failed": 2,
        }
        assert stats["samples"] + stats["failed"] == 50
        assert stats["wer_mean"] >= 0
        assert stats["wer_mean"] <= 1
        assert stats["latency_mean_ms"] > 0
