###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Memory and timing benchmark for TraceDiff comparing mi300 vs h100 traces.

Measures peak Python allocations (tracemalloc) and OS RSS (psutil) at each
major TraceDiff stage (tree build, merge_trees, generate_diff_stats).

Usage:
    python tests/benchmark_tracediff.py
    python tests/benchmark_tracediff.py --model Qwen_Qwen1.5-0.5B-Chat__1016005
    python tests/benchmark_tracediff.py --model all
"""

import argparse
import gc
import gzip
import json
import time
import tracemalloc

import orjson
import psutil

# Pre-import all TraceLens modules so import overhead doesn't skew measurements
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TraceDiff.trace_diff import TraceDiff

TRACES_DIR = "tests/traces"

MODELS = [
    "Falconsai_nsfw_image_detection__1016002",
    "gaunernst_bert-small-uncased__1016001",
    "Qwen_Qwen1.5-0.5B-Chat__1016005",
    "google_owlv2-large-patch14-ensemble__1016001",
    "facebook_timesformer-base-finetuned-k400__1016002",
    "Wan-AI_Wan2.1-T2V-1.3B-Diffusers__1016009",
]

_proc = psutil.Process()


def _rss_mb() -> float:
    gc.collect()
    return _proc.memory_info().rss / 1024**2


def measure(label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), record timing + memory, return result."""
    gc.collect()
    rss_before = _rss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()

    result = fn(*args, **kwargs)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _cur, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_mb()

    row = dict(
        label=label,
        elapsed_ms=elapsed_ms,
        peak_py_mb=peak_py / 1024**2,
        rss_delta_mb=rss_after - rss_before,
        rss_after_mb=rss_after,
    )
    print(
        f"  {label:<46} {elapsed_ms:>8.0f} ms  "
        f"peak_py={peak_py/1024**2:>7.1f} MB  "
        f"ΔRSS={rss_after-rss_before:>+7.1f} MB  "
        f"RSS={rss_after:>7.1f} MB"
    )
    return result, row


def load_trace(path: str):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    data = orjson.loads(raw)
    return data["traceEvents"]


def run_model(model: str) -> dict:
    mi300_path = f"{TRACES_DIR}/mi300/{model}.json.gz"
    h100_path = f"{TRACES_DIR}/h100/{model}.json.gz"

    print(f"\n{'='*80}")
    print(f"Model: {model}")
    print(f"  mi300: {mi300_path}")
    print(f"  h100 : {h100_path}")
    print()

    # Load traces (not measured — we want pure TraceDiff cost)
    events_mi300 = load_trace(mi300_path)
    events_h100 = load_trace(h100_path)
    n_mi300 = len(events_mi300)
    n_h100 = len(events_h100)
    print(f"  Events: mi300={n_mi300:,}  h100={n_h100:,}")
    print()

    rows = []

    # Stage 1: Build TraceToTree for both traces
    _, r1 = measure("1. TraceToTree mi300", TraceToTree, events_mi300)
    rows.append(r1)
    tree_mi300 = TraceToTree(events_mi300)

    _, r2 = measure("2. TraceToTree h100", TraceToTree, events_h100)
    rows.append(r2)
    tree_h100 = TraceToTree(events_h100)

    # Stage 2: build_tree for both
    def _build(tree):
        tree.build_tree()
        return tree

    _, r3 = measure("3. build_tree mi300", _build, tree_mi300)
    rows.append(r3)
    tree_mi300.build_tree()

    _, r4 = measure("4. build_tree h100", _build, tree_h100)
    rows.append(r4)
    tree_h100.build_tree()

    # Stage 3: TraceDiff __init__ (calls merge_trees internally)
    def _tracediff_init():
        return TraceDiff(tree_mi300, tree_h100)

    td, r5 = measure("5. TraceDiff.__init__ (merge_trees)", _tracediff_init)
    rows.append(r5)

    # Stage 4: generate_diff_stats
    def _gen_diff_stats():
        try:
            return td.generate_diff_stats()
        except (KeyError, Exception) as e:
            # Empty diff stats (no comparable nodes) — pre-existing edge case
            print(f"     generate_diff_stats: skipped ({type(e).__name__}: {e})")
            return None

    _, r6 = measure("6. generate_diff_stats", _gen_diff_stats)
    rows.append(r6)

    # Summary
    sep = "─" * 90
    print(f"\n  {'Stage':<46} {'ms':>8}    {'peak_py MB':>10}  {'ΔRSS MB':>9}  {'RSS MB':>7}")
    print(f"  {sep}")
    for r in rows:
        print(
            f"  {r['label']:<46} {r['elapsed_ms']:>8.0f}    "
            f"{r['peak_py_mb']:>10.1f}  "
            f"{r['rss_delta_mb']:>+9.1f}  "
            f"{r['rss_after_mb']:>7.1f}"
        )
    print(f"  {sep}")
    total_ms = sum(r["elapsed_ms"] for r in rows)
    peak_rss = max(r["rss_after_mb"] for r in rows)
    print(f"\n  Total wall-clock : {total_ms:.0f} ms")
    print(f"  Peak RSS         : {peak_rss:.1f} MB")

    return {
        "model": model,
        "n_mi300": n_mi300,
        "n_h100": n_h100,
        "total_ms": total_ms,
        "peak_rss_mb": peak_rss,
        "stages": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen_Qwen1.5-0.5B-Chat__1016005",
        help=f"Model name or 'all'. Available: {', '.join(MODELS)}",
    )
    args = parser.parse_args()

    if args.model == "all":
        models_to_run = MODELS
    else:
        models_to_run = [args.model]

    all_results = []
    for model in models_to_run:
        try:
            result = run_model(model)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"  Skipping {model}: {e}")

    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print("SUMMARY ACROSS ALL MODELS")
        print(f"{'Model':<55} {'Events(mi300)':>13} {'Events(h100)':>12} {'Total ms':>10} {'Peak RSS MB':>12}")
        print("─" * 110)
        for r in all_results:
            print(
                f"{r['model']:<55} {r['n_mi300']:>13,} {r['n_h100']:>12,} "
                f"{r['total_ms']:>10.0f} {r['peak_rss_mb']:>12.1f}"
            )


if __name__ == "__main__":
    main()
