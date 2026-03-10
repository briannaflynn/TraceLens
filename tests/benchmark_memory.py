###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Memory and timing benchmark for the TraceLens PyTorch perf-report pipeline.

Measures peak Python allocations (tracemalloc) and OS RSS (psutil) at each
major stage, so each code change can be evaluated against a stable baseline.

All heavy imports happen before measurements begin so that module-load time
and memory do not skew per-stage numbers.

Usage:
    python tests/benchmark_memory.py
    python tests/benchmark_memory.py --trace path/to/other.json.gz

Columns:
    ms          wall-clock time for the stage
    peak_py MB  peak Python heap allocated *during* this stage (tracemalloc)
    ΔRSS MB     change in OS resident-set size after the stage completes
    RSS MB      absolute RSS after the stage completes
"""

import argparse
import gc
import gzip
import time
import tracemalloc
from TraceLens.util import TraceEventUtils

import orjson
import psutil

# Pre-import all TraceLens modules so that import overhead does not appear
# inside any measured stage.
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.gpu_event_analyser import PytorchGPUEventAnalyser
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

DEFAULT_TRACE = "tests/traces/mi300/Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz"

_proc = psutil.Process()


def _rss_mb() -> float:
    gc.collect()
    return _proc.memory_info().rss / 1024**2


_rows: list[dict] = []


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
    _rows.append(row)
    print(
        f"  {label:<46} {elapsed_ms:>8.0f} ms  "
        f"peak_py={peak_py/1024**2:>7.1f} MB  "
        f"ΔRSS={rss_after-rss_before:>+7.1f} MB  "
        f"RSS={rss_after:>7.1f} MB"
    )
    return result


def _print_summary(n_events: int) -> None:
    sep = "─" * 90
    hdr = f"  {'Stage':<46} {'ms':>8}    {'peak_py MB':>10}  {'ΔRSS MB':>9}  {'RSS MB':>7}"
    print(f"\n{hdr}")
    print(f"  {sep}")
    for r in _rows:
        print(
            f"  {r['label']:<46} {r['elapsed_ms']:>8.0f}    "
            f"{r['peak_py_mb']:>10.1f}  "
            f"{r['rss_delta_mb']:>+9.1f}  "
            f"{r['rss_after_mb']:>7.1f}"
        )
    print(f"  {sep}")

    total_ms = sum(r["elapsed_ms"] for r in _rows)
    print(f"\n  Total wall-clock : {total_ms:.0f} ms")
    print(f"  Event count      : {n_events:,}")
    if n_events:
        us_per_event = total_ms * 1000 / n_events
        print(f"  Time / event     : {us_per_event:.2f} µs")
        peak_rss = max(r["rss_after_mb"] for r in _rows)
        print(f"  Peak RSS         : {peak_rss:.1f} MB")
        print(f"  Peak RSS / event : {peak_rss * 1024 / n_events:.2f} KB")


def run(trace_path: str) -> None:
    print(f"\n=== TraceLens memory benchmark ===")
    print(f"Trace: {trace_path}\n")

    # ── Stage 1: decompress bytes ──────────────────────────────────────────
    def _decompress():
        with gzip.open(trace_path, "rb") as f:
            return f.read()

    raw_bytes = measure("1. gzip decompress → bytes", _decompress)
    raw_size_mb = len(raw_bytes) / 1024**2
    print(f"     uncompressed size: {raw_size_mb:.1f} MB")

    # ── Stage 2: JSON parse ────────────────────────────────────────────────
    def _parse(data):
        return orjson.loads(data)

    parsed = measure("2. orjson parse → dict", _parse, raw_bytes)
    del raw_bytes

    # ── Stage 3: extract traceEvents ──────────────────────────────────────
    def _extract(data):
        return data["traceEvents"]

    events = measure("3. extract traceEvents list", _extract, parsed)
    n_events = len(events)
    del parsed
    print(f"     event count: {n_events:,}")

    # ── Stage 4a: isolated UID-stamp comparison (copy vs in-place) ───────────
    # Measures only the cost of adding UIDs so the improvement from Change A
    # is visible independently of events_by_uid and other init overhead.
    print()
    print("  [Change A comparison: UID stamping strategy]")
    _UID_KEY = TraceEventUtils.TraceKeys.UID

    def _uid_copy(evts):
        return [{**e, _UID_KEY: i} for i, e in enumerate(evts)]

    def _uid_inplace(evts):
        for i, e in enumerate(evts):
            e[_UID_KEY] = i
        return list(evts)

    # Remove any UIDs added by earlier runs so both approaches start clean
    for e in events:
        e.pop(_UID_KEY, None)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    _copy_result = _uid_copy(events)
    _copy_ms = (time.perf_counter() - t0) * 1000
    _, _copy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del _copy_result
    for e in events:
        e.pop(_UID_KEY, None)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    _ip_result = _uid_inplace(events)
    _ip_ms = (time.perf_counter() - t0) * 1000
    _, _ip_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del _ip_result

    _copy_peak_mb = _copy_peak / 1024**2
    _ip_peak_mb = _ip_peak / 1024**2
    _saving_pct = (1 - _ip_peak_mb / max(_copy_peak_mb, 1e-9)) * 100
    print(f"    dict-copy  peak_py={_copy_peak_mb:>6.2f} MB  {_copy_ms:>6.1f} ms")
    print(f"    in-place   peak_py={_ip_peak_mb:>6.2f} MB  {_ip_ms:>6.1f} ms")
    print(f"    savings    {_copy_peak_mb - _ip_peak_mb:>6.2f} MB  ({_saving_pct:.0f}%)")
    print(f"    projected for 1M events (×{1_000_000 // n_events}):  "
          f"{_copy_peak_mb * 1_000_000 // n_events:.0f} MB → "
          f"{_ip_peak_mb * 1_000_000 // n_events:.0f} MB")
    print()

    # ── Stage 4: TraceToTree construction (full) ──────────────────────────
    tree = measure("4. TraceToTree.__init__  (UID stamp)", TraceToTree, events)

    # ── Stage 5: build_tree (host call-stack tree) ─────────────────────────
    def _build_tree(t):
        t.build_tree()
        return t

    tree = measure("5. tree.build_tree()  (call stack)", _build_tree, tree)

    # ── Stage 6: GPU event analysis ───────────────────────────────────────
    def _gpu_analysis(t):
        analyser = PytorchGPUEventAnalyser(t.events)
        return analyser.compute_metrics()

    measure("6. GPUEventAnalyser.compute_metrics()", _gpu_analysis, tree)

    # ── Stage 7: get_df_kernel_launchers ──────────────────────────────────
    def _build_perf(t):
        analyzer = TreePerfAnalyzer(t)
        return analyzer.get_df_kernel_launchers(include_kernel_details=True)

    df_kl = measure("7. get_df_kernel_launchers()", _build_perf, tree)
    if df_kl is not None:
        print(f"     kernel launcher rows: {len(df_kl):,}")

    # ── Summary ────────────────────────────────────────────────────────────
    _print_summary(n_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=DEFAULT_TRACE, help="Path to trace file")
    args = parser.parse_args()
    run(args.trace)
