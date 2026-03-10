###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Generate a synthetic large-scale trace by tiling an existing trace N times.

Each tile is a copy of the original traceEvents with timestamps shifted by
  tile_index * (trace_span + gap_us)
so tiles are non-overlapping and structurally identical — the tree builder sees
N independent iterations of the same model, just as a real long trace would.

Usage:
    python tests/generate_synthetic_trace.py --multiplier 10 --output /tmp/synth_10x.json.gz
    python tests/generate_synthetic_trace.py --multiplier 55 --output /tmp/synth_1M_events.json.gz
    python tests/generate_synthetic_trace.py --list-sizes
"""

import argparse
import copy
import gzip
import os

import orjson

DEFAULT_SOURCE = "tests/traces/mi300/Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz"


def _load(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return orjson.loads(f.read())


def _trace_span_us(events: list) -> float:
    """Return the wall-clock span of all events in microseconds."""
    ts_vals = [e["ts"] for e in events if "ts" in e]
    dur_vals = [e.get("dur", 0) for e in events if "ts" in e]
    t_ends = [ts + dur for ts, dur in zip(ts_vals, dur_vals)]
    return max(t_ends) - min(ts_vals)


def generate(source_path: str, multiplier: int, gap_fraction: float = 0.1) -> dict:
    """Return a new trace dict with traceEvents tiled `multiplier` times.

    Args:
        source_path:    Path to the source .json.gz trace.
        multiplier:     How many copies of the original events to produce.
        gap_fraction:   Fractional gap between tiles (0.1 = 10% of span).
    """
    data = _load(source_path)
    orig_events = data["traceEvents"]
    span_us = _trace_span_us(orig_events)
    tile_step = span_us * (1 + gap_fraction)

    # Compute a single offset stride that covers both the ac2g flow ids and
    # the args["correlation"] values — they use the same numeric space.
    # Each tile shifts its ids/correlations by (tile_index * stride) so that
    # CPU↔GPU links are self-contained within each tile.
    link_ids = [e["id"] for e in orig_events if "id" in e] + [
        e.get("args", {}).get("correlation", 0)
        for e in orig_events
        if "args" in e and "correlation" in e.get("args", {})
    ]
    id_stride = (max(link_ids) + 1) if link_ids else 0

    tiled: list[dict] = []
    for i in range(multiplier):
        ts_offset = i * tile_step
        id_offset = i * id_stride
        for e in orig_events:
            ph = e.get("ph")
            # Metadata events (ph="M"): process/thread names — emit once only.
            if ph == "M":
                if i == 0:
                    tiled.append(e)
                continue
            new_e = dict(e)  # shallow copy
            new_e["ts"] = e["ts"] + ts_offset
            # Flow events (ac2g): offset the linking id.
            if ph in ("s", "f"):
                new_e["id"] = e["id"] + id_offset
            # Events with args.correlation: must offset to match the shifted
            # ac2g ids so each tile's CPU↔GPU links resolve correctly.
            if i > 0 and "args" in e and "correlation" in e["args"]:
                new_args = dict(e["args"])  # new dict; other fields still shared
                new_args["correlation"] = e["args"]["correlation"] + id_offset
                new_e["args"] = new_args
            tiled.append(new_e)

    result = dict(data)
    result["traceEvents"] = tiled
    result["traceName"] = f"{data.get('traceName', 'trace')}__synthetic_{multiplier}x"
    return result


def save(trace: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = orjson.dumps(trace)
    with gzip.open(output_path, "wb", compresslevel=1) as f:
        f.write(payload)
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Wrote {len(trace['traceEvents']):,} events → {output_path}  ({size_mb:.1f} MB)")


def _list_sizes(source_path: str) -> None:
    data = _load(source_path)
    n = len(data["traceEvents"])
    span = _trace_span_us(data["traceEvents"])
    print(f"Source: {source_path}")
    print(f"  {n:,} events  |  {span/1000:.1f} ms span")
    print()
    print(f"  {'multiplier':>12}  {'events':>12}  {'approx span':>14}")
    print(f"  {'-'*42}")
    for m in [1, 5, 10, 20, 55, 100]:
        print(f"  {m:>12}x  {n*m:>12,}  {span*m/1e6:>12.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE, help="Source trace (.json.gz)"
    )
    parser.add_argument(
        "--multiplier", type=int, default=10, help="Number of tiles (default: 10)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: /tmp/synth_<N>x.json.gz)",
    )
    parser.add_argument(
        "--gap-fraction",
        type=float,
        default=0.1,
        help="Gap between tiles as a fraction of the trace span (default: 0.1)",
    )
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="Print size table for common multipliers and exit",
    )
    args = parser.parse_args()

    if args.list_sizes:
        _list_sizes(args.source)
    else:
        output = args.output or f"/tmp/synth_{args.multiplier}x.json.gz"
        trace = generate(args.source, args.multiplier, args.gap_fraction)
        save(trace, output)
