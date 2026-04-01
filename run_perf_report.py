###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Standalone CLI for generating a TraceLens PyTorch perf report.

Usage:
    python run_perf_report.py --trace path/to/trace.json.gz
    python run_perf_report.py --trace path/to/trace.json.gz --output_dir ./results
    python run_perf_report.py --trace path/to/trace.json.gz --xlsx report.xlsx
    python run_perf_report.py --trace path/to/trace.json.gz --no_collective_analysis

Requires TraceLens to be installed or the repo root on PYTHONPATH:
    pip install -e .               (from the repo root)
    PYTHONPATH=/path/to/TraceLens python run_perf_report.py --trace ...
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Generate a TraceLens perf report from a PyTorch trace file."
    )

    # Required
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to the trace file (.json or .json.gz).",
    )

    # Output — at least one of these should be provided
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write per-sheet CSV files. "
             "Defaults to <trace_basename>_perf_report_csvs/ next to the trace.",
    )
    parser.add_argument(
        "--xlsx",
        default=None,
        help="Path to write an Excel (.xlsx) report. "
             "Defaults to <trace_basename>_perf_report.xlsx next to the trace.",
    )

    # Common options
    parser.add_argument(
        "--no_collective_analysis",
        action="store_true",
        help="Skip NCCL/collective analysis (useful for non-distributed or "
             "inference-only traces).",
    )
    parser.add_argument(
        "--kernel_summary",
        action="store_true",
        help="Include a kernel summary sheet.",
    )
    parser.add_argument(
        "--short_kernel_study",
        action="store_true",
        help="Include short kernel study sheets.",
    )
    parser.add_argument(
        "--short_kernel_threshold_us",
        type=int,
        default=10,
        help="Threshold (µs) below which a kernel is considered short. Default: 10.",
    )
    parser.add_argument(
        "--gpu_arch_json",
        default=None,
        help="Path to a GPU arch JSON file for roofline analysis.",
    )

    args = parser.parse_args()

    # ── Validate input ─────────────────────────────────────────────────────
    if not os.path.isfile(args.trace):
        print(f"ERROR: trace file not found: {args.trace}", file=sys.stderr)
        sys.exit(1)

    # ── Derive default output paths ────────────────────────────────────────
    trace_dir = os.path.dirname(os.path.abspath(args.trace))
    base = os.path.basename(args.trace)
    for suffix in (".json.gz", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    output_dir = args.output_dir or os.path.join(trace_dir, f"{base}_perf_report_csvs")
    xlsx_path = args.xlsx or os.path.join(trace_dir, f"{base}_perf_report.xlsx")

    # ── Import TraceLens ───────────────────────────────────────────────────
    try:
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            generate_perf_report_pytorch,
        )
    except ImportError as e:
        print(
            f"ERROR: could not import TraceLens: {e}\n"
            "Install with:  pip install -e .  (from repo root)\n"
            "or set:        PYTHONPATH=/path/to/TraceLens",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Run ────────────────────────────────────────────────────────────────
    print(f"Trace      : {args.trace}")
    print(f"CSV output : {output_dir}")
    print(f"XLSX output: {xlsx_path}")
    print()

    t0 = time.perf_counter()
    generate_perf_report_pytorch(
        profile_json_path=args.trace,
        output_csvs_dir=output_dir,
        output_xlsx_path=xlsx_path,
        collective_analysis=not args.no_collective_analysis,
        kernel_summary=args.kernel_summary,
        short_kernel_study=args.short_kernel_study,
        short_kernel_threshold_us=args.short_kernel_threshold_us,
        gpu_arch_json_path=args.gpu_arch_json,
    )
    elapsed = time.perf_counter() - t0

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  CSVs : {output_dir}/")
    print(f"  XLSX : {xlsx_path}")


if __name__ == "__main__":
    main()
