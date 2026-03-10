# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable) with dev extras
pip install -e .[dev]

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_perf_report_regression.py -v

# Run a single test by name
python -m pytest tests/test_perf_model.py -v -k "test_gemm"

# Format code (required before PRs — CI checks changed files only)
black .
black path/to/your_file.py

# CLI entry points (after install)
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/trace.json
TraceLens_compare_perf_reports_pytorch baseline.xlsx candidate.xlsx --names baseline candidate --sheets all -o comparison.xlsx
TraceLens_generate_multi_rank_collective_report_pytorch --trace_dir /path/to/traces --world_size 8
TraceLens_generate_perf_report_rocprof --profile_json_path trace_results.json
TraceLens_generate_perf_report_pftrace_hip_activity --trace_path sample.pftrace
```

CI runs tests with: `pip install . pytest openpyxl orjson tensorboard-plugin-profile==2.19.0`

## Architecture

The core data flow is: **Trace file → DataLoader → Trace2Tree → TreePerf → Reporting (Excel)**

### Module Overview

**`TraceLens/util.py`** — Shared utilities used across all modules:
- `DataLoader`: Loads `.json`, `.json.gz`, and `.pb` (TensorBoard XPlane) trace files
- `TraceEventUtils`: Common event field keys and helpers
- `JaxProfileProcessor`: JAX-specific trace preprocessing

**`Trace2Tree/`** — Parses raw trace events into a hierarchical tree linking Python ops → CPU dispatches → GPU kernels. `BaseTraceToTree` is the abstract base; `TraceToTree` handles PyTorch, `JaxTraceToTree` handles JAX. The `extensions/` subdirectory adds pseudo-op support (e.g., MoE fused ops).

**`PerfModel/`** — Performance models for individual ops. `perf_model.py` contains base classes (e.g., `GEMM`) that compute theoretical FLOP counts and memory bytes from event metadata. `torch_op_mapping.py` and `jax_op_mapping.py` map op names to their model class. `kernel_name_parser.py` extracts GEMM dimensions from GPU kernel names. The `extensions/` subdirectory adds models for custom ops.

**`TreePerf/`** — Combines the trace tree with PerfModel to produce per-op performance metrics (GPU time, TFLOP/s, TB/s, roofline bound). `TreePerfAnalyzer` (PyTorch) and `JaxTreePerfAnalyzer` are the main analysis classes. `GPUEventAnalyser` (and its variants) handles GPU timeline segmentation (idle/busy/compute/comms). `jax_analyses.py` contains JAX-specific analyses like conv op analysis.

**`NcclAnalyser/`** — Multi-rank collective analysis. Separates pure communication time from synchronization skew, computes effective bandwidth. `nccl_analyser.py` for PyTorch NCCL, `jax_nccl_analyser.py` for JAX collectives.

**`TraceFusion/`** — Merges multi-rank PyTorch traces into a single Perfetto-compatible JSON for visualization.

**`TraceDiff/`** — Morphological tree comparison across two traces to identify structural divergences at the CPU dispatch level.

**`EventReplay/`** — Generates minimal standalone replay scripts from trace event metadata for isolated kernel debugging.

**`Reporting/`** — CLI entry points that orchestrate the above modules and write Excel (`.xlsx`) output via `openpyxl`. Each `generate_perf_report_*.py` file is a self-contained script with a `main()` function registered as a console script in `setup.py`. `reporting_utils.py` contains shared Excel-writing helpers.

### Regression Tests

Tests in `tests/test_perf_report_regression.py` and similar compare DataFrames against reference outputs stored in `tests/`. The shared helpers in `tests/conftest.py` (`compare_cols`, `normalize_value`, `format_diff_details`) are used across multiple test files to provide detailed diff output on failure.

## Conventions

- **Copyright header**: All new Python files must have the AMD copyright header (see any existing `.py` file for the exact text).
- **Formatting**: Black is enforced in CI on all changed `.py` files.
- **Branch naming**: `<type>/<scope>/<short-description>` (e.g., `feat/perfmodel/new-op`, `fix/tracediff/reporting-bug`).
- **Commit messages**: Conventional Commits format — `feat(perfmodel): add perf model for X`.
- **This is a public repo**: Do not add private, confidential, or customer-related data.
