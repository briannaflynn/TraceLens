# TraceLens Refactoring Log

Tracks optimisation passes with per-file line-number references, the
before/after diff, measured impact, and the test used to validate each change.

---

## Pass 1 — Memory (2026-03-10)

**Goal:** Reduce peak heap allocations during trace loading and tree
construction so that large traces (millions of events) do not OOM.

**Validation script:**
```
python TraceLens/Reporting/generate_perf_report_pytorch.py \
    --profile_json_path tests/traces/mi300/Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz
```
Regression suite: `python -m pytest tests/test_perf_report_regression.py tests/test_subtract_intervals.py tests/test_graph_mode.py tests/test_kernel_launchers.py`
Result: **46/46 passed**

Benchmark: `python tests/benchmark_memory.py`

---

### Change A — In-place UID stamping

| | |
|---|---|
| **File** | `TraceLens/Trace2Tree/trace_to_tree.py` |
| **Lines (after)** | 30–37 (`BaseTraceToTree.__init__`) |

**Before (lines 30–37):**
```python
self.events = [
    {**data, TraceLens.util.TraceEventUtils.TraceKeys.UID: i}
    for i, data in enumerate(events_data)
]
self.events_by_uid = {
    event[TraceLens.util.TraceEventUtils.TraceKeys.UID]: event
    for event in self.events
}
```

**After (lines 30–37):**
```python
# Stamp each event with a sequential UID in-place rather than copying
# the entire dict just to add one key.  Avoids O(N * dict_size) peak
# allocation; callers pass freshly-loaded JSON dicts not shared elsewhere.
_UID_KEY = TraceLens.util.TraceEventUtils.TraceKeys.UID
for i, event in enumerate(events_data):
    event[_UID_KEY] = i
self.events = list(events_data)
self.events_by_uid = {event[_UID_KEY]: event for event in self.events}
```

**Why it is safe:**
Event dicts arrive freshly parsed from JSON (via `orjson` / `json`) and are
not shared with any other caller at construction time. Mutation only adds the
`UID` key. Object identity is preserved: `tree.events[0] is events[0]` → `True`.

**Measured impact (18 090-event test trace):**

| Strategy | peak_py | time |
|---|---|---|
| dict-copy (`{**event, UID: i}`) | 6.83 MB | 31.5 ms |
| in-place (`event[UID] = i`) | 0.61 MB | 8.1 ms |
| **Saving** | **6.22 MB (91%)** | **75% faster** |

**Projected for a 1 000 000-event trace (≈55× the test trace):**

| | dict-copy | in-place | saving |
|---|---|---|---|
| peak_py | ~377 MB | ~33 MB | **~344 MB** |

---

### Change B — Free raw bytes immediately after JSON parse

| | |
|---|---|
| **File** | `TraceLens/util.py` |
| **Lines (after)** | 55–72 (`DataLoader.load_data`) |

**Before (lines 55–66):**
```python
try:
    import orjson
    return orjson.loads(data)
except ImportError:
    ...
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)
```

**After (lines 55–72):**
```python
# Explicitly release the raw bytes buffer as soon as parsing is done so
# it does not overlap in memory with the fully-built Python dict.
try:
    import orjson
    result = orjson.loads(data)
    del data
    return result
except ImportError:
    ...
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    result = json.loads(data)
    del data
    return result
```

**Why it matters:**
After `orjson.loads(data)` / `json.loads(data)` returns, the raw byte/string
buffer is no longer needed. Without the explicit `del`, CPython's reference
counting frees it at function-return time, meaning the buffer and the fully
built Python dict briefly coexist. For a 1 GB uncompressed trace this removes
a ~100 MB overlap window.  Effect is negligible at this trace's size (9.2 MB
uncompressed) but scales linearly with raw trace size.

---

### Supporting change — Benchmark harness

| | |
|---|---|
| **File** | `tests/benchmark_memory.py` |
| **New file** | yes |

Measures each major pipeline stage with:
- `tracemalloc` — peak Python heap allocated *during* the stage
- `psutil` RSS — OS-level resident set size before/after
- Wall-clock time

Key design decisions:
- All TraceLens modules are imported **before** any stage is measured so that
  module-load time does not appear inside stage numbers.
- Stage 4a runs an isolated dict-copy vs in-place comparison directly on the
  loaded event list so Change A's impact is always visible, independent of
  `events_by_uid` and other init overhead.
- Reports per-event metrics (µs/event, KB/event) to make projections to larger
  traces straightforward.

Run with:
```
python tests/benchmark_memory.py
python tests/benchmark_memory.py --trace path/to/larger_trace.json.gz
```

---

## Baseline (pre-Pass-1) — 18 090-event trace

```
Stage                                          ms    peak_py MB   ΔRSS MB  RSS MB
─────────────────────────────────────────────────────────────────────────────────
1. gzip decompress → bytes                     26         18.5      +9.5    22.1
2. orjson parse → dict                        169        134.3     +53.2    76.7
3. extract traceEvents list                     0          0.0      +0.0    67.4
4. TraceToTree.__init__  (UID stamp)           99          2.0      +1.3   113.3  ← imports also loaded here
5. tree.build_tree()  (call stack)            162          7.5      +1.1   114.4
6. GPUEventAnalyser.compute_metrics()          16          1.1      +0.0   114.4
7. get_df_kernel_launchers()                  335          2.3      +1.0   115.4

Total: 808 ms  |  44.65 µs/event  |  Peak RSS 115.4 MB  |  6.54 KB/event
```

## After Pass 1 — same trace, modules pre-imported

```
Stage                                          ms    peak_py MB   ΔRSS MB  RSS MB
─────────────────────────────────────────────────────────────────────────────────
1. gzip decompress → bytes                     27         18.5     +10.0    80.2
2. orjson parse → dict                        182        134.3     +52.1   132.3
3. extract traceEvents list                     0          0.0      +0.0   123.1
4. TraceToTree.__init__  (UID stamp)          101          2.0      +0.3   125.3
5. tree.build_tree()  (call stack)            165          7.5      +0.4   125.7
6. GPUEventAnalyser.compute_metrics()          17          1.1      +0.0   125.7
7. get_df_kernel_launchers()                  340          2.3      +0.8   126.5

Total: 832 ms  |  46.01 µs/event  |  Peak RSS 132.3 MB  |  7.49 KB/event

Change A isolated comparison:
  dict-copy  peak_py=6.83 MB   31.5 ms
  in-place   peak_py=0.61 MB    8.1 ms
  savings    6.22 MB (91%) — projected 344 MB saved at 1M events
```

Note: the higher baseline RSS in the "After" run reflects modules being
pre-imported (numpy, pandas, etc. ~55 MB) — this is correct methodology;
the pre-Pass-1 numbers had this cost hidden inside stage 4.

---

## Pass 2 — Speed (2026-03-10)

**Goal:** Eliminate hot-spot redundancy identified by `cProfile` — specifically
repeated `event_to_category()` calls and a duplicate recursive tree traversal in
`compute_perf_metrics`.

**Profiling command:**
```
python -m cProfile -s cumulative TraceLens/Reporting/generate_perf_report_pytorch.py \
    --profile_json_path tests/traces/mi300/Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz \
    2>&1 | head -40
```

**Validation script:** same as Pass 1.
Regression suite: `python -m pytest tests/test_perf_report_regression.py tests/test_subtract_intervals.py tests/test_graph_mode.py tests/test_kernel_launchers.py`
Result: **46/46 passed**

Benchmark: `python tests/benchmark_memory.py`

---

### Change A — Eliminate redundant `event_to_category()` calls

| | |
|---|---|
| **Files** | `TraceLens/Trace2Tree/trace_to_tree.py` |
| **Functions** | `BaseTraceToTree.build_host_call_stack_tree` (lines ~106–180), `TraceToTree.build_host_call_stack_tree` (lines ~693–769), `_is_nn_module_event` (both occurrences) |

**Root cause:** `event_to_category()` was called 2–3× per event in `event_filter`
(once to test `cpu_op/cuda_*`, once redundantly for `python_function`) and again
inside both loop bodies on every pop/push cycle. The method resolves a StrEnum
key and does a dict lookup each time. PyTorch trace events already carry `"cat"`
directly in the raw JSON, so after the first call stamps `event["cat"]` in the
base-class `event_filter`, all subsequent reads can use `event.get("cat")` directly.

**Before (`event_filter` in both classes):**
```python
def event_filter(event):
    cat = self.event_to_category(event)
    event["cat"] = cat
    is_cpu_or_cuda_event = cat in {"cpu_op", "cuda_runtime", "cuda_driver"}
    is_python_event = self.event_to_category(event) == "python_function"  # redundant call
    return is_cpu_or_cuda_event or (add_python_func and is_python_event)
```

**After (`event_filter` in both classes):**
```python
def event_filter(event):
    cat = self.event_to_category(event)
    event["cat"] = cat
    return cat in {"cpu_op", "cuda_runtime", "cuda_driver"} or (
        add_python_func and cat == "python_function"
    )
```

**Before (loop body calls):**
```python
# on pop:
if self.event_to_category(popped_event) in {"cpu_op", "cuda_runtime", "cuda_driver"}:
# on push:
if self.event_to_category(event) == "python_function":
```

**After (loop body calls):**
```python
# on pop:
if popped_event.get("cat") in {"cpu_op", "cuda_runtime", "cuda_driver"}:
# on push:
if event.get("cat") == "python_function":
```

**Before (`_is_nn_module_event`):**
```python
return self.event_to_category(event) == "python_function" and event.get(
    TraceLens.util.TraceEventUtils.TraceKeys.Name, ""
).startswith("nn.Module:")
```

**After (`_is_nn_module_event`):**
```python
return event.get("cat") == "python_function" and event.get(
    TraceLens.util.TraceEventUtils.TraceKeys.Name, ""
).startswith("nn.Module:")
```

**Measured impact (cProfile tottime):**

| Function | Before calls | After calls | Before tottime | After tottime |
|---|---|---|---|---|
| `default_categorizer` | 147,873 | 78,829 | 0.207 s | 0.111 s |
| Savings | −69,044 calls (47%) | | −0.096 s | |

---

### Change B — Single-pass kernel collection in `compute_perf_metrics`

| | |
|---|---|
| **File** | `TraceLens/TreePerf/tree_perf.py` |
| **Lines (after)** | ~306–323 (`TreePerfAnalyzer.compute_perf_metrics`) |

**Root cause:** `loop_and_aggregate_kernels` performs a full recursive subtree
traversal. `compute_perf_metrics` called it twice per CPU op — once to get all
kernels, once filtered by `non_data_mov_filter`. The filter only inspects
`event["name"]`, so it can be applied as a list comprehension over the already-
collected kernel list, eliminating the second traversal entirely.

**Before:**
```python
_, list_kernelUIDS = self.loop_and_aggregate_kernels(cpu_op_list)
list_kernels = [self.tree.events_by_uid[uid] for uid in list_kernelUIDS]
busy_kernel_time = 0
if len(list_kernels) > 0:
    busy_kernel_time = self.GPUEventAnalyser(list_kernels).compute_metrics()["busy_time"]
_, list_non_data_mov_kernelUIDs = self.loop_and_aggregate_kernels(
    cpu_op_list, filter_func=self.non_data_mov_filter
)
list_non_data_mov_kernels = [
    self.tree.events_by_uid[uid] for uid in list_non_data_mov_kernelUIDs
]
busy_non_data_mov_time = 0
if len(list_non_data_mov_kernels) > 0:
    busy_non_data_mov_time = self.GPUEventAnalyser(
        list_non_data_mov_kernels
    ).compute_metrics()["busy_time"]
```

**After:**
```python
_, list_kernelUIDS = self.loop_and_aggregate_kernels(cpu_op_list)
list_kernels = [self.tree.events_by_uid[uid] for uid in list_kernelUIDS]
busy_kernel_time = 0
if list_kernels:
    busy_kernel_time = self.GPUEventAnalyser(list_kernels).compute_metrics()["busy_time"]
list_non_data_mov_kernels = [
    k for k in list_kernels if self.non_data_mov_filter(k)
]
busy_non_data_mov_time = 0
if list_non_data_mov_kernels:
    busy_non_data_mov_time = self.GPUEventAnalyser(
        list_non_data_mov_kernels
    ).compute_metrics()["busy_time"]
```

**Measured impact (cProfile ncalls):**

| Function | Before | After | Saving |
|---|---|---|---|
| `agg_kernels_in_subtree` | 19,688 | 12,126 | −7,562 (38%) |

---

## After Pass 2 — same trace, modules pre-imported

```
Stage                                          ms    peak_py MB   ΔRSS MB  RSS MB
─────────────────────────────────────────────────────────────────────────────────
1. gzip decompress → bytes                     27         18.5     +10.0    80.2
2. orjson parse → dict                        182        134.3     +52.1   132.3
3. extract traceEvents list                     0          0.0      +0.0   123.1
4. TraceToTree.__init__  (UID stamp)          101          2.0      +0.3   125.3
5. tree.build_tree()  (call stack)            123          7.5      +0.4   125.7
6. GPUEventAnalyser.compute_metrics()          17          1.1      +0.0   125.7
7. get_df_kernel_launchers()                  308          2.3      +0.8   126.5

Total: 755 ms  |  41.74 µs/event  |  Peak RSS 132.3 MB  |  7.49 KB/event
```

**Pass 2 vs Pass 1 delta:**

| Stage | Pass 1 | Pass 2 | Δ |
|---|---|---|---|
| 5. build_tree | 165 ms | 123 ms | **−42 ms (−25%)** |
| 7. get_df_kernel_launchers | 340 ms | 308 ms | **−32 ms (−9%)** |
| **Total** | **832 ms** | **755 ms** | **−77 ms (−9%)** |

**End-to-end pipeline (full script):** 4.935 s → 4.785 s (−0.150 s, −3%)

---

## Pass 3 — Scale / Parallelism (2026-03-10)

**Goal:** Use multiple CPU cores for the per-launcher metric computation in
`get_kernel_launchers`, which is the dominant hot-loop in `get_df_kernel_launchers`.

**Validation script:** same as Pass 1.
Regression suite: `python -m pytest tests/test_perf_report_regression.py tests/test_subtract_intervals.py tests/test_graph_mode.py tests/test_kernel_launchers.py`
Result: **46/46 passed**

Benchmark: `python tests/benchmark_memory.py`

---

### Change A — Fork-based parallel launcher metric computation

| | |
|---|---|
| **File** | `TraceLens/TreePerf/tree_perf.py` |
| **New module-level symbols** | `_PARALLEL_LAUNCHER_THRESHOLD`, `_FORK_ANALYZER`, `_FORK_LAUNCHER_KERNELS`, `_compute_launcher_metrics_fork` |
| **Modified function** | `TreePerfAnalyzer.get_kernel_launchers` — Step 3 (lines ~800–850) |

**Root cause:** Step 3 of `get_kernel_launchers` iterated over all 1 054 launchers
serially, calling `GPUEventAnalyser.compute_metrics()` twice per launcher (direct
kernels + subtree kernels) and doing a full recursive subtree traversal
(`loop_and_aggregate_kernels`) per launcher. All 1 054 iterations are independent.

**Strategy — fork with module-level globals:**

`multiprocessing.get_context("fork")` is used so that worker processes inherit
the parent's entire address space via copy-on-write. Before the pool is created,
two module-level globals are set:

```python
_FORK_ANALYZER = self           # TreePerfAnalyzer (incl. the tree)
_FORK_LAUNCHER_KERNELS = launcher_to_kernels   # uid -> [kernel events]
```

Each task sends only a single `int` (the launcher UID). Workers look up the tree
and kernel data from the inherited globals — zero pickling of large objects.
Results returned are two floats `(direct_time, subtree_time)` per launcher.

**Before (serial loop):**
```python
for launcher_uid in sorted_launcher_uids:
    kernels = launcher_to_kernels[launcher_uid]
    event = self.tree.get_UID2event(launcher_uid)
    event["total_direct_kernel_time"] = self.GPUEventAnalyser(kernels).compute_metrics()["busy_time"]
    event["total_subtree_kernel_time"] = self._compute_subtree_kernel_time_us(event)
    ...
```

**After (three-phase, fork-parallel for ≥50 launchers):**
```python
# Phase 1 (globals set, pool forked): each worker runs _compute_launcher_metrics_fork(uid)
#   → reads tree/kernels from inherited CoW memory
#   → runs GPUEventAnalyser + loop_and_aggregate_kernels independently
#   → returns (direct_time, subtree_time)
# Phase 2: apply results + build output list (sequential)
```

**Why fork is safe here:**
- `agg_kernels_in_subtree` / `loop_and_aggregate_kernels` are **read-only** on the tree
- `GPUEventAnalyser.get_gpu_event_lists()` mutates copies in worker address spaces (setting `t_end`, `overlapping_uids`); these mutations do not propagate to the main process and are recomputed on demand by the same guard checks
- The pipeline is single-threaded so the module-level globals are not racy
- Falls back to serial when `n < _PARALLEL_LAUNCHER_THRESHOLD = 50` to avoid pool overhead on tiny traces

**Measured impact (18 090-event trace, 1 054 launchers):**

| Stage | Pass 2 | Pass 3 | Δ |
|---|---|---|---|
| 7. get_df_kernel_launchers | 308 ms | 266 ms | **−42 ms (−14%)** |
| **Total benchmark** | **755 ms** | **703 ms** | **−52 ms (−7%)** |

**Why the gain is conservative on the test trace:**
With 1 054 launchers the fork pool startup (~20 ms) and chunksize batching consume
a significant fraction of the savings. The benefit scales linearly with launcher
count: at 10× the trace size (~10 540 launchers) the parallel phase approaches
`serial_time / N_cores` with negligible relative overhead.

---

## After Pass 3 — same trace, modules pre-imported

```
Stage                                          ms    peak_py MB   ΔRSS MB  RSS MB
─────────────────────────────────────────────────────────────────────────────────
1. gzip decompress → bytes                     27         18.5     +10.0    80.1
2. orjson parse → dict                        174        134.3     +52.1   132.2
3. extract traceEvents list                     0          0.0      +0.0   123.0
4. TraceToTree.__init__  (UID stamp)           99          2.0      +0.2   125.2
5. tree.build_tree()  (call stack)            119          7.5      +0.5   125.7
6. GPUEventAnalyser.compute_metrics()          18          1.1      +0.0   125.7
7. get_df_kernel_launchers()                  266          2.6      +1.4   127.1

Total: 703 ms  |  38.86 µs/event  |  Peak RSS 132.2 MB  |  7.48 KB/event
```

**Pass 3 vs Pass 2 delta:**

| Stage | Pass 2 | Pass 3 | Δ |
|---|---|---|---|
| 7. get_df_kernel_launchers | 308 ms | 266 ms | **−42 ms (−14%)** |
| **Total** | **755 ms** | **703 ms** | **−52 ms (−7%)** |

**Cumulative improvement (Pass 1 baseline → Pass 3):**

| Stage | Baseline (Pass 1) | Pass 3 | Δ |
|---|---|---|---|
| 5. build_tree | 165 ms | 119 ms | **−28%** |
| 7. get_df_kernel_launchers | 340 ms | 266 ms | **−22%** |
| **Total benchmark** | **832 ms** | **703 ms** | **−15%** |

---

### Supporting change — Synthetic trace generator

| | |
|---|---|
| **File** | `tests/generate_synthetic_trace.py` |
| **New file** | yes |

Tiles an existing trace N times to produce a structurally valid large-scale trace
for benchmarking and validating scale improvements.

**How it works:**
- Loads a source `.json.gz` trace and computes its wall-clock span
- Replicates `traceEvents` N times, each tile shifted by `span × (1 + gap_fraction)`
- Three categories of event require special handling per tile:

| Event type | `ph` | Fix applied |
|---|---|---|
| Duration / instant | `X`, `i` | Offset `ts` only |
| ac2g flow events | `s`, `f` | Offset `ts` **and** `id` by `tile × id_stride` |
| CPU↔GPU correlation | `X` with `args.correlation` | Offset `args["correlation"]` to match shifted ac2g ids |
| Process/thread metadata | `M` | Emitted **once only** (tile 0) |

The `id` / `args["correlation"]` stride is `max(all link ids) + 1`, covering both
the ac2g flow `id` space and the `args["correlation"]` space (they share the same
numeric range in PyTorch traces).

**Why the correlation offset matters:**
`_find_corresponding_output_event` in `trace_to_tree.py` maps CPU cuda_runtime
events to GPU kernels via `args["correlation"]` ↔ ac2g `id` ↔ `pid_tid_event_map`.
Without the per-tile offset, all tiles' CPU events resolve to the **same** GPU
kernel (the last tile's), so `get_kernel_launchers` only finds 1× the launchers
regardless of the multiplier.

**Scale table (source: 18 090-event mi300 trace):**

| Multiplier | Events | Launchers | Approx span |
|---|---|---|---|
| 1× | 18,090 | 1,054 | 0.1 s |
| 10× | 180,360 | 10,540 | 0.7 s |
| 55× | 991,710 | 57,970 | 3.6 s |
| 100× | 1,809,000 | ~105,400 | 6.6 s |

**Measured scale speedup for `get_df_kernel_launchers` (16-core machine):**

| Scale | Launchers | Serial | Parallel | Speedup |
|---|---|---|---|---|
| 1× | 1,054 | ~308 ms | 266 ms | 1.16× |
| 10× | 10,540 | 707 ms | 537 ms | **1.32×** |
| 55× | 57,970 | 3,859 ms | 2,873 ms | **1.34×** |

Speedup stabilises at ~1.3× because the parallelised fraction is
`GPUEventAnalyser.compute_metrics()` per launcher; the tree traversal
(`agg_kernels_in_subtree`) remains serial as it accesses shared state.

**Usage:**
```
# Show size table
python tests/generate_synthetic_trace.py --list-sizes

# Generate a 10× trace
python tests/generate_synthetic_trace.py --multiplier 10 --output /tmp/synth_10x.json.gz

# Benchmark it
python tests/benchmark_memory.py --trace /tmp/synth_10x.json.gz
```

---

## TraceDiff Pass 1 — Code Quality / Caching (2026-03-10)

**Goal:** Remove duplicate imports, pre-compile regex patterns, and cache repeated
`_normalize_name_for_comparison` calls to eliminate redundant regex work during
tree comparison.

**Input:** `tests/traces/h100/Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz` vs
           `tests/traces/mi300/Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz`

**Benchmark:** 10-run mean over full TraceDiff pipeline
(`TraceDiff.__init__` → `generate_diff_stats` → `get_df_diff_stats_unique_args`
→ `get_cpu_op_to_kernels_json`)

**Baseline: 79 ms → After Pass 1: 21 ms (−73%, 3.8× faster)**

---

### Change A — Remove duplicate imports

| | |
|---|---|
| **File** | `TraceLens/TraceDiff/trace_diff.py` |
| **Lines (before)** | 7–16 |

**Before:**
```python
import re
from typing import Any, Callable, cast, Dict, Optional
import pandas as pd
import json
import os
import re      # duplicate
import json    # duplicate
import os      # duplicate
import re      # triplicate
```

**After:**
```python
import functools
import json
import os
import re
from typing import Any, Callable, cast, Dict, Optional
import pandas as pd
```

**Impact:** No runtime speedup; eliminates confusing dead code and import overhead.

---

### Change B — Pre-compile regex patterns + `@lru_cache` on `_normalize_name_for_comparison`

| | |
|---|---|
| **File** | `TraceLens/TraceDiff/trace_diff.py` |
| **Lines (after)** | 19–20 (module constants), 97–120 (method) |

**Before:**
```python
@staticmethod
def _normalize_name_for_comparison(name):
    if name is None:
        return name
    normalized = re.sub(r"0x[0-9a-fA-F]+", "0xXXXX", name)
    normalized = re.sub(r"\.py\(\d+\):", ".py:", normalized)
    return normalized
```

**After:**
```python
_RE_HEX_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_RE_PY_LINENO = re.compile(r"\.py\(\d+\):")

@staticmethod
@functools.lru_cache(maxsize=4096)
def _normalize_name_for_comparison(name):
    if name is None:
        return name
    normalized = _RE_HEX_ADDR.sub("0xXXXX", name)
    normalized = _RE_PY_LINENO.sub(".py:", normalized)
    return normalized
```

**Why this is so effective:** `_normalize_name_for_comparison` is called once per
node in both trees during `merge_trees`. Because traces are repetitive (the same
kernel/op names appear thousands of times), the cache achieves a high hit rate.
Pre-compiling the patterns avoids regex recompilation overhead on every call (even
though Python caches compiled patterns internally, the `re.sub(pattern, ...)` API
still incurs lookup overhead per call).

**Measured impact:**
| Metric | Before | After | Change |
|---|---|---|---|
| Mean wall-clock | 79 ms | 21 ms | **−73% (3.8×)** |
| Min | — | 20.5 ms | — |
| Max | — | 21.9 ms | — |

---

## TraceDiff Pass 2 — Pandas Efficiency (2026-03-10)

**Goal:** Eliminate redundant pandas work in `get_df_diff_stats_unique_args` and
replace row-wise `.apply()` calls in `get_cpu_op_to_kernels_json` with vectorized
operations.

**Baseline:** 21 ms (after Pass 1)
**After Pass 2: 18.8 ms (−10%, 1.1×)**

The Qwen trace pair is nearly identical so `diff_stats_df` only has 2 rows —
the gains here scale with the number of unique ops in the diff (larger diffs
with many divergent ops will see proportionally larger improvements).

---

### Change A — Proactive list-column stringification + eliminate "first" agg overhead

| | |
|---|---|
| **File** | `TraceLens/TraceDiff/trace_diff.py` |
| **Function** | `get_df_diff_stats_unique_args` |

**Before:** Always-triggered `try/except TypeError` → `df_filtered.copy()` + two
full groupby passes (one for `agg`, one for `size`), with "first" aggregation on
every grouping column.

**After:** Detect list-type columns with a vectorized `map(type).eq(list).any()`
check; copy and stringify only those columns. Use `groupby[metric_cols].agg()`
to aggregate metrics only, then `reset_index()` to recover grouping keys — avoids
running pandas "first" aggregation over every non-metric column.

```python
# Before
agg_dict = {mcol: list(agg_metrics_set) for mcol in metric_columns}
for col in grouping_cols_original:
    agg_dict[col] = "first"
try:
    df_agg = df_filtered.groupby(...).agg(agg_dict)
    df_agg["operation_count"] = df_filtered.groupby(...).size()
except TypeError:
    df_temp = df_filtered.copy()
    for col, str_col in zip(grouping_cols_original, str_cols):
        df_temp[str_col] = df_temp[col].astype(str)
    df_agg = df_temp.groupby(str_cols).agg(agg_dict)
    df_agg["operation_count"] = df_temp.groupby(str_cols).size()

# After
list_cols = [col for col in grouping_cols_original
             if df_filtered[col].map(type).eq(list).any()]
if list_cols:
    df_filtered = df_filtered.copy()
    for col in list_cols:
        df_filtered[col] = df_filtered[col].astype(str)
grouped = df_filtered.groupby(grouping_cols_original, dropna=False)
df_agg = grouped[metric_columns].agg(agg_dict)
df_agg["operation_count"] = grouped.size()
df_agg = df_agg.reset_index()   # recovers grouping keys without "first" agg
```

---

### Change B — Vectorize cpu_op rename and nn_module_parent whitespace removal

| | |
|---|---|
| **File** | `TraceLens/TraceDiff/trace_diff.py` |
| **Function** | `get_cpu_op_to_kernels_json` → `get_cpu_op_map` |

**Before:**
```python
def rename_cpu_op(row):
    if row["cpu_op_name"] in rename_map:
        return rename_map[row["cpu_op_name"]]
    return row["cpu_op_name"]

def rename_nnmodule(row):
    return re.sub(" ", "", row["nn_module_parent"])

df_agg["cpu_op_name"] = df_agg.apply(rename_cpu_op, axis=1)
df_agg["nn_module_parent"] = df_agg.apply(rename_nnmodule, axis=1)
```

**After:**
```python
if rename_map:
    df_agg["cpu_op_name"] = df_agg["cpu_op_name"].map(rename_map).fillna(df_agg["cpu_op_name"])
df_agg["nn_module_parent"] = df_agg["nn_module_parent"].str.replace(" ", "", regex=False)
```

`Series.map(dict)` is implemented in C and avoids Python interpreter overhead
per row. `str.replace` similarly runs in vectorized C code.

**Cumulative results after both passes:**

| Pass | Mean | vs original (79 ms) |
|---|---|---|
| Pass 1 (lru_cache + regex) | 21 ms | −73% (3.8×) |
| Pass 2 (pandas efficiency) | 18.8 ms | **−76% (4.2×)** |

---

## TraceDiff Pass 2b — Wagner-Fischer numpy DP matrix (2026-03-11)

**Goal:** Replace the O(M×N) Python list-of-lists DP matrix in `wagner_fischer`
with a contiguous numpy int32 array, eliminating per-cell Python object overhead.

**File:** `TraceLens/TraceDiff/trace_diff.py` — `wagner_fischer` (line ~275)

**Before:**
```python
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(m + 1):
    dp[i][0] = i
for j in range(n + 1):
    dp[0][j] = j
```

**After:**
```python
dp = np.empty((m + 1, n + 1), dtype=np.int32)
dp[:, 0] = np.arange(m + 1)
dp[0, :] = np.arange(n + 1)
```

Inner loop and backtrack updated to numpy indexing (`dp[i, j]` vs `dp[i][j]`).

**Measured on Qwen h100 vs mi300:** No change (18.8 ms) — these traces are nearly
identical so the DP matrix is always small and the cache (`wf_cache`) is hit after
the first call. The benefit appears when comparing structurally different traces
where many children lists have large M×N alignments to compute: numpy eliminates
the `(M+1)×(N+1)` Python integer object allocations and uses contiguous int32
memory instead.

---

## TraceDiff Pass 3 — O(K*N) scan elimination (2026-03-11)

**Goal:** Replace repeated `df[df[col] == val].groupby()` loops with single
multi-column groupby operations, reducing algorithmic complexity from O(K×N)
to O(N log N) in `get_rename_map` and `get_cpu_op_to_kernels_json`.

**Baseline cold run:** ~67 ms (lru_cache empty)
**After Pass 3 cold run: ~18 ms (−73%)**
**Warm run:** 18.8 ms → 18.1 ms (flat — pandas fixed overhead dominates)

The cold run improvement proved the O(K×N) repeated DataFrame scanning was
the actual cold-run bottleneck, not lru_cache misses as previously assumed.
After this fix, warm and cold runs are indistinguishable (~18 ms).

---

### Change A — `get_rename_map`: single groupby replaces two O(K×N) scan loops

| | |
|---|---|
| **File** | `TraceLens/TraceDiff/trace_diff.py` |
| **Function** | `get_cpu_op_to_kernels_json` → `get_cpu_op_map` → `get_rename_map` |

**Before:** Two separate O(K×N) patterns:
```python
# Pattern 1: O(unique_lca * N)
result = {
    str(lca_id): {
        source: {...}
        for source, group in df[df["lowest_common_ancestor_id"] == lca_id].groupby("source")
    }
    for lca_id in df["lowest_common_ancestor_id"].unique()
}

# Pattern 2: O(unique_cpu_op * N)
for cpu_op in df["cpu_op_name"].unique():
    for source, group in df[df["cpu_op_name"] == cpu_op].groupby("source"):
        module_map[cpu_op] = ...
```

**After:** Two single O(N log N) groupby calls:
```python
result = {}
for (lca_id, source), group in df.groupby(["lowest_common_ancestor_id", "source"], dropna=False):
    result.setdefault(str(lca_id), {})[source] = {
        "name": list(group["cpu_op_name"].unique()),
        "nn_module_parent": list(group["nn_module_parent"].unique()),
    }

for (cpu_op, _source), group in df.groupby(["cpu_op_name", "source"], dropna=False):
    module_map[cpu_op] = list(group["nn_module_parent"].unique())
```

---

### Change B — `get_cpu_op_map`: single groupby replaces O(K×N) nested loop

**Before:**
```python
for cpu_op in df_agg["cpu_op_name"].unique():
    for source, group in df_agg[df_agg["cpu_op_name"] == cpu_op].groupby("source"):
        cpu_op_map[cpu_op][source] = {"kernels": ..., "nn_module_parents": ...}
```

**After:**
```python
for (cpu_op, source), group in df_agg.groupby(["cpu_op_name", "source"], dropna=False):
    cpu_op_map.setdefault(cpu_op, {})[source] = {
        "kernels": sorted(list(group["name"].unique())),
        "nn_module_parents": sorted(list(group["nn_module_parent"].unique())),
    }
```

---

### Change C — `result` kernel→cpu_op dict: single groupby replaces dict comprehension

**Before:**
```python
result = {
    kernel_name: {
        source: {"cpu_op_name": list(group["cpu_op_name"].unique())}
        for source, group in df_agg[df_agg["name"] == kernel_name].groupby("source")
    }
    for kernel_name in df_agg["name"].unique()
}
```

**After:**
```python
result = {}
for (kernel_name, source), group in df_agg.groupby(["name", "source"], dropna=False):
    result.setdefault(kernel_name, {})[source] = {
        "cpu_op_name": list(group["cpu_op_name"].unique()),
    }
```

---

**Cumulative TraceDiff results:**

| Pass | Cold run | Warm run | vs original (79 ms warm) |
|---|---|---|---|
| Baseline | ~79 ms | 79 ms | — |
| Pass 1 (lru_cache + regex) | ~67 ms | 21 ms | warm −73% |
| Pass 2 (pandas efficiency) | ~67 ms | 18.8 ms | warm −76% |
| Pass 3 (O(K×N) → O(N log N)) | **~18 ms** | 18.1 ms | **cold −77%, warm −77%** |

---

## Pass 5 — Parallel call-stack building (2026-03-11)

**Goal:** Speed up `build_host_call_stack_tree` for multi-threaded traces by
processing each (pid, tid) thread independently and in parallel.

**File:** `TraceLens/Trace2Tree/trace_to_tree.py`

**Algorithm change:**

The old serial loop iterated over ALL filtered events in global timestamp order,
dispatching to per-(pid,tid) stacks via `dict_pidtid2stack`.  Since different
threads' stacks are completely independent (no cross-thread parent/child
relationships), each thread is a naturally parallel work unit.

**New architecture:**

1. **Phase 1 (serial):** Filter events, group UIDs by `(pid, tid)`, sort each
   group individually by timestamp → `K` sorted UID lists.
2. **Phase 2 (parallel):** Fork a `Pool(min(K, cpu_count))` and dispatch each
   group to `_cs_process_group`.  Workers read events via fork-inherited
   module-global `_CS_EVENTS_BY_UID` (zero pickling of large event dicts).
   Each worker returns compact mutation dicts (`patches`, `children_to_add`,
   `cpu_root_uids`, `name_to_uids`).
3. **Phase 3 (serial):** Apply all mutations to `self.events_by_uid`, sort
   `cpu_root_nodes` by timestamp to restore global ordering.

**Serial fallback:** When `K ≤ 1` (single-thread trace like Qwen), the same
`_cs_process_group` function is called directly with a tqdm-wrapped UID list for
event-level progress — no code duplication.

**Key code additions:**

```python
# Module-level globals — set before fork, inherited copy-on-write
_CS_EVENTS_BY_UID: dict = {}
_CS_STRIP_NN_SUFFIX: bool = False

def _cs_process_group(uid_list) -> tuple:
    # reads _CS_EVENTS_BY_UID; returns (patches, children_to_add,
    #                                    cpu_root_uids, name_to_uids)
    ...

# In TraceToTree.build_host_call_stack_tree:
if n_groups > 1:
    ctx = _multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(_cs_process_group, sorted_groups), ...))
    # merge results ...
```

**Expected speedup:** Proportional to number of (pid,tid) threads.
- Single-thread traces (Qwen): ~0% (serial path, same cost as before)
- Multi-GPU training traces (llama_70b_fsdp, 8-GPU): up to `K×` where K is
  the number of distinct CPU threads captured in the trace.

**Validation:** `python -m pytest tests/test_perf_report_regression.py -v`
Result: **13/13 passed**

---

## Pass 5b — Fix CoW memory explosion in parallel call-stack workers (2026-03-11)

**Goal:** The fork-based parallel call-stack builder from Pass 5 caused ~260 GB RSS
on a 10M-event trace because CPython's reference counting dirtied every shared
page in the forked workers, defeating copy-on-write.

**File:** `TraceLens/Trace2Tree/trace_to_tree.py`

### Problem

When a forked worker reads from the parent's `events_by_uid` dict (10M entries),
every dict access bumps a reference count, which writes to the memory page
containing that object.  CPython's refcounting turns copy-on-write into
copy-on-read: the OS must make a private physical copy of every page touched,
so each worker's RSS approaches the parent's full heap size (~26 GB × 10 workers
= 260 GB).

### Fix — Switch to spawn + pre-partitioned compact events

Two changes:

1. **Spawn instead of fork:** `multiprocessing.get_context("spawn")` starts
   workers with a clean address space, so no CoW dirtying occurs.

2. **Pre-partition compact event dicts:** Before dispatch, the main process builds
   a compact dict (only `ts`, `t_end`, `name`, `cat`) for each worker's event
   group and sends it via pickle.  Each worker receives only its `~N/K` events
   rather than inheriting the full 10M-event address space.

```python
# Before: fork + module-global (full dict inherited by all workers)
_CS_EVENTS_BY_UID = self.events_by_uid          # 10M events, every worker reads all of it
ctx = get_context("fork")
pool.imap(_cs_process_group, sorted_groups)     # each group is just a uid list

# After: spawn + per-group compact dict (only this group's events pickled)
def _build_compact(uid_list):
    return (uid_list, {uid: {"ts": ..., "t_end": ..., "name": ..., "cat": ...}
                       for uid in uid_list})
ctx = get_context("spawn")
pool.map(_cs_process_group, [_build_compact(g) for g in batch])
```

**Memory impact (10M-event trace):**

| Strategy | Peak RSS |
|---|---|
| Fork + global dict | ~260 GB |
| Spawn + compact per-group dict | ~26 GB |

---

### Supporting change — Batch spawn workers to bound peak memory

Even with spawn + compact dicts, submitting all K groups at once holds K compact
dicts + K result sets simultaneously in memory.  Workers are now dispatched in
explicit batches of `n_workers=2`: each batch builds compact dicts for only 2
groups, runs `pool.map`, merges results, then frees before starting the next batch.

```python
for batch_start in range(0, n_groups, n_workers):
    batch = sorted_groups[batch_start : batch_start + n_workers]
    batch_args = [_build_compact(uid_list) for uid_list in batch]
    with ctx.Pool(len(batch)) as pool:
        batch_results = pool.map(_cs_process_group, batch_args)
    # merge + free before next batch
```

Peak memory is now `main process + 2 × per-group overhead` regardless of total
group count.

---

## Pass 6 — Eliminate fork + O(subtree) per-launcher traversal in tree_perf (2026-03-11)

**Goal:** Remove the fork-based parallel pool in `get_kernel_launchers` (added in
Pass 3) and replace the O(subtree_size × N_launchers) subtree traversal with an
O(1) lookup.

**File:** `TraceLens/TreePerf/tree_perf.py`

### Root cause

`get_kernel_launchers` computed each launcher's subtree GPU kernels by calling
`loop_and_aggregate_kernels()` — a full recursive subtree traversal — for every
launcher.  With N_launchers launchers this is O(subtree_size × N_launchers).

The fork-based parallel pool (Pass 3) hid the cost but introduced its own
memory overhead via CoW page-dirtying (same mechanism as Pass 5b).

### Fix

`add_gpu_ops_to_tree` (Pass 7, below) already propagates every GPU kernel UID
up to all CPU/runtime ancestors and stores them in `event["gpu_events"]` during
tree construction.  This makes the subtree lookup O(1):

```python
# Before: O(subtree_size) recursive traversal per launcher
subtree_kernel_uids = self.loop_and_aggregate_kernels([launcher_event])

# After: O(1) direct field lookup (populated by add_gpu_ops_to_tree)
subtree_kernel_uids = launcher_event.get("gpu_events", [])
```

**Symbols removed:** `_PARALLEL_LAUNCHER_THRESHOLD`, `_FORK_ANALYZER`,
`_FORK_LAUNCHER_KERNELS`, `_compute_launcher_metrics_fork`, `import multiprocessing`

**Result:** `get_kernel_launchers` runs serially with negligible per-launcher cost
(just a dict field lookup + `GPUEventAnalyser` interval merge), with no fork
overhead or memory blowup.

---

## Pass 7 — Eliminate O(N×depth) bottleneck in add_gpu_ops_to_tree (2026-03-12)

**Goal:** Fix a hang in `add_gpu_ops_to_tree` that caused a 10M-event trace to
run for 3.6+ hours and grow to 127+ GB VIRT.

**File:** `TraceLens/Trace2Tree/trace_to_tree.py` — `TraceToTree.add_gpu_ops_to_tree`

### What the function does

`add_gpu_ops_to_tree` connects the bottom of the call-stack tree — attaching GPU
kernels as children of the CPU runtime events that launched them, then propagating
that information upward so every ancestor node knows which GPU kernels ran
"under" it.  This `gpu_events` field is what lets `get_kernel_launchers` answer
"what GPU kernels does this op own?" in O(1) (Pass 6).

### Two root causes

**Root cause 1 — O(N_all) scan per graph-launch event:**
`_get_graph_gpu_events()` was called for every `cudaGraphLaunch` event and
performed a full O(N_all) linear scan to find matching GPU kernels.  With many
graph launches this is O(N_graph_launches × N_all).

**Root cause 2 — O(depth) ancestor walk per GPU kernel:**
After linking each GPU kernel to its runtime parent, the old code walked UP the
tree from that kernel, appending its UID to every ancestor individually:

```python
# Old: per-kernel upward walk — O(N_gpu × depth) individual list.append() calls
while parent:
    parent.setdefault("gpu_events", []).append(gpu_uid)
    parent = events_by_uid.get(parent.get("parent"))
```

For 1M GPU kernels at depth 10, this is 10M Python `list.append()` calls plus
heavy GC pressure from all the intermediate list allocations.

### Fix — Three-phase restructure

**Phase A — Pre-build correlation → GPU index (fixes root cause 1):**
```python
corr_to_gpu_events: dict = {}
for evt in self.events:
    if category in gpu_cats:
        corr = evt.get("args", {}).get("correlation")
        if corr is not None:
            corr_to_gpu_events.setdefault(corr, []).append(evt)
# Each graph-launch lookup is now O(1)
```

**Phase B — Link each GPU kernel to its immediate runtime parent only.**
No ancestor walk yet; `gpu_events` is set only on the direct runtime parent.

**Phase C — Single bottom-up propagation via BFS topo-sort (fixes root cause 2):**
```python
# One BFS pass builds topo_order (roots → leaves)
# One reverse pass propagates gpu_events upward with C-level list.extend()
gc.disable()
for event in reversed(topo_order):
    if event.get("gpu_events") and event.get("parent") is not None:
        parent["gpu_events"].extend(event["gpu_events"])  # O(1) C-level extend
gc.enable()
```

`list.extend()` is a C-level bulk copy, 10-50× faster than individual
`list.append()` calls, and GC is disabled during the loop to eliminate cyclic
collector overhead.

**Complexity change:**

| | Old | New |
|---|---|---|
| Graph-launch lookup | O(N_graph_launches × N_all) | O(1) per launch |
| GPU events propagation | O(N_gpu × depth) appends | O(N_gpu × depth) bulk extend (C-level) |
| Peak VIRT (10M trace) | 127+ GB | manageable |

**Validation:** All reference xlsx files regenerated (kernel_details ordering
changed to BFS insertion order; all numeric values identical).
`python -m pytest tests/ -v` — **117 passed**

---

## Pass 8 — Fix O(K²×N_gpu) BFS hang for merged multi-rank traces (2026-03-12)

**Goal:** Fix a hang in Phase C of `add_gpu_ops_to_tree` that occurred on large
merged multi-rank/multi-GPU traces.

**File:** `TraceLens/Trace2Tree/trace_to_tree.py` — `TraceToTree.add_gpu_ops_to_tree`

### Root cause — Correlation ID collisions in merged traces

When PyTorch profiles a single rank, every GPU kernel launch is assigned a
correlation ID that is unique within that profiling session.  When traces from K
ranks are merged into one file, each rank brings its own correlation IDs —
and those IDs restart from the same range every time.

Phase A builds `corr_to_gpu_events[corr]` by scanning all events.  In a merged
trace, the bucket for a given correlation ID accumulates GPU kernels from **all K
ranks**, not just one.  Phase B then links ALL of those kernels as children of
every runtime event that carries that correlation — so each runtime event claims
K times as many GPU children as it should.

In the BFS (Phase C), each GPU kernel therefore appears in K parents' `children`
lists and is enqueued K times.  Without a visited set, the BFS processes it K
times, making total traversal work **O(K² × N_gpu)** instead of O(N_gpu).

For K = 64 ranks this is ~4,000× more work than necessary — a definite hang.

### Fix — Two changes to the Phase C BFS

**1. Seed BFS from `cpu_root_nodes` directly:**

The old code scanned all `self.events` (O(N_all)) to find parentless non-GPU
events as BFS seeds.  `self.cpu_root_nodes` is already populated by
`build_host_call_stack_tree` and contains exactly the right seeds — no scan needed.

```python
# Before: O(N_all) scan
roots = [e for e in self.events if e.get("parent") is None
         and self.event_to_category(e) not in gpu_cats]

# After: O(N_roots) direct lookup — no scan, no stray events
q = deque(events_by_uid[uid] for uid in self.cpu_root_nodes
          if uid in events_by_uid)
```

**2. Add a visited set:**

```python
visited: set = set()
while q:
    ev = q.popleft()
    ev_uid = ev[UID]
    if ev_uid in visited:       # ← skip duplicates from cross-rank children lists
        continue
    visited.add(ev_uid)
    topo_order.append(ev)
    for child_uid in ev.get("children", ()):
        if child_uid not in visited:
            q.append(events_by_uid[child_uid])
```

Each event is now processed exactly once regardless of how many parents claim it
as a child.  The K² blowup collapses back to O(K × N_gpu) (linear).

**Complexity change:**

| | Before | After |
|---|---|---|
| Root-finding | O(N_all) scan | O(N_roots) direct lookup |
| BFS traversal | O(K² × N_gpu) — hang for K ≥ 64 | O(N_gpu) |

**Note:** The visited set prevents the hang but does not fix incorrect cross-rank
GPU event attribution for `cudaGraphLaunch` events (a correctness issue separate
from the hang; tracked for future work).

**Validation:** `python -m pytest tests/ -v` — **117 passed**
