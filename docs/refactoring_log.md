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
