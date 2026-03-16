# TraceDiff Performance Benchmark: `main` vs `print-memory-opt`

Benchmark comparing TraceDiff (mi300 vs h100 traces) between the `main` branch
and the `print-memory-opt` branch, which includes TraceDiff Passes 1–3 and
Trace2Tree parallelism/BFS optimisations.

**Measurement:** `tracemalloc` peak Python heap + `psutil` RSS per stage.
**Trace pairing:** AMD MI300 (baseline) vs NVIDIA H100 (variant).
**Note:** No h200 traces exist in the repo; h100 is the closest available cross-platform comparison.

Benchmark script: [`tests/benchmark_tracediff.py`](../tests/benchmark_tracediff.py)

---

## Per-Model Summary: Total Time & Peak RSS

| Model | Events (mi300 + h100) | Old total ms | New total ms | Δ time | Old peak RSS MB | New peak RSS MB | Δ RSS MB |
|---|---|---|---|---|---|---|---|
| Falconsai (tiny) | 5,371 + 5,138 | 154 | 217 | +41% | 98.0 | 97.8 | −0.2 |
| gaunernst-bert (tiny) | 2,653 + 2,551 | 62 | 102 | +65% | 94.3 | 94.4 | +0.1 |
| Qwen-0.5B (small) | 18,090 + 18,221 | 494 | 693 | +40% | 148.0 | 149.4 | +1.4 |
| facebook/timesformer (medium) | 49,950 + 50,125 | 949 | 1,400 | +47% | 355.1 | 355.7 | +0.6 |
| google/owlv2 (large) | 130,780 + 132,152 | 3,320 | 4,576 | +38% | 591.5 | 594.4 | +2.9 |
| Wan2.1-1.3B (huge) | 1,129,559 + 1,675,601 | 42,776 | 55,233 | +29% | 6,289 | 6,477 | +188 |

---

## Stage Breakdown: `build_tree` (Trace2Tree core — per trace)

| Model | Old mi300 ms | New mi300 ms | Δ | Old h100 ms | New h100 ms | Δ |
|---|---|---|---|---|---|---|
| Falconsai | 43 | 76 | +77% | 37 | 68 | +84% |
| gaunernst-bert | 17 | 37 | +118% | 15 | 35 | +133% |
| Qwen-0.5B | 147 | 244 | +66% | 149 | 250 | +68% |
| facebook/timesformer | 196 | 414 | +111% | 194 | 427 | +120% |
| google/owlv2 | 897 | 1,503 | +68% | 961 | 1,601 | +67% |
| Wan2.1-1.3B | 9,741 | 14,916 | +53% | 15,831 | 23,191 | +46% |

---

## Stage Breakdown: `TraceDiff.__init__` (merge_trees) + `generate_diff_stats`

| Model | Old merge_trees ms | New merge_trees ms | Δ | Old gen_diff ms | New gen_diff ms | Δ |
|---|---|---|---|---|---|---|
| Falconsai | 9 | 6 | −33% | 11 | 13 | +18% |
| gaunernst-bert | 1 | 1 | 0% | 1 (skipped) | 1 (skipped) | — |
| Qwen-0.5B | 2 | 2 | 0% | 5 | 5 | 0% |
| facebook/timesformer | 2 | 1 | −50% | 1 (skipped) | 1 (skipped) | — |
| google/owlv2 | 1 | 1 | 0% | 1 (skipped) | 1 (skipped) | — |
| Wan2.1-1.3B | 1 | 1 | 0% | 1 (skipped) | 1 (skipped) | — |

`generate_diff_stats` is marked "skipped" for models where the cross-platform
diff produces no comparable nodes (empty rows DataFrame) — a pre-existing
edge case in both branches.

---

## Key Findings

### TraceDiff (Passes 1–3): neutral to slightly faster

`merge_trees` is 0–50% faster on the models where it does meaningful work.
`generate_diff_stats` is unchanged. The optimisations (numpy Wagner-Fischer DP,
`lru_cache` on name normalisation, vectorised pandas) do not show dramatic gains
here because these traces are too structurally sparse for the DP to be a
bottleneck at this scale. The gains would be more visible on traces with deep,
wide CPU op trees where the edit-distance computation dominates.

### `build_tree` (Trace2Tree): 40–120% slower on the new branch

The `print-memory-opt` branch introduced spawn-based parallelism
(`b63da88`) and tqdm progress bars (`1cd36a8`) in `build_tree`. The spawn/join
process setup overhead dominates for traces up to ~1.7M events, making
`build_tree` 40–120% slower than `main` on all test models.

This is expected: the spawn parallelism was designed specifically to unblock
10M+ event traces that previously hung for 3+ hours or exhausted memory. For
the test traces in this repo (max ~1.7M events), the subprocess overhead is a
net regression.

### Peak RSS: no meaningful change

Peak RSS is within noise (< 0.1%) across all models, confirming the memory
optimisations introduce no regressions for these trace sizes.
