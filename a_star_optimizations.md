# Optimization Opportunities for A* Extractors

This document collects optimization opportunities for:

- `src/extract/a_star_interleaved.rs` (`InterleavedAStarExtractor`)
- `src/extract/a_star_two_phase.rs` (`TwoPhaseAStarExtractor`)

The recommendations account for `egraph-serialize`'s representation (`EGraph.nodes: IndexMap<NodeId, Node>` and `EGraph.classes(): &IndexMap<ClassId, Class>`, both using default SipHash over `Arc<str>` keys) as well as algorithmic search-space pruning. Line numbers and "Current behavior" notes in Sections 1–4 refer to the pre-optimization `HEAD` baseline; see the Summary and Empirical Ablation Results for what was kept vs. dropped.

---

## Summary

| # | Optimization | Applies To | Status | Measured Impact |
|---|---|---|---|---|
| 1.1 | Prune dominated leaves & branch nodes (`node.cost >= min_leaf_cost`) | Both | **Dropped** | Helped vs. baseline (+1.6% / +4.7%), but on optimized code (`#1.2`+`#3.1`): **-1.6% geomean** on interleaved, **+0.4% geomean / -0.1% total** on two-phase (still +10–13% on `egg`/`flexc` two-phase) |
| 1.2 | Skip direct self-loops (`child == eqc`) with unary fast-path | Both | **Dropped** | Only +0.2% / +0.1% geomean (+2.4% / -1.1% total); dropped for simplicity & identical search space |
| 1.3 | Suppress dominated `Assign` pushes via `best_enqueued_bu` | Both | **Dropped** | -2.0% / -1.7% geomean (helped `herbie` +6%, regressed others) |
| 1.4 | Tighten `Visit` merit using resolved sibling costs | Interleaved | **Dropped** | Regressed `m3b` (`1.73× → 1.59×` geomean; `herbie` -22%, `rover` -21%) |
| 1.5 | Tie-break `Assign` before `Visit` on equal merit | Interleaved | **Dropped** | -4.0% geomean / -3.0% total (extra compare in `Ord::cmp` hurt all except `egg`) |
| 2.1 | Remove per-node `FxHashSet` & re-summing via incremental `acc_cost` | Both | **Implemented** | **-11.0% / -9.9% total; 1.20× / 1.19× geomean** (+25% `herbie`, +23% `rover`) |
| 2.2 | Carry `bottom_up_cost` in queue payload (skip `NodeState::Resolved`) | Both | **Implemented** | +3.3% / +3.5% extra geomean (combined with #2.3) |
| 2.3 | Defer building `ExtractionResult.choices` until `target` is resolved | Both | **Implemented** | Combined with #2.2; also -7% on `set_covering` two-phase |
| 2.4 | Use `eqc_state.entry(...)` to avoid double hashing on first reach | Both | **Implemented** | +0.5% / +1.1% geomean isolated |
| 2.5 | Avoid heap allocations for 0/1-parent e-classes (`Parents` enum) | Both | **Implemented** | +5.7% / +6.8% extra geomean (combined with #1.1 + #2.4) |
| 3.1 | Index nodes by `IndexMap` `u32` index (`get_full` / `get_index`) | Both | **Implemented** | **Single biggest win: -59.3% / -58.0% total** (`set_covering` -65%, `herbie` -56%) |
| 3.1b | Dense `Vec<PendingNode>` instead of `FxHashMap<u32, PendingNode>` | Two-Phase only | **Dropped** | +9.7% total / +3.4% geomean on two-phase, but regressed `flexc` -16.4% on interleaved; dropped to keep implementations symmetric & output-sensitive |
| 4.1 | Linear-time heapify for `leaves` (`Vec` → `BinaryHeap::from`) | Two-Phase | **Dropped** | Neutral (+0.5% geomean, -0.4% total); dropped for symmetry |
| 4.2 | Batch uniform-cost e-classes in Phase 1 | Two-Phase | Not tested | — |
| 4.3 | 4-ary min-heap | Both | **Dropped** | Helped `set_covering` (+12.6% / +3.5%), regressed geomean (`-2.1%` / `-0.8%`), adds `unsafe` |

---

## 1. Algorithmic & Search-Space Pruning

### 1.1 Prune dominated leaves and branch nodes in `enqueue_visit_eqc` / `enqueue_eqc`

- **Locations:** `a_star_interleaved.rs` (`enqueue_visit_eqc`, lines 150–174), `a_star_two_phase.rs` (`enqueue_eqc`, lines 118–138).
- **Current behavior:** Every leaf and every branch e-node in a newly reached e-class is enqueued.
- **Optimization:**
  1. **Single cheapest leaf:** All leaves in an e-class share the exact same `td_cost`, so only the single leaf with minimum `leaf.cost` can ever win. Enqueue at most one leaf per e-class.
  2. **Prune dominated branches (`node.cost >= min_leaf_cost`):** Because A\* already assumes non-negative node costs, any branch node `b` in the same e-class with `b.cost >= min_leaf_cost` satisfies `bottom_up_cost(b) >= b.cost >= min_leaf_cost`. Replacing any subtree rooted at `b` with the cheapest leaf preserves or improves tree cost while strictly reducing tree size, so an optimal tree never needs `b`.
- **Why it helps:**
  - In `a_star_two_phase.rs`, Phase 1 does not resolve e-classes, so without this check it pushes and fully expands the subgraphs of branch nodes like `x + 0` or `x * 1` even when the class already contains a leaf of equal or lower cost.
  - In `a_star_interleaved.rs`, it avoids bloating the heap with dominated `Visit` actions and prevents tie-breaking from expanding `b` when `b.cost == min_leaf_cost` (very common with uniform cost `1.0`).
- **Correctness note:** Relies on non-negative costs (`cost >= 0`), which A\* already requires for admissibility.

### 1.2 Skip direct self-loops (`child == eqc`)

- **Locations:** `a_star_interleaved.rs` (`visit_node`, lines 177–211), `a_star_two_phase.rs` (`visit_children`, lines 140–152).
- **Current behavior:** A branch node whose children include its own e-class is registered as a parent of its own e-class, explores its other children, and gets a delay entry.
- **Optimization:** If any child of `node` maps to `node`'s own e-class, skip `node` immediately before mutating `eqc_state` of any other child.
- **Correctness note:** No acyclic tree can contain a direct self-loop `eqc -> node -> eqc`, so pruning `node` cannot remove any valid tree and cannot make `td_cost` inadmissible for other classes.

### 1.3 Suppress dominated `Assign` pushes via `best_enqueued_bu`

- **Locations:** `a_star_interleaved.rs` (`update_parents`, lines 230–251; `visit_node`, lines 203–207), `a_star_two_phase.rs` (`update_parents`, lines 207–220).
- **Current behavior:** Every e-node that reaches `Delay(1)` (or has all children resolved) pushes an `Assign` entry into the priority queue unconditionally, even when a cheaper candidate for the same e-class is already queued.
- **Optimization:** Store `best_enqueued_bu: Cost` on unresolved `EqcState` (initialized to `min_leaf_cost` if the class has a leaf, otherwise `INFINITY`). Only push an `Assign` action when `bottom_up_cost < best_enqueued_bu`, updating `best_enqueued_bu` at the same time.
- **Why it helps:** Eliminates redundant heap pushes and later skipped pops when many equivalent branch nodes in an e-class become ready before the class is popped.

### 1.4 Interleaved-only: tighten `Visit` merit using already-resolved sibling costs

- **Location:** `a_star_interleaved.rs` (`visit_node`, lines 177–211; `run`, lines 262–284).
- **Current behavior:** When `Visit(node)` is popped at `popped_merit = td_cost + node.cost`, unresolved children are immediately expanded at `node_td_cost = td_cost + node.cost`, even if sibling children of `node` are already `EqcStatus::Resolved(c)` with large costs.
- **Optimization:** For nodes with `arity >= 2`, scan children first:
  - Sum the costs of all currently `Resolved` children (`resolved_sum`) and count `unresolved`.
  - If `unresolved == 0`, directly enqueue `Assign(node)` with `node.cost + resolved_sum`.
  - Otherwise, compute `new_merit = td_cost + node.cost + resolved_sum`. If `new_merit > popped_merit`, **re-enqueue** `Visit(node)` with merit `new_merit` and return without expanding unresolved children.
  - Only when `new_merit == popped_merit` (i.e., no newly resolved positive-cost siblings since enqueueing) expand unresolved children using `node_td_cost = popped_merit`.
- **Correctness note:**
  - Every `Resolved` child's cost is its exact optimal tree cost, so `td_cost + node.cost + resolved_sum` is a valid lower bound on any tree using `node` (preserving admissibility).
  - Because `Visit(node)` is only re-enqueued when `new_merit > popped_merit`, it can be re-enqueued at most `arity` times, guaranteeing termination.
  - Note that you must *not* simply pass `td_cost + node.cost + resolved_sum` directly to `enqueue_visit_eqc` without re-enqueueing, because another unpopped path to an unresolved child could have cost between `popped_merit` and `new_merit`; re-enqueueing ensures all cheaper paths are processed first.

### 1.5 Interleaved-only: tie-break `Assign` before `Visit`

- **Location:** `a_star_interleaved.rs` (`WithOrdRev::cmp`, lines 325–330).
- **Current behavior:** `WithOrdRev::cmp` compares only the cost (`other.1.cmp(&self.1)`), breaking ties arbitrarily.
- **Optimization:** On equal merit, order `ActionTag::Assign` higher than `ActionTag::Visit` (so `Assign` pops first from the max-heap).
- **Why it helps:** On equal-merit actions (frequent with uniform/integer costs), popping `Assign` first:
  1. Immediately terminates when `eqc == target`, avoiding wasted equal-merit `Visit` expansions.
  2. Resolves intermediate e-classes earlier, allowing equal-merit `Visit`/`Assign` entries for the same class to be skipped in O(1) and avoiding parent registration for visits targeting that class.

---

## 2. Hot-Loop Allocations & Redundant Work

### 2.1 Eliminate per-node `FxHashSet` allocations and child re-summing via incremental `acc_cost`

- **Locations:**
  - `a_star_interleaved.rs`: `delayed_eqcs` (`visit_node`, line 179), `node_child_costs` / `node_cost` (lines 87–104), `update_parents` (lines 239–241).
  - `a_star_two_phase.rs`: `unique_child_eqcs` (`visit_children`, line 142), `min_node_cost` (lines 174–186), `update_parents` (line 217).
- **Current behavior:**
  - Every visited branch node heap-allocates a fresh `FxHashSet<&ClassId>` to deduplicate child e-classes (even for unary/binary nodes).
  - `a_star_interleaved.rs` additionally does two lookups (`contains` then `insert`) per unresolved child, and `node_child_costs` allocates a `Vec<Cost>` on the heap solely to sum it and drop it (plus `vec![]` for leaves).
  - When `Delay(1)` fires in `update_parents`, both extractors re-read `egraph[parent]`, re-map every child via `egraph.nid_to_cid`, and re-query `eqc_state` for every child.
- **Optimization:**
  - Do not deduplicate children. Instead, push `node` into `parents[child]` once per unresolved child slot, count total unresolved child slots in `delay: u32`, and store an accumulator `acc_cost: Cost` initialized to `node.cost + sum(already_resolved_children)`:
    ```rust
    struct PendingNode {
        delay: u32,
        cost: Cost,
    }
    ```
    (`PendingNode` is 16 bytes — the exact same size as the current `NodeState` enum.)
  - When a child e-class resolves with `resolved_cost`, `update_parents` simply executes:
    ```rust
    pending.cost += resolved_cost;
    pending.delay -= 1;
    if pending.delay == 0 {
        // pending.cost is already the exact bottom-up cost!
    }
    ```
  - Duplicate children (e.g., `mul(x, x)`) appear twice in `parents[x]` and increment `delay` by 2, so when `x` resolves, `resolved_cost` is added twice and `delay` reaches 0 at the exact right moment.
- **Caveat:** Floating-point addition is not strictly associative, so summing child costs in resolution order rather than left-to-right child order can differ by 1 ULP for fractional floats (exact for integer costs and well within `EPSILON_ALLOWANCE`). If exact left-to-right float summation is desired, an alternative is fast-pathing arity 1 and 2 and reusing a scratch `Vec` on `self` instead of allocating a per-node `FxHashSet`.

### 2.2 Carry `bottom_up_cost` in the queue payload (remove `NodeState::Resolved`)

- **Locations:** `a_star_interleaved.rs` (lines 26–40, 140–141, 243, 287), `a_star_two_phase.rs` (lines 10–26, 110–111, 218, 241).
- **Current behavior:** `NodeState::Resolved(Cost)` is inserted into `node_state` for every leaf and every ready branch node only to be read once when the node is popped for assignment.
- **Optimization:** Store `bottom_up_cost` directly in the queue entry (e.g., `Action::Assign(u32, Cost)` in interleaved, `(u32, Cost)` payload in two-phase bottom-up).
  - Leaves and immediately-resolved branches never insert into `node_state`.
  - `update_parents` never writes back a `Resolved` state when `delay` hits 0.
  - `run` avoids looking up `self.node_state[node]` on every assignment.
  - Note: Do **not** compute `bottom_up_cost = merit - td_cost` from `f64` values, as subtraction can lose precision when `td_cost >> bottom_up_cost`. Carrying `bottom_up_cost` directly avoids rounding issues.

### 2.3 Defer building `ExtractionResult.choices` until `target` is resolved

- **Locations:** `a_star_interleaved.rs` (lines 83, 288), `a_star_two_phase.rs` (lines 169, 242).
- **Current behavior:** Every resolved e-class is inserted into `eqc_min: IndexMap<ClassId, NodeId>` during the search loop, paying a SipHash string hash and two atomic `Arc::clone`s even for e-classes never used in the extracted tree.
- **Optimization:** Store the winning node index inside `EqcStatus::Resolved { cost: Cost, choice: u32 }` (16 bytes, still smaller than `Parents(Vec)`'s 24 bytes, so `EqcStatus` does not grow). After `target` is resolved, run a small DFS/BFS from `target` following only `choice` pointers to populate `ExtractionResult.choices`.
- **Verification:** `ExtractionResult::check` (`src/extract/mod.rs`, lines 78–109) only requires `root_eclasses` and the classes reachable from them via `choices` to be present.

### 2.4 Eliminate double lookups in `eqc_state`

- **Locations:** `a_star_interleaved.rs` (lines 188–200, 151), `a_star_two_phase.rs` (lines 90–93, 119).
- **Current behavior:** First calls `self.eqc_state.get_mut(child)`, and on `None` calls `enqueue_visit_eqc` / `enqueue_eqc`, which calls `self.eqc_state.insert(child, ...)` (hashing `child` twice).
- **Optimization:** Use `self.eqc_state.entry(child)` and insert into the `Entry::Vacant` before iterating over `egraph.classes()[child].nodes` (the entry borrow ends immediately upon insertion).

### 2.5 Avoid heap allocations for 0/1-parent e-classes

- **Locations:** `a_star_interleaved.rs` (line 199), `a_star_two_phase.rs` (line 92).
- **Current behavior:** Every newly reached e-class allocates `vec![node]` (a 1-element heap buffer).
- **Optimization:** Replace `Vec<NodeIdx>` in `EqcStatus::Parents` with an inline enum (still 24 bytes, no new dependencies) or `SmallVec`:
  ```rust
  enum Parents {
      Empty,
      One(u32),
      Many(Vec<u32>),
  }
  ```
  This eliminates heap allocations for all e-classes with 0 or 1 parent.

---

## 3. Eliminating `EGraph` SipHash Lookups

- **Root cause:** In `egraph-serialize`, `EGraph.nodes` is an `IndexMap<NodeId, Node>` using default `RandomState` (SipHash) over `Arc<str>`. Every `egraph[node]` and `egraph.nid_to_cid(node)` hashes a string and probes `egraph.nodes`.
- Currently, a single branch node in `a_star_interleaved.rs` incurs ~10+ SipHash lookups on `egraph.nodes` over its lifetime (`enqueue_visit_eqc`, `enqueue_visit_node`, `run` Visit pop, `visit_node` cost & children, `update_parents`, `node_child_costs`, `node_cost`, `run` Assign pop).

### 3.1 Use `IndexMap`'s dense `u32` node indices (`get_full` / `get_index`)

- **Optimization:**
  - In `enqueue_visit_eqc` / `enqueue_eqc`, look up each node **once** via `egraph.nodes.get_full(node)` to obtain `(node_idx, nid_ref, node_data)`.
  - Store `node_idx: u32` in the priority queue, `parents`, `node_state: FxHashMap<u32, PendingNode>`, and `EqcStatus::Resolved { cost, choice }`.
  - Replace all later `egraph[node]` and `egraph.nid_to_cid(node)` calls with `egraph.nodes.get_index(node_idx as usize)` — a direct O(1) array lookup with **zero hashing**.
- **Additional benefits:**
  - `node_state` hashes a 4-byte `u32` (a single multiply in `FxHash`) instead of hashing variable-length strings and chasing `Arc<str>` pointers.
  - `parents` shrinks from 8 bytes/entry (`&NodeId`) to 4 bytes/entry (`u32`).
  - `Action::Visit(u32)` / `Action::Assign(u32, Cost)` stays 16 bytes (same 24-byte heap entry as before, while also carrying `Cost`).
  - Remains **100% sparse / output-sensitive** (no $O(|EGraph|)$ upfront conversion, preserving `a_star_interleaved`'s advantage on large e-graphs where only a small subset is explored).
- **Note on dense `Vec`s (Two-Phase trade-off):** In `a_star_two_phase.rs`, Phase 1 already visits the entire reachable subgraph from `target`. When most of the e-graph is reachable, replacing `node_state: FxHashMap<u32, ...>` (and potentially `eqc_state`) with dense `Vec`s indexed by `IndexMap` indices eliminates hashing completely, at the cost of $O(|EGraph|)$ initialization. Benchmark before choosing dense `Vec` over sparse `FxHashMap<u32, ...>`, especially for multi-root e-graphs run with `--single-root`.

---

## 4. Two-Phase-Specific & Queue Optimizations

### 4.1 Linear-time heapify for `leaves` (`a_star_two_phase.rs`)

- **Location:** `a_star_two_phase.rs` (lines 68, 112, 194).
- **Current behavior:** Phase 1 pushes leaves one by one into a `BinaryHeap` ($O(L \log L)$) even though nothing is popped until Phase 2.
- **Optimization:** Accumulate leaves into a `Vec` during Phase 1 and construct the heap with `BinaryHeap::from(vec)` in `TwoPhaseAStarBottomUpExtractor::init` ($O(L)$ Floyd heapify, better cache locality).

### 4.2 Batch uniform-cost e-classes in Phase 1 (`a_star_two_phase.rs`)

- **Location:** `a_star_two_phase.rs` (`enqueue_eqc`, lines 118–138; `run`, lines 154–161).
- **Current behavior:** Every branch e-node in an e-class is pushed individually into Phase 1's priority queue even when all branch e-nodes in the class have the same cost (e.g., uniform cost `1.0`).
- **Optimization:** Since `enqueue_eqc` already scans `class.nodes` to find `min_leaf_cost`, detect whether all remaining branch nodes share a single cost `c`. If so, push a single class-level entry at `td_cost + c` instead of `k` individual node entries, reducing Phase 1 heap size and operations proportionally to average e-class size.

### 4.3 4-ary min-heap & `PrioQueue` cleanup

- **Locations:** `a_star_interleaved.rs` (lines 317–354), `a_star_two_phase.rs` (lines 271–308).
- **Optimization:**
  - Both extractors are push-heavy (many pushed items are never popped or skipped immediately on pop). A compact 4-ary min-heap halves tree depth compared to `std::collections::BinaryHeap` (`log_4 n` vs `log_2 n`), reducing comparisons/moves on `push` and improving cache locality for child checks on `pop`.
  - **Code hygiene:** `PrioQueue` / `WithOrdRev` is duplicated across `a_star_interleaved.rs`, `a_star_two_phase.rs`, `prio_queue.rs`, and `uniform_cost_search.rs`. Also note that `WithOrdRev` derives `PartialEq`/`Eq` on both fields `(T, U)` while `Ord` compares only `U`, which technically violates Rust's `Eq`/`Ord` consistency contract (`a.cmp(b) == Equal` iff `a == b`).

---

## Final Kept Implementations & Remaining Ideas

The original baselines remain intact in `src/extract/a_star_interleaved.rs` (`a-star-interleaved`) and `src/extract/a_star_two_phase.rs` (`a-star-two-phase`). The optimized variants live in `src/extract/a_star_interleaved_optimized.rs` (`a-star-interleaved-optimized`) and `src/extract/a_star_two_phase_optimized.rs` (`a-star-two-phase-optimized`) using **completely symmetric** data structures and preserving the exact original A\* search space (zero algorithmic search-space pruning):

- **Active optimizations (identical in both files):** `#2.1` incremental `PendingNode { delay, cost }` (no per-node `FxHashSet` or child re-summing), `#2.2` queue payload `bottom_up_cost`, `#2.3` deferred `ExtractionResult.choices` DFS, `#2.4` `eqc_state.entry(...)`, `#2.5` inline `Parents`, `#3.1` dense `u32` `IndexMap` node indices (`get_full` / `get_index`) with sparse `FxHashMap<u32, PendingNode>`.
- **Measured speedup vs. `HEAD`:** ~**2.5× total** and ~**1.65–1.73× geomean** on both extractors (while being net **102 lines shorter** than `HEAD`).
- **Verification:** Passed `ExtractionResult::check` and matched `faster-bottom-up` optimal tree costs (within `1e-5`) across all **816 e-graphs / 9,643 single-root extractions** (`43` in `test_data/`, `773` in `data/`).

### Remaining Untested Ideas
- **`#4.2` Uniform-cost e-class batching** in Phase 1 of `a_star_two_phase.rs`.
- **Avoiding string hashing in `eqc_state: FxHashMap<&ClassId, EqcState>`** (e.g., dense class indexing for `a_star_two_phase.rs` where `egraph.classes()` is already populated).
- **Shared `PrioQueue` helper** deduplication across `a_star_interleaved.rs`, `a_star_two_phase.rs`, `prio_queue.rs`, and `uniform_cost_search.rs`.

---

## Empirical Ablation Results

Measured across 300 e-graphs (all suites in `data/` except tiny `lean-egg` / `dummy_examples`, 3 interleaved runs per file, pinned to CPU 1, median per file, verified identical optimal tree costs):

### Isolated Quick Wins (vs. `HEAD` Baseline)

| Variant | Interleaved Total | Interleaved Geomean | Two-Phase Total | Two-Phase Geomean | Final Decision |
|---|---|---|---|---|---|
| `v0_baseline` | 3868.8 ms | 1.000× | 3514.1 ms | 1.000× | Baseline |
| `#1.1` Leaf/branch pruning | 3806.9 ms (+1.6%) | 1.017× (+1.6%) | 3409.9 ms (+3.0%) | 1.050× (+4.7%) | **Dropped** (see direct A/B on optimized code below) |
| `#2.4` `eqc_state.entry` | 3823.7 ms (+1.2%) | 1.005× (+0.5%) | 3517.0 ms (-0.1%) | 1.012× (+1.1%) | **Kept** (tiny consistent win) |
| `#4.1` `BinaryHeap::from` | — | — | 3528.6 ms (-0.4%) | 1.005× (+0.5%) | **Dropped** (neutral; dropped for symmetry) |
| `#1.3` `best_bu` on `EqcState` | 3869.9 ms (-0.0%) | 0.980× (-2.0%) | 3491.4 ms (+0.6%) | 0.983× (-1.7%) | **Dropped** (helped `herbie` +6%, regressed others due to larger `EqcState` + extra lookup) |
| `#1.5` `Assign` > `Visit` tie-break | 3983.2 ms (-3.0%) | 0.962× (-4.0%) | — | — | **Dropped** (only helped `egg` +5%, branch in `Ord::cmp` hurt everywhere else) |

### Major Optimizations (Cumulative Ladder vs. `HEAD` Baseline)

| Variant | Interleaved Total | Interleaved Geomean | Two-Phase Total | Two-Phase Geomean | Final Decision / Notes |
|---|---|---|---|---|---|
| `v0_baseline` | 3896.1 ms | 1.000× | 3523.5 ms | 1.000× | Baseline |
| `m1`: + `#2.1` incremental `acc_cost`, no `FxHashSet` | 3468.8 ms (+11.0%) | 1.198× (+16.5%) | 3176.0 ms (+9.9%) | 1.188× (+15.8%) | **Kept** — huge universal win (+25% `herbie`, +23% `rover`, +16% `babble`) |
| `m2`: + `#3.1` `u32` `IndexMap` node indices | 1585.3 ms (**+59.3%**) | 1.586× (+36.9%) | 1479.4 ms (**+58.0%**) | 1.499× (+33.3%) | **Kept** — single biggest win (`set_covering` -65%, `herbie` -56%, `tensat` -40%) |
| `m3a`: + `#2.2` queue `bu_cost` + `#2.3` deferred `choices` | 1557.5 ms (+60.0%) | 1.638× (+38.9%) | 1404.3 ms (+60.1%) | 1.551× (+35.5%) | **Kept** — solid win (+3–4% extra geomean, -7% `set_covering` on two-phase) |
| `m3b`: + `#1.1` leaf prune + `#2.4` `entry` + `#2.5` inline `Parents` | 1555.5 ms (+60.1%) | 1.732× (+42.3%) | 1394.7 ms (+60.4%) | 1.656× (+39.6%) | **Kept `#2.4` + `#2.5`** (most of the +5.7% / +6.8% extra geomean); **`#1.1` dropped** |
| `m6`: `m3b` + `#1.2` self-loop skip (unary fast-path) | 1515.2 ms (+61.1%) | 1.735× (+42.4%) | 1409.8 ms (+60.0%) | 1.658× (+39.7%) | **Dropped** (+32% / +19% `egg`, but only +0.2% / +0.1% geomean; dropped for simplicity & identical search space) |
| `m8`: `m6` + dense `Vec<PendingNode>` | 1367.0 ms (+64.9%) | 1.775× (+43.7%) | 1259.9 ms (+64.2%) | 1.714× (+41.7%) | **Dropped** (won two-phase +9.7% total, but regressed `flexc` -16.4% on interleaved; dropped for symmetry & output-sensitivity) |
| `m4`: `m3b` + `#1.4` sibling-cost re-enqueue (interleaved) | 1629.9 ms (+58.2%) | 1.589× (+37.1%) | — | — | **Dropped** (2-pass check regressed `herbie` -22% & `rover` -21% vs `m3b`) |
| `m7`: `m6` + 4-ary hole-sift min-heap | 1433.1 ms (+63.2%) | 1.696× (+41.0%) | 1357.8 ms (+61.5%) | 1.644× (+39.2%) | **Dropped** (helped huge `set_covering` heap, regressed geomean & `rover` -27%, adds `unsafe`) |

### Direct A/B of `#1.1` on Optimized Code (`without #1.1` → `with #1.1`)

Measured on top of `m6` (interleaved) / `m8` (two-phase):

- **`a-star-interleaved`:** Total `1523.2 → 1525.0 ms` (**-0.1%**), Geomean **`0.984×` (-1.6%)** (`babble` -2.1%, `herbie` -1.8%, `impress` -1.8%, `egg` +0.3%, `flexc` +0.6%).
- **`a-star-two-phase`:** Total `1273.6 → 1274.4 ms` (**-0.1%**), Geomean **`1.004×` (+0.4%)** (`flexc` +13.5%, `egg` +10.2%, `babble` -0.9%, `set_covering` -0.7%).

