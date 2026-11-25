# Optimization Opportunities for A* Extractors

This document collects optimization opportunities for:

- `src/extract/a_star_interleaved.rs` (`InterleavedAStarExtractor`)
- `src/extract/a_star_two_phase.rs` (`TwoPhaseAStarExtractor`)

The recommendations account for `egraph-serialize`'s representation (`EGraph.nodes: IndexMap<NodeId, Node>` and `EGraph.classes(): &IndexMap<ClassId, Class>`, both using default SipHash over `Arc<str>` keys) as well as algorithmic search-space pruning. Line numbers and "Current behavior" notes in Sections 1–4 refer to the original baselines in `a_star_interleaved.rs` / `a_star_two_phase.rs`; see the Summary and Empirical Ablation Results for isolated measurements and what was kept vs. dropped in `a_star_interleaved_optimized.rs` / `a_star_two_phase_optimized.rs`.

---

## Summary

| # | Optimization | Applies To | Status | Isolated Impact vs. Baseline (Interleaved / Two-Phase) |
|---|---|---|---|---|
| 1.1 | Prune dominated leaves & branch nodes (`node.cost >= min_leaf_cost`) | Both | **Dropped** | `-0.9%` / `+0.1%` total; `0.936×` / `1.006×` geomean (on optimized code: `-6.8%` / `-5.6%` total, `0.872×` / `0.919×` geomean) |
| 1.2 | Skip direct self-loops (`child == eqc`) | Both | **Dropped** | `-1.8%` / `-3.3%` total; `0.940×` / `0.941×` geomean (helped `egg` `+41%`/`+23%`, regressed `rover`/`eggcc-bril`) |
| 1.3 | Suppress dominated `Assign` pushes via `best_enqueued_bu` | Both | **Dropped** | `+0.9%` / `+0.4%` total; `0.967×` / `0.972×` geomean (helped `herbie` `+5–7%`, regressed geomean) |
| 1.4 | Tighten `Visit` merit using resolved sibling costs | Interleaved | **Dropped** | `-6.5%` total; `0.890×` geomean (regressed across suites: `diospyros -34%`, `herbie -24%`, `rover -23%`) |
| 1.5 | Tie-break `Assign` before `Visit` on equal merit | Interleaved | **Dropped** | `-2.7%` total; `0.955×` geomean (only helped `egg` `+23%`, hurt other suites) |
| 2.1 | Remove per-node `FxHashSet` via delay counter (originally incremental `acc_cost`; later generalized to `node_delay` + `common::node_cost` at `delay == 0`) | Both | **Implemented** | **`+11.9%` / `+10.5%` total; `1.276×` / `1.210×` geomean** (universal win across suites; measured with incremental `acc_cost`) |
| 2.2 | Carry `bottom_up_cost` in queue payload (skip `NodeState::Resolved`) | Both | **Implemented** | **`+4.9%` / `+9.1%` total; `1.008×` / `1.053×` geomean** (LOO confirms `+2.7%` / `+0.5%` geomean, `+0.4%` / `+1.2%` total) |
| 2.3 | Defer building `ExtractionResult.choices` until `target` is resolved | Both | **Dropped** | `+1.0%` / `+3.5%` total; `0.992×` / `1.047×` geomean (LOO: `-1.2%` / `-0.4%` total, `+1.4%` / `+1.3%` geomean; hurts `maxsat`/`constructed` & adds ~20 LOC DFS) |
| 2.4 | Use `eqc_state.entry(...)` to avoid double hashing on first reach | Both | **Dropped** | `+1.6%` / `+0.8%` total; `1.015×` / `1.007×` geomean (LOO: `-0.1%` / `-1.8%` total, `+0.8%` / `-0.5%` geomean) |
| 2.5 | Avoid heap allocations for 0/1-parent e-classes (`Parents` enum) | Both | **Dropped** | `-0.1%` / `+3.1%` total; `1.016×` / `1.083×` geomean (LOO: `+0.4%` / `-2.1%` total, `+5.9%` / `+2.8%` geomean; dropped for simplicity — ~30 LOC/file) |
| 3.1 | Index nodes by `IndexMap` `u32` index (`get_full` / `get_index`) | Both | **Implemented** | **Single biggest win: `+53.6%` / `+53.6%` total; `1.347×` / `1.338×` geomean** (`set_covering +64–66%`, `herbie +37–38%`) |
| 3.1b | Dense `Vec` for `node_state` instead of `FxHashMap` (requires `#3.1`) | Both | **Dropped** | `+58.2%` / `+59.9%` total; `1.477×` / `1.473×` geomean vs. baseline (on optimized: `+11.0%`/`+10.0%` total, but `flexc -28%` / `egg -45%`; not output-sensitive) |
| 4.1 | Linear-time heapify for `leaves` (`Vec` → `BinaryHeap::from`) | Two-Phase | **Dropped** | `+2.4%` total; `1.045×` geomean (two-phase only; dropped for symmetry) |
| 4.2 | Batch uniform-cost e-classes in Phase 1 | Two-Phase | **Dropped** | `+2.5%` total; `1.008×` geomean isolated, but `-11.9%` total / `0.938×` geomean on optimized code (re-hashes class nodes on pop) |
| 4.3 | 4-ary min-heap | Both | **Dropped** | `+5.0%` / `+5.2%` total; `0.973×` / `1.024×` geomean (hurts interleaved geomean & adds custom heap maintenance) |
| 4.4 | Bucketed `BTreeMap<Cost, Vec<...>>` priority queue (replaces `BinaryHeap`) | Both (+ UCS/PQ opt) | **Implemented** | On optimized Interleaved (`9,569` roots): **`+18.5%` total** (`+21.3%` / `1.241×` geomean on `>= 1 ms`), `1.002×` overall geomean; deletes `WithOrdRev`/`PrioQueue` boilerplate |

---

## 1. Algorithmic & Search-Space Pruning

### 1.1 Prune dominated leaves and branch nodes in `enqueue_visit_eqc` / `enqueue_eqc`

- **Locations:** `a_star_interleaved.rs` (`enqueue_visit_eqc`, lines 150–174), `a_star_two_phase.rs` (`enqueue_eqc`, lines 118–138).
- **Current behavior:** Every leaf and every branch e-node in a newly reached e-class is enqueued.
- **Optimization:**
  1. **Single cheapest leaf:** All leaves in an e-class share the exact same `td_cost`, so only the single leaf with minimum `leaf.cost` can ever win. Enqueue at most one leaf per e-class.
  2. **Prune dominated branches (`node.cost >= min_leaf_cost`):** Because A\* already assumes non-negative node costs, any branch node `b` in the same e-class with `b.cost >= min_leaf_cost` satisfies `bottom_up_cost(b) >= b.cost >= min_leaf_cost`. Replacing any subtree rooted at `b` with the cheapest leaf preserves or improves tree cost while strictly reducing tree size, so an optimal tree never needs `b`.
- **Why it did not pay off:** Performing two passes over `egraph.classes()[eqc].nodes` (first to find `min_leaf_cost`, then to enqueue surviving branches) doubles `egraph.nodes` SipHash lookups (`egraph[node]` / `get_full`) on every reached e-class. That overhead outweighs the pruning savings on most suites, especially on top of the optimized code (`-6.8%` / `-5.6%` total, `0.872×` / `0.919×` geomean).
- **Correctness note:** Relies on non-negative costs (`cost >= 0`), which A\* already requires for admissibility.

### 1.2 Skip direct self-loops (`child == eqc`)

- **Locations:** `a_star_interleaved.rs` (`visit_node`, lines 177–211), `a_star_two_phase.rs` (`visit_children`, lines 140–152).
- **Current behavior:** A branch node whose children include its own e-class is registered as a parent of its own e-class, explores its other children, and gets a delay entry.
- **Optimization:** If any child of `node` maps to `node`'s own e-class, skip `node` immediately before mutating `eqc_state` of any other child.
- **Why it did not pay off:** Checking `egraph.nid_to_cid(child) == eqc` up front adds extra SipHash lookups on every non-self-loop branch node. While it speeds up `egg` (`+41%` / `+23%`), it regresses most other suites (`-1.8%` / `-3.3%` total, `0.940×` / `0.941×` geomean).
- **Correctness note:** No acyclic tree can contain a direct self-loop `eqc -> node -> eqc`, so pruning `node` cannot remove any valid tree and cannot make `td_cost` inadmissible for other classes.

### 1.3 Suppress dominated `Assign` pushes via `best_enqueued_bu`

- **Locations:** `a_star_interleaved.rs` (`update_parents`, lines 230–251; `visit_node`, lines 203–207), `a_star_two_phase.rs` (`update_parents`, lines 207–220).
- **Current behavior:** Every e-node that reaches `Delay(1)` (or has all children resolved) pushes an `Assign` entry into the priority queue unconditionally, even when a cheaper candidate for the same e-class is already queued.
- **Optimization:** Store `best_enqueued_bu: Cost` on unresolved `EqcState` (initialized to `INFINITY`). Only push an `Assign` action when `bottom_up_cost < best_enqueued_bu`, updating `best_enqueued_bu` at the same time.
- **Why it did not pay off:** Growing `EqcState` by 8 bytes and adding a lookup/write on every `Assign` push hurts cache footprint and geomean (`0.967×` / `0.972×`) despite helping `herbie` (`+5–7%`).

### 1.4 Interleaved-only: tighten `Visit` merit using already-resolved sibling costs

- **Location:** `a_star_interleaved.rs` (`visit_node`, lines 177–211; `run`, lines 262–284).
- **Current behavior:** When `Visit(node)` is popped at `popped_merit = td_cost + node.cost`, unresolved children are immediately expanded at `node_td_cost = td_cost + node.cost`, even if sibling children of `node` are already `EqcStatus::Resolved(c)` with large costs.
- **Optimization:** For nodes with `arity >= 2`, scan children first:
  - Sum the costs of all currently `Resolved` children (`resolved_sum`) and count `unresolved`.
  - If `unresolved == 0`, directly enqueue `Assign(node)` with `node.cost + resolved_sum`.
  - Otherwise, compute `new_merit = td_cost + node.cost + resolved_sum`. If `new_merit > popped_merit`, **re-enqueue** `Visit(node)` with merit `new_merit` and return without expanding unresolved children.
  - Only when `new_merit == popped_merit` (i.e., no newly resolved positive-cost siblings since enqueueing) expand unresolved children using `node_td_cost = popped_merit`.
- **Why it did not pay off:** The extra pre-scan over children and re-enqueueing overhead regressed almost every suite (`-6.5%` total, `0.890×` geomean).
- **Correctness note:**
  - Every `Resolved` child's cost is its exact optimal tree cost, so `td_cost + node.cost + resolved_sum` is a valid lower bound on any tree using `node` (preserving admissibility).
  - Because `Visit(node)` is only re-enqueued when `new_merit > popped_merit`, it can be re-enqueued at most `arity` times, guaranteeing termination.
  - Note that you must *not* simply pass `td_cost + node.cost + resolved_sum` directly to `enqueue_visit_eqc` without re-enqueueing, because another unpopped path to an unresolved child could have cost between `popped_merit` and `new_merit`; re-enqueueing ensures all cheaper paths are processed first.

### 1.5 Interleaved-only: tie-break `Assign` before `Visit`

- **Location:** `a_star_interleaved.rs` (`WithOrdRev::cmp`, lines 325–330).
- **Current behavior:** `WithOrdRev::cmp` compares only the cost (`other.1.cmp(&self.1)`), breaking ties arbitrarily.
- **Optimization:** On equal merit, order `ActionTag::Assign` higher than `ActionTag::Visit` (so `Assign` pops first from the max-heap).
- **Why it did not pay off:** While tie-breaking helped `egg` (`+22.8%`), the extra comparison branch inside `Ord::cmp` on every heap sift hurt overall runtime (`-2.7%` total, `0.955×` geomean).

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
    (`PendingNode` is 16 bytes — the exact same size as the original `NodeState` enum.)
  - When a child e-class resolves with `resolved_cost`, `update_parents` simply executes:
    ```rust
    pending.cost += resolved_cost;
    pending.delay -= 1;
    if pending.delay == 0 {
        // pending.cost is already the exact bottom-up cost!
    }
    ```
  - Duplicate children (e.g., `mul(x, x)`) appear twice in `parents[x]` and increment `delay` by 2, so when `x` resolves, `resolved_cost` is added twice and `delay` reaches 0 at the exact right moment.
- **Caveat & subsequent generalization:** Floating-point addition is not strictly associative, so summing child costs in resolution order rather than left-to-right child order can differ by 1 ULP for fractional floats, and incremental `+=` restricts `*_optimized` to associative/commutative additive folds. To support arbitrary non-additive cost functions (see `non_additive_cost_example.md`) while keeping the `FxHashSet`-removal win, the final `*_optimized` extractors replace `PendingNode { delay, cost }` with a plain `node_delay: FxHashMap<u32, u32>` counter, store `EqcStatus::Resolved(Cost)` in `common.rs`, and evaluate a shared `common::node_cost` helper once when `delay` reaches `0` (and for leaves). This shrinks `node_delay` entries from 16 bytes to 4, preserves exact left-to-right child order, and still avoids per-node `FxHashSet` child deduplication and per-pop re-evaluation.

### 2.2 Carry `bottom_up_cost` in the queue payload (remove `NodeState::Resolved`)

- **Locations:** `a_star_interleaved.rs` (lines 26–40, 140–141, 243, 287), `a_star_two_phase.rs` (lines 10–26, 110–111, 218, 241).
- **Current behavior:** `NodeState::Resolved(Cost)` is inserted into `node_state` for every leaf and every ready branch node only to be read once when the node is popped for assignment.
- **Optimization:** Store `bottom_up_cost` directly in the queue entry (e.g., `Action::Assign(u32, Cost)` in interleaved, `(u32, Cost)` payload in two-phase bottom-up).
  - Leaves and immediately-resolved branches never insert into `node_state` / `node_delay`.
  - `update_parents` never writes back a `Resolved` state when `delay` hits 0.
  - `run` avoids looking up `self.node_state[node]` on every assignment.
  - Note: Do **not** compute `bottom_up_cost = merit - td_cost` from `f64` values, as subtraction can lose precision when `td_cost >> bottom_up_cost`. Carrying `bottom_up_cost` directly avoids rounding issues.

### 2.3 Defer building `ExtractionResult.choices` until `target` is resolved

- **Locations:** `a_star_interleaved.rs` (lines 83, 288), `a_star_two_phase.rs` (lines 169, 242).
- **Current behavior:** Every resolved e-class is inserted into `eqc_min: IndexMap<ClassId, NodeId>` during the search loop, paying a SipHash string hash and two atomic `Arc::clone`s even for e-classes never used in the extracted tree.
- **Optimization:** Store the winning node index inside `EqcStatus::Resolved { cost: Cost, choice: u32 }` (interleaved) or `EqcStatus::Resolved(u32)` (two-phase). After `target` is resolved, run a DFS from `target` following only `choice` pointers to populate `ExtractionResult.choices`.
- **Why it was dropped:** In isolation this helps Two-Phase (`+3.5%` total, `1.047×` geomean) but is slightly negative on Interleaved geomean (`0.992×`). In leave-one-out on top of the optimized code, `#2.3` slightly improves geomean (`+1.4%` / `+1.3%`) on suites where many unused classes resolve (`babble`, `tensat`, `impress`), but **hurts total runtime** (`-1.2%` / `-0.4%`, with `maxsat -20.8%/-13.9%` and `constructed -17.6%/-17.8%`) because the post-search DFS invokes `egraph.nid_to_cid` (SipHash) for every child edge in the extracted tree, while adding ~20 lines of extra DFS code per extractor. Dropped for faster total runtime and simpler code.
- **Verification:** `ExtractionResult::check` (`src/extract/mod.rs`, lines 78–109) only requires `root_eclasses` and the classes reachable from them via `choices` to be present.

### 2.4 Eliminate double lookups in `eqc_state`

- **Locations:** `a_star_interleaved.rs` (lines 188–200, 151), `a_star_two_phase.rs` (lines 90–93, 119).
- **Current behavior:** First calls `self.eqc_state.get_mut(child)`, and on `None` calls `enqueue_visit_eqc` / `enqueue_eqc`, which calls `self.eqc_state.insert(child, ...)` (hashing `child` twice).
- **Optimization:** Use `self.eqc_state.entry(child)` and insert into the `Entry::Vacant` before iterating over `egraph.classes()[child].nodes` (the entry borrow ends immediately upon insertion).
- **Why it was dropped:** In isolation this yields a tiny gain (`+1.6%`/`+0.8%` total, `1.015×`/`1.007×` geomean), but in leave-one-out on the optimized code it is **essentially neutral on Interleaved** (`-0.1%` total, `+0.8%` geomean) and **slightly hurts Two-Phase** (`-1.8%` total, `-0.5%` geomean) because the common occupied fast-path is slightly heavier with `Entry` than with `get_mut`. Dropped.

### 2.5 Avoid heap allocations for 0/1-parent e-classes

- **Locations:** `a_star_interleaved.rs` (line 199), `a_star_two_phase.rs` (line 92).
- **Current behavior:** Every newly reached e-class allocates `vec![node]` (a 1-element heap buffer).
- **Optimization:** Replace `Vec<u32>` in `EqcStatus::Parents` with an inline enum (still 24 bytes, no new dependencies):
  ```rust
  enum Parents {
      Empty,
      One(u32),
      Many(Vec<u32>),
  }
  ```
  This eliminates heap allocations for all e-classes with 0 or 1 parent.
- **Why it was dropped (note):** `#2.5` is a genuine per-file geomean win in both isolation (`1.016×` / `1.083×`) and leave-one-out (`+5.9%` / `+2.8%`, especially `+15–18%` on `constructed` and `+10–12%` on `impress`), because many e-classes in smaller/sparser graphs have only 0 or 1 parent. However, it adds ~30 lines of boilerplate per file (`Parents` enum + `update_parent`/`update_parents` split) and slightly slows Two-Phase total time (`-2.1%` in LOO due to enum-dispatch overhead on high-fanin suites like `tensat` and `rover`). Dropped to keep the optimized extractors minimal (`#2.1` + `#2.2` + `#3.1`), but easy to re-enable if geomean on small/sparse graphs becomes a priority.

---

## 3. Eliminating `EGraph` SipHash Lookups

- **Root cause:** In `egraph-serialize`, `EGraph.nodes` is an `IndexMap<NodeId, Node>` using default `RandomState` (SipHash) over `Arc<str>`. Every `egraph[node]` and `egraph.nid_to_cid(node)` hashes a string and probes `egraph.nodes`.
- In the baseline, a single branch node in `a_star_interleaved.rs` incurs ~10+ SipHash lookups on `egraph.nodes` over its lifetime (`enqueue_visit_eqc`, `enqueue_visit_node`, `run` Visit pop, `visit_node` cost & children, `update_parents`, `node_child_costs`, `node_cost`, `run` Assign pop).

### 3.1 Use `IndexMap`'s dense `u32` node indices (`get_full` / `get_index`)

- **Optimization:**
  - In `enqueue_eqc_nodes`, look up each node **once** via `egraph.nodes.get_full(node)` to obtain `(node_idx, nid_ref, node_data)`.
  - Store `node_idx: u32` in the priority queue, `parents`, `node_delay: FxHashMap<u32, u32>` (originally `node_state: FxHashMap<u32, PendingNode>`), and `EqcStatus`.
  - Replace all later `egraph[node]` and `egraph.nid_to_cid(node)` calls with `egraph.nodes.get_index(node_idx as usize)` — a direct O(1) array lookup with **zero hashing**.
- **Additional benefits:**
  - `node_state` hashes a 4-byte `u32` (a single multiply in `FxHash`) instead of hashing variable-length strings and chasing `Arc<str>` pointers.
  - `parents` shrinks from 8 bytes/entry (`&NodeId`) to 4 bytes/entry (`u32`).
  - `Action::Visit(u32)` / `Action::Assign(u32, Cost)` stays 16 bytes (same 24-byte heap entry as before, while also carrying `Cost`).
  - Remains **100% sparse / output-sensitive** (no $O(|EGraph|)$ upfront conversion, preserving `a_star_interleaved`'s advantage on large e-graphs where only a small subset is explored).

### 3.1b Dense `Vec` for `node_state` instead of `FxHashMap<u32, ...>`

- **Optimization:** Building on `#3.1`'s `u32` indices, store `node_state` in a dense `Vec` of length `egraph.nodes.len()` instead of a sparse `FxHashMap<u32, ...>`, eliminating `node_state` hashing altogether.
- **Why it was dropped:** Even though it wins another `+10–11%` total time on top of the optimized code (`set_covering +15–16%`, `herbie +7–9%`), initializing an $O(|EGraph|)$ `Vec` per single-root extraction severely regresses multi-root / sparse-reachability suites (`flexc -28.3%` on interleaved, `egg -45.2%` on two-phase), leaving optimized geomean neutral-to-negative (`0.989×` interleaved, `1.003×` two-phase). Dropped to preserve output-sensitivity.

---

## 4. Two-Phase-Specific & Queue Optimizations

### 4.1 Linear-time heapify for `leaves` (`a_star_two_phase.rs`)

- **Location:** `a_star_two_phase.rs` (lines 68, 112, 194).
- **Current behavior:** Phase 1 pushes leaves one by one into a `BinaryHeap` ($O(L \log L)$) even though nothing is popped until Phase 2.
- **Optimization:** Accumulate leaves into a `Vec` during Phase 1 and construct the heap with `BinaryHeap::from(vec)` in `TwoPhaseAStarBottomUpExtractor::init` ($O(L)$ Floyd heapify, better cache locality).
- **Result:** Modest win in isolation (`+2.4%` total, `1.045×` geomean), dropped to keep the two implementations structurally symmetric.

### 4.2 Batch uniform-cost e-classes in Phase 1 (`a_star_two_phase.rs`)

- **Location:** `a_star_two_phase.rs` (`enqueue_eqc`, lines 118–138; `run`, lines 154–161).
- **Current behavior:** Every branch e-node in an e-class is pushed individually into Phase 1's priority queue even when all branch e-nodes in the class have the same cost (e.g., uniform cost `1.0`).
- **Optimization:** In `enqueue_eqc`, scan `class.nodes` to detect whether all branch nodes share a single cost `c`. If there are $\ge 2$ branch nodes with identical cost `c`, push a single class-level entry at `td_cost + c` instead of `k` individual node entries.
- **Why it did not pay off:** In isolation against baseline it gave `+2.5%` total / `1.008×` geomean (`egg +17.7%`), but on top of the optimized code (`#3.1`) popping a class-level entry has to re-resolve each `NodeId` via `egraph.nodes.get_full` (SipHash), regressing runtime by `-11.9%` total / `0.938×` geomean.

### 4.3 4-ary min-heap & `PrioQueue` cleanup

- **Locations:** `a_star_interleaved.rs` (lines 317–354), `a_star_two_phase.rs` (lines 271–308).
- **Optimization:**
  - A compact 4-ary min-heap halves tree depth compared to `std::collections::BinaryHeap` (`log_4 n` vs `log_2 n`), reducing comparisons/moves on `push` and improving cache locality for child checks on `pop`.
  - **Result:** Helped large heaps (`+5.0%` / `+5.2%` total, `set_covering +6.8%` / `+5.3%`), but regressed interleaved geomean (`0.973×`) and adds custom heap code; dropped.
  - **Code hygiene (baseline only):** `PrioQueue` / `WithOrdRev` is duplicated across the baseline files `a_star_interleaved.rs`, `a_star_two_phase.rs`, `prio_queue.rs`, and `uniform_cost_search.rs` (removed in all `*_optimized` variants via `#4.4`). Also note that `WithOrdRev` derives `PartialEq`/`Eq` on both fields `(T, U)` while `Ord` compares only `U`, which technically violates Rust's `Eq`/`Ord` consistency contract (`a.cmp(b) == Equal` iff `a == b`).

### 4.4 Bucketed `BTreeMap<Cost, Vec<...>>` priority queue

- **Locations:** `a_star_interleaved_optimized.rs`, `a_star_two_phase_optimized.rs`, `uniform_cost_search_optimized.rs`, `prio_queue_optimized.rs`.
- **Current behavior (baseline):** Each extractor defines a ~38-line local `PrioQueue` / `WithOrdRev` min-heap wrapper around `std::collections::BinaryHeap`, storing `(payload, Cost)` in every heap slot and performing $O(\log N)$ sifts on every push/pop.
- **Optimization:** Replace `PrioQueue` with `std::collections::BTreeMap<Cost, Vec<T>>`:
  - **Insert:** `queue.entry(cost).or_default().push(item)`
  - **Pop:** pop from the back of `queue.first_entry()`'s `Vec` (removing the entry when empty).
  - Deletes the entire `WithOrdRev` / `PrioQueue` module (~30–40 LOC per file) and requires neither `Ord` nor `Eq` on the queued payload.
- **Why it wins:**
  1. **Cost multiplicity:** Most benchmark suites use a handful of discrete integer node costs (`1`–`7` distinct node costs per file across `set_covering`, `herbie`, `maxsat`, `egg`, `eggcc-bril`, `flexc`), so the `BTreeMap` only holds $K \ll N$ distinct active costs. `Cost` is stored once per bucket rather than per item, and most operations reduce to contiguous `Vec::push`/`Vec::pop`.
  2. **LIFO tie-breaking within equal-cost buckets:** Popping from the back of the `Vec` pops newly-pushed `Assign` actions at the current minimum cost immediately instead of interleaving them with thousands of pending equal-cost `Visit`s, propagating assignments toward `target` faster.
  3. **Trade-off on distinct-cost graphs:** When costs are mostly distinct (`constructed`: 4,001 distinct node costs / 7,999 nodes; `rover`: ~176 distinct node costs/file), buckets shrink to 1–2 elements, causing regressions (`constructed -18.1%`, `rover` large instances `-12%` to `-16%`). LIFO tie-breaking can also occasionally hit an unproductive equal-cost plateau first (`herbie/356[18]` regressed `2.44×` while `herbie/5799[15]` sped up `3.55×`).
- **Theoretical complexity:** With $N$ queued items, $E$ explored child edges, and $K \le N$ distinct active priority costs in the queue, `BTreeMap<Cost, Vec<T>>` takes $O(E + N \log K)$ time and $O(N)$ space — the **exact same $O(E + N \log N)$ worst-case bound** as `BinaryHeap` when $K = \Theta(N)$, and strictly tighter ($O(\log K)$ vs. $O(\log N)$ per queue op) when $K \ll N$.

---

## Final Kept Implementations & Remaining Ideas

The original baselines remain intact in `src/extract/a_star_interleaved.rs` (`a-star-interleaved`) and `src/extract/a_star_two_phase.rs` (`a-star-two-phase`). The optimized variants live in `src/extract/a_star_interleaved_optimized.rs` (`a-star-interleaved-optimized`) and `src/extract/a_star_two_phase_optimized.rs` (`a-star-two-phase-optimized`) using **completely symmetric** data structures and preserving the exact original A\* search space (zero algorithmic search-space pruning):

- **Active optimizations (identical in both files, and mirrored in `uniform_cost_search_optimized.rs` & `prio_queue_optimized.rs`):** `#2.1` delay-only `node_delay: FxHashMap<u32, u32>` (no per-node `FxHashSet` child deduplication; evaluates shared `common::node_cost` once at `delay == 0` using `EqcStatus::Resolved(Cost)` so non-additive cost functions are supported directly), `#2.2` queue payload `bottom_up_cost`, `#3.1` dense `u32` `IndexMap` node indices (`get_full` / `get_index`) with sparse `FxHashMap<u32, u32>`, and `#4.4` bucketed `BTreeMap<Cost, Vec<...>>` priority queue.
- **Removed after LOO ablation:** `#2.3` (deferred `choices` DFS), `#2.4` (`eqc_state.entry(...)`), and `#2.5` (inline `Parents` enum) were initially included in the `combined` benchmark below (`#2.1`–`#2.5` + `#3.1`, achieving `2.71×` / `2.62×` total and `1.863×` / `1.713×` geomean), then removed from the final optimized implementations: `#2.3` hurt total runtime (`-1.2%`/`-0.4%`) while adding ~20 LOC of DFS per file, `#2.4` was neutral-to-negative (`-0.1%`/`-1.8%` total), and `#2.5` was dropped for simplicity (~30 LOC/file) and slightly faster Two-Phase total time (`+2.1%`) despite its `+5.9%`/`+2.8%` geomean benefit on smaller/low-fanin graphs.
- **Verification:** Every isolated variant and the optimized implementations passed `ExtractionResult::check` and matched baseline optimal tree costs (within `1e-5`).

### Remaining Untested Ideas
- **Avoiding string hashing in `eqc_state: FxHashMap<&ClassId, EqcState>`** (e.g., dense `u32` `IndexMap` class indices via `egraph.classes().get_full(...)` / `get_index(...)`).
- **Shared `PrioQueue` helper** deduplication across the untouched baseline extractors (`a_star_interleaved.rs`, `a_star_two_phase.rs`, `prio_queue.rs`, and `uniform_cost_search.rs`).

---

## Empirical Ablation Results

Measured in-process across all **310 e-graphs** (9,070 single-root extractions; all suites in `data/` except tiny `lean-egg` and `dummy_examples`), 3 interleaved runs per file under `scripts/benchmode.py` (fixed base frequency, SMT/boost disabled, isolated CPU), taking the per-file median in nanoseconds. Every row below was evaluated **in isolation directly against `v0_baseline`** (except `#3.1b`, which structurally requires `#3.1`'s `u32` indices, and `combined`, which combines `#2.1`–`#2.5` + `#3.1`); every variant passed `ExtractionResult::check` and matched `v0_baseline` optimal tree costs (`< 1e-5`). Positive percentages indicate runtime reduction (faster).

| Variant | Interleaved Total (vs. `v0`) | Interleaved Geomean | Two-Phase Total (vs. `v0`) | Two-Phase Geomean | Status & Notes (incl. on-optimized / LOO checks) |
|---|---|---|---|---|---|
| `v0_baseline` | `4166.1 ms` | `1.000×` | `3759.4 ms` | `1.000×` | Baseline (`a_star_interleaved.rs` / `a_star_two_phase.rs`) |
| `#1.1` Leaf & branch pruning | `4203.7 ms` (`-0.9%`) | `0.936×` (`-6.8%`) | `3755.3 ms` (`+0.1%`) | `1.006×` (`+0.6%`) | **Dropped** — double class scan hurts interleaved even in isolation; on `kept_optimized` regresses both (`-6.8%` total / `0.872×` geomean interleaved; `-5.6%` total / `0.919×` geomean two-phase) |
| `#1.2` Skip direct self-loops | `4239.2 ms` (`-1.8%`) | `0.940×` (`-6.4%`) | `3883.7 ms` (`-3.3%`) | `0.941×` (`-6.3%`) | **Dropped** — helped `egg` (`+41.4%` / `+22.8%`), but extra `nid_to_cid` checks hurt `rover` (`-22.8%` / `-18.2%`) and geomean |
| `#1.3` `best_enqueued_bu` on `EqcState` | `4128.0 ms` (`+0.9%`) | `0.967×` (`-3.4%`) | `3746.0 ms` (`+0.4%`) | `0.972×` (`-2.8%`) | **Dropped** — helped `herbie` (`+5.4%` / `+6.8%`), but larger `EqcState` + extra lookup hurt geomean |
| `#1.4` Sibling-cost `Visit` tightening | `4438.5 ms` (`-6.5%`) | `0.890×` (`-12.4%`) | — | — | **Dropped** — 2-pass check regressed across suites (`diospyros -33.9%`, `herbie -23.6%`, `rover -22.9%`) |
| `#1.5` `Assign` > `Visit` tie-break | `4278.0 ms` (`-2.7%`) | `0.955×` (`-4.7%`) | — | — | **Dropped** — helped `egg` (`+22.8%`), but extra compare in `Ord::cmp` hurt other suites |
| `#2.1` Incremental `acc_cost` (no `FxHashSet`) | `3669.8 ms` (`+11.9%`) | `1.276×` (`+21.6%`) | `3364.6 ms` (`+10.5%`) | `1.210×` (`+17.4%`) | **Kept** — huge universal win (`rover +30%/+26%`, `herbie +24%/+23%`, `babble +22%`) |
| `#2.2` Queue payload `bottom_up_cost` | `3960.1 ms` (`+4.9%`) | `1.008×` (`+0.8%`) | `3417.4 ms` (`+9.1%`) | `1.053×` (`+5.1%`) | **Kept** — solid win in isolation (`set_covering +6.7%/+11.3%`, `maxsat +8.8%/+9.0%`); LOO confirms `+0.4%`/`+1.2%` total & `+2.7%`/`+0.5%` geomean |
| `#2.3` Defer `ExtractionResult.choices` DFS | `4125.9 ms` (`+1.0%`) | `0.992×` (`-0.8%`) | `3628.5 ms` (`+3.5%`) | `1.047×` (`+4.5%`) | **Dropped** — helps `babble`/`tensat`, but hurts `maxsat`/`constructed` (`-19%`/`-14%` interleaved); LOO showed `-1.2%`/`-0.4%` total, `+1.4%`/`+1.3%` geomean, and ~20 extra LOC per file |
| `#2.4` `eqc_state.entry(...)` | `4100.5 ms` (`+1.6%`) | `1.015×` (`+1.5%`) | `3729.0 ms` (`+0.8%`) | `1.007×` (`+0.7%`) | **Dropped** — tiny gain isolated; LOO showed `-0.1%` total / `+0.8%` geomean interleaved and `-1.8%` total / `-0.5%` geomean two-phase |
| `#2.5` Inline `Parents` enum | `4170.6 ms` (`-0.1%`) | `1.016×` (`+1.6%`) | `3644.3 ms` (`+3.1%`) | `1.083×` (`+7.6%`) | **Dropped (simplicity trade-off)** — LOO confirmed `+5.9%`/`+2.8%` geomean (`constructed +15–18%`, `impress +10–12%`), with `+0.4%`/`-2.1%` total; dropped to cut ~30 LOC/file (easy to re-add if small-graph geomean matters) |
| `#3.1` Dense `u32` `IndexMap` node indices | `1931.2 ms` (`+53.6%`) | `1.347×` (`+25.8%`) | `1742.7 ms` (`+53.6%`) | `1.338×` (`+25.2%`) | **Kept** — single biggest win (`set_covering +63.8%/+65.6%`, `herbie +37.6%/+37.3%`, `eggcc-bril +34.5%/+32.3%`) |
| `#3.1b` Dense `Vec` `node_state` (with `#3.1`) | `1743.2 ms` (`+58.2%`) | `1.477×` (`+32.3%`) | `1508.5 ms` (`+59.9%`) | `1.473×` (`+32.1%`) | **Dropped** — on `kept_optimized` adds `+11.0%`/`+10.0%` total (`set_covering +15–16%`), but regresses `flexc -28.3%` (interleaved) and `egg -45.2%` (two-phase); `0.989×`/`1.003×` geomean |
| `#4.1` `BinaryHeap::from` for `leaves` | — | — | `3670.5 ms` (`+2.4%`) | `1.045×` (`+4.3%`) | **Dropped** — modest consistent win (`constructed +5.9%`, `flexc +5.6%`, `babble +5.2%`); two-phase only, dropped for symmetry |
| `#4.2` Batch uniform-cost e-classes (Phase 1) | — | — | `3663.5 ms` (`+2.5%`) | `1.008×` (`+0.8%`) | **Dropped** — small win isolated (`egg +17.7%`), but `-11.9%` total / `0.938×` geomean on `kept_optimized` (re-hashes class nodes via `get_full` on pop) |
| `#4.3` 4-ary min-heap | `3958.8 ms` (`+5.0%`) | `0.973×` (`-2.8%`) | `3564.6 ms` (`+5.2%`) | `1.024×` (`+2.3%`) | **Dropped** — helps large heaps (`set_covering +6.8%/+5.3%`, `egg +22.7%/+9.8%`), regresses interleaved geomean |
| **`combined`** (`#2.1`–`#2.5` + `#3.1`) | **`1538.9 ms` (`+63.1%`)** | **`1.863×` (`+46.3%`)** | **`1433.4 ms` (`+61.9%`)** | **`1.713×` (`+41.6%`)** | Pre-trim combined run (`2.71×` / `2.62×` total); `#2.3`, `#2.4` & `#2.5` subsequently dropped from `a_star_*_optimized.rs` (leaving `#2.1` + `#2.2` + `#3.1` + `#4.4`) |

### Follow-Up A/B: `#4.4` (`BTreeMap<Cost, Vec<Action>>` vs. `BinaryHeap` on `a-star-interleaved-optimized`)

Measured interleaved per file across all **773 e-graphs / 9,569 single-root extractions** (`--single-root`, 1 warmup + median of 3 runs, non-`benchmode` run, 0 cost mismatches) on top of `#2.1` + `#2.2` + `#3.1`:

| Slice | `N` | `BinaryHeap` | `BTreeMap` | Runtime Reduction | Geomean Speedup (`Heap / BTree`) | Wins (`BTree` vs. `Heap`) |
|---|---:|---:|---:|---:|---:|---:|
| All | 9,569 | `1478.0 ms` | **`1205.3 ms`** | **`+18.5%`** | `1.002×` | — |
| `max >= 100 µs` | 368 | `1317.4 ms` | **`1044.9 ms`** | **`+20.7%`** | **`1.124×`** | **341** vs. 26 |
| `max >= 1,000 µs` | 93 | `1237.9 ms` | **`973.8 ms`** | **`+21.3%`** | **`1.241×`** | **85** vs. 8 |

Largest suite gains: `set_covering` (`+22.7%`, `848.8 → 655.9 ms`), `herbie` (`+19.7%`, `321.4 → 258.0 ms`), `maxsat` (`+29.8%`), `egg` (`+21.0%`), `eggcc-bril` (`+15.3%`), `tensat` (`+11.9%`), `flexc` (`+11.7%`). Main regressions on high-distinct-cost suites: `constructed` (`-18.1%`) and `rover` (`-1.0%` total; `-12%` to `-16%` on `fir_8_tap_{5,6,7}iteration`).
