// Optimized `PrioQueueExtractor` (see `a_star_optimizations.md`):
// - #2.1: Incremental `PendingNode { delay, cost }` (no per-node `FxHashSet` child deduplication
//         or child re-summing when `delay` reaches 0).
// - #2.2: Use `bottom_up_cost` directly from the priority queue payload (leaves skip `node_state`,
//         and pops do not re-sum child costs).
// - #3.1: Dense `u32` `IndexMap` node indices (`get_full` / `get_index`) with sparse
//         `FxHashMap<u32, PendingNode>` and `FxHashMap<&ClassId, EqcStatus>` (avoids repeated
//         `egraph.nodes` SipHash lookups and `ClassId`/`NodeId` clones while preserving the
//         baseline initialization order and algorithm).
// - #4.4: Bucketed `BTreeMap<Cost, Vec<u32>>` priority queue (replaces `BinaryHeap` + `PrioQueue`).

use std::collections::BTreeMap;
use std::mem;

use rustc_hash::FxHashMap;

use super::*;

// State of a branch e-node waiting for its child e-classes to be resolved.
// `delay` counts remaining unresolved child occurrences (not deduplicated), and `cost`
// accumulates `node.cost` plus resolved child costs.
struct PendingNode {
    delay: u32,
    cost: Cost,
}

// Whether an e-class has been resolved yet. As an e-class' parents are only needed at the moment
// it is assigned a minimal e-node, the two states are mutually exclusive.
enum EqcStatus {
    Parents(Vec<u32>),
    Resolved,
}

fn pop_min(queue: &mut BTreeMap<Cost, Vec<u32>>) -> Option<(u32, Cost)> {
    let mut entry = queue.first_entry()?;
    let cost = *entry.key();
    let node = entry.get_mut().pop().unwrap();
    if entry.get().is_empty() {
        entry.remove();
    }
    Some((node, cost))
}

pub struct PrioQueueOptimizedExtractor;

impl Extractor for PrioQueueOptimizedExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        let mut eqc_state = FxHashMap::<&ClassId, EqcStatus>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );
        for class in egraph.classes().values() {
            eqc_state.insert(&class.id, EqcStatus::Parents(Vec::new()));
        }

        let mut node_state: FxHashMap<u32, PendingNode> = Default::default();
        let mut analysis_pending: BTreeMap<Cost, Vec<u32>> = BTreeMap::new();

        for class in egraph.classes().values() {
            for node_id in &class.nodes {
                let (idx, _, node) = egraph.nodes.get_full(node_id).unwrap();
                let node_idx = idx as u32;
                if node.children.is_empty() {
                    analysis_pending
                        .entry(node.cost)
                        .or_default()
                        .push(node_idx);
                } else {
                    for child in &node.children {
                        let c = egraph.nid_to_cid(child);
                        match eqc_state.get_mut(c).unwrap() {
                            EqcStatus::Parents(parents) => parents.push(node_idx),
                            EqcStatus::Resolved => unreachable!(),
                        }
                    }
                    node_state.insert(
                        node_idx,
                        PendingNode {
                            delay: node.children.len() as u32,
                            cost: node.cost,
                        },
                    );
                }
            }
        }

        let mut result = ExtractionResult::default();
        while let Some((node_idx, resolved_cost)) = pop_min(&mut analysis_pending) {
            let (nid, node) = egraph.nodes.get_index(node_idx as usize).unwrap();
            let class_id = &node.eclass;
            let status = eqc_state.get_mut(class_id).unwrap();
            if matches!(status, EqcStatus::Resolved) {
                continue;
            }
            let prev = mem::replace(status, EqcStatus::Resolved);
            result.choose(class_id.clone(), nid.clone());
            if let EqcStatus::Parents(parents) = prev {
                for parent in parents {
                    let pending = node_state.get_mut(&parent).unwrap();
                    pending.cost += resolved_cost;
                    pending.delay -= 1;
                    if pending.delay == 0 {
                        let (_, parent_node) = egraph.nodes.get_index(parent as usize).unwrap();
                        if matches!(eqc_state[&parent_node.eclass], EqcStatus::Resolved) {
                            continue;
                        }
                        analysis_pending
                            .entry(pending.cost)
                            .or_default()
                            .push(parent);
                    }
                }
            }
        }

        result
    }
}
