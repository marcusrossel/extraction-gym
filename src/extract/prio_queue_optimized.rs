// Optimized `PrioQueueExtractor` (see `a_star_optimizations.md`):
// - #2.1: Delay-counter `node_delay: FxHashMap<u32, u32>` (no per-node `FxHashSet` child
//         deduplication; evaluates `node_cost` once when `delay` reaches 0 so non-additive cost
//         functions are supported directly).
// - #2.2: Use `bottom_up_cost` directly from the priority queue payload (leaves skip `node_delay`,
//         and pops do not re-evaluate child costs).
// - #3.1: Dense `u32` `IndexMap` node indices (`get_full` / `get_index`) with sparse
//         `FxHashMap<u32, u32>` and `FxHashMap<&ClassId, EqcStatus>` (avoids repeated
//         `egraph.nodes` SipHash lookups and `ClassId`/`NodeId` clones while preserving the
//         baseline initialization order and algorithm).
// - #4.4: Bucketed `BTreeMap<Cost, Vec<u32>>` priority queue (replaces `BinaryHeap` + `PrioQueue`).

use std::mem;

use rustc_hash::FxHashMap;

use super::common::{BucketQueue, EqcStatus, node_cost};
use super::*;

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

        let mut node_delay: FxHashMap<u32, u32> = Default::default();
        let mut analysis_pending: BucketQueue<u32> = BucketQueue::new();

        for class in egraph.classes().values() {
            for node_id in &class.nodes {
                let (idx, _, node) = egraph.nodes.get_full(node_id).unwrap();
                let node_idx = idx as u32;
                if node.children.is_empty() {
                    analysis_pending.push(node_cost(egraph, node, |_| unreachable!()), node_idx);
                } else {
                    for child in &node.children {
                        let c = egraph.nid_to_cid(child);
                        match eqc_state.get_mut(c).unwrap() {
                            EqcStatus::Parents(parents) => parents.push(node_idx),
                            EqcStatus::Resolved(_) => unreachable!(),
                        }
                    }
                    node_delay.insert(node_idx, node.children.len() as u32);
                }
            }
        }

        let mut result = ExtractionResult::default();
        while let Some((node_idx, resolved_cost)) = analysis_pending.pop_with_cost() {
            let (nid, node) = egraph.nodes.get_index(node_idx as usize).unwrap();
            let class_id = &node.eclass;
            let status = eqc_state.get_mut(class_id).unwrap();
            if matches!(status, EqcStatus::Resolved(_)) {
                continue;
            }
            let prev = mem::replace(status, EqcStatus::Resolved(resolved_cost));
            result.choose(class_id.clone(), nid.clone());
            if let EqcStatus::Parents(parents) = prev {
                for parent in parents {
                    let delay = node_delay.get_mut(&parent).unwrap();
                    *delay -= 1;
                    if *delay == 0 {
                        let (_, parent_node) = egraph.nodes.get_index(parent as usize).unwrap();
                        if matches!(eqc_state[&parent_node.eclass], EqcStatus::Resolved(_)) {
                            continue;
                        }
                        let cost =
                            node_cost(egraph, parent_node, |cid| eqc_state[cid].resolved_cost());
                        analysis_pending.push(cost, parent);
                    }
                }
            }
        }

        result
    }
}
