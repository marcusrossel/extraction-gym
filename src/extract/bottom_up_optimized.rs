// Optimized `BottomUpExtractor` (see `a_star_optimizations.md`):
// - #3.1: Pre-resolve `NodeId`s to dense `u32` `IndexMap` indices (`get_index_of` / `get_index`)
//         once before the fixed-point loop in exact `egraph.classes().values()` -> `class.nodes`
//         order, plus borrowed `&ClassId` keys in `costs: FxHashMap<&ClassId, Cost>` (avoids
//         per-iteration `egraph[node]` SipHash lookups and `ClassId` clones while preserving the
//         exact baseline sweep order and algorithm).

use rustc_hash::FxHashMap;

use super::*;

pub struct BottomUpOptimizedExtractor;

impl Extractor for BottomUpOptimizedExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        let node_indices: Vec<u32> = egraph
            .classes()
            .values()
            .flat_map(|class| {
                class
                    .nodes
                    .iter()
                    .map(|node| egraph.nodes.get_index_of(node).unwrap() as u32)
            })
            .collect();

        let mut result = ExtractionResult::default();
        let mut costs = FxHashMap::<&ClassId, Cost>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );
        let mut did_something = false;

        loop {
            for &node_idx in &node_indices {
                let (node_id, node) = egraph.nodes.get_index(node_idx as usize).unwrap();
                let class_id = &node.eclass;
                let cost = node.cost
                    + node
                        .children
                        .iter()
                        .map(|n| costs.get(egraph.nid_to_cid(n)).unwrap_or(&INFINITY))
                        .sum::<Cost>();
                if &cost < costs.get(class_id).unwrap_or(&INFINITY) {
                    result.choose(class_id.clone(), node_id.clone());
                    costs.insert(class_id, cost);
                    did_something = true;
                }
            }

            if did_something {
                did_something = false;
            } else {
                break;
            }
        }

        result
    }
}
