// Optimized `FasterBottomUpExtractor` (see `a_star_optimizations.md`):
// - #3.1: Dense `u32` `IndexMap` node indices (`get_full` / `get_index`) in `parents` and
//         `UniqueQueue<u32>`, plus borrowed `&ClassId` keys in `FxHashMap` for `parents` and
//         `costs` (avoids repeated `egraph.nodes` SipHash lookups and `ClassId`/`NodeId` clones
//         while preserving the exact baseline initialization/queue order and algorithm).

use rustc_hash::FxHashMap;

use super::common::UniqueQueue;
use super::*;

pub struct FasterBottomUpOptimizedExtractor;

impl Extractor for FasterBottomUpOptimizedExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        let mut parents = FxHashMap::<&ClassId, Vec<u32>>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );
        let n2c = |nid: &NodeId| egraph.nid_to_cid(nid);
        let mut analysis_pending = UniqueQueue::<u32>::default();

        for class in egraph.classes().values() {
            parents.insert(&class.id, Vec::new());
        }

        for class in egraph.classes().values() {
            for node_id in &class.nodes {
                let (idx, _, node) = egraph.nodes.get_full(node_id).unwrap();
                let node_idx = idx as u32;
                for c in &node.children {
                    // compute parents of this enode
                    parents.get_mut(n2c(c)).unwrap().push(node_idx);
                }

                // start the analysis from leaves
                if node.is_leaf() {
                    analysis_pending.insert(node_idx);
                }
            }
        }

        let mut result = ExtractionResult::default();
        let mut costs = FxHashMap::<&ClassId, Cost>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );

        while let Some(node_idx) = analysis_pending.pop() {
            let (node_id, node) = egraph.nodes.get_index(node_idx as usize).unwrap();
            let class_id = &node.eclass;
            let prev_cost = costs.get(class_id).unwrap_or(&INFINITY);
            let cost = node.cost
                + node
                    .children
                    .iter()
                    .map(|n| costs.get(n2c(n)).unwrap_or(&INFINITY))
                    .sum::<Cost>();
            if cost < *prev_cost {
                result.choose(class_id.clone(), node_id.clone());
                costs.insert(class_id, cost);
                analysis_pending.extend(parents[class_id].iter().copied());
            }
        }

        result
    }
}
