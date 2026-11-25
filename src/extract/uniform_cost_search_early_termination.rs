// `UniformCostSearchExtractor` with only the early-termination optimization from
// `UniformCostSearchOptimizedExtractor`: stop Phase 2 as soon as all root e-classes are resolved.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use super::common::PrioQueue;
use super::*;

pub struct UniformCostSearchEarlyTerminationExtractor;

impl Extractor for UniformCostSearchEarlyTerminationExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let n2c = |nid: &NodeId| egraph.nid_to_cid(nid);

        let mut parents: IndexMap<ClassId, Vec<NodeId>> = IndexMap::default();
        let mut child_counter: IndexMap<NodeId, usize> = IndexMap::new();
        let mut analysis_pending: PrioQueue<NodeId, Cost> = PrioQueue::new();
        let mut result = ExtractionResult::default();
        let mut costs = FxHashMap::<ClassId, Cost>::default();

        let mut visited: FxHashSet<ClassId> = FxHashSet::default();
        let mut queue: VecDeque<ClassId> = VecDeque::new();
        let mut root_set: FxHashSet<&ClassId> = Default::default();
        let single_root = if roots.len() == 1 {
            Some(&roots[0])
        } else {
            root_set.extend(roots.iter());
            None
        };
        for root in roots {
            parents.entry(root.clone()).or_default();
            if visited.insert(root.clone()) {
                queue.push_back(root.clone());
            }
        }

        while let Some(cid) = queue.pop_front() {
            for node_id in &egraph.classes()[&cid].nodes {
                let child_classes: FxHashSet<&ClassId> =
                    egraph[node_id].children.iter().map(n2c).collect();

                child_counter.insert(node_id.clone(), child_classes.len());

                for c in child_classes {
                    parents.entry(c.clone()).or_default().push(node_id.clone());
                    if visited.insert(c.clone()) {
                        queue.push_back(c.clone());
                    }
                }

                if egraph[node_id].is_leaf() {
                    let cost = result.node_sum_cost(egraph, &egraph[node_id], &costs);
                    analysis_pending.insert(node_id.clone(), cost);
                }
            }
        }

        while let Some((node_id, _cost)) = analysis_pending.pop() {
            let class_id = n2c(&node_id);
            if costs.contains_key(class_id) {
                continue;
            }

            let node = &egraph[&node_id];
            let cost = result.node_sum_cost(egraph, node, &costs);
            result.choose(class_id.clone(), node_id.clone());
            match single_root {
                Some(target) => {
                    if class_id == target {
                        break;
                    }
                }
                None => {
                    if root_set.remove(class_id) && root_set.is_empty() {
                        break;
                    }
                }
            }
            costs.insert(class_id.clone(), cost);
            for p in parents[class_id].iter() {
                if costs.contains_key(n2c(p)) {
                    continue;
                }

                let ctr = child_counter.get_mut(p).unwrap();
                *ctr -= 1;
                if *ctr == 0 {
                    let cost = result.node_sum_cost(egraph, &egraph[p], &costs);
                    analysis_pending.insert(p.clone(), cost);
                }
            }
        }

        result
    }
}
