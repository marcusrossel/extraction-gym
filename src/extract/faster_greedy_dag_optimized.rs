// Optimized `FasterGreedyDagExtractor` (see `a_star_optimizations.md`):
// - #3.1: Dense `u32` `IndexMap` node indices (`get_full` / `get_index`) in `parents`,
//         `UniqueQueue<u32>`, and `CostSet.choice`, plus borrowed `&ClassId` keys in `parents`,
//         `costs`, and `CostSet.costs` (avoids repeated `egraph.nodes` SipHash lookups and
//         `ClassId`/`NodeId` clones while preserving the exact baseline initialization/queue order
//         and `calculate_cost_set` logic).
//
// Calculates the cost where shared nodes are just costed once,
// For example (+ (* x x ) (* x x )) has one multiplication
// included in the cost.

use rustc_hash::FxHashMap;

use super::common::UniqueQueue;
use super::*;

struct CostSet<'a> {
    // It's slightly faster if this is a HashMap rather than an FxHashMap.
    costs: HashMap<&'a ClassId, Cost>,
    total: Cost,
    choice: u32,
}

pub struct FasterGreedyDagOptimizedExtractor;

impl FasterGreedyDagOptimizedExtractor {
    fn calculate_cost_set<'a>(
        egraph: &'a EGraph,
        node_idx: u32,
        node: &'a Node,
        costs: &FxHashMap<&'a ClassId, CostSet<'a>>,
        best_cost: Cost,
    ) -> CostSet<'a> {
        let cid = &node.eclass;

        if node.children.is_empty() {
            return CostSet {
                costs: HashMap::from([(cid, node.cost)]),
                total: node.cost,
                choice: node_idx,
            };
        }

        // Get unique classes of children.
        let mut childrens_classes = node
            .children
            .iter()
            .map(|c| egraph.nid_to_cid(c))
            .collect::<Vec<&ClassId>>();
        childrens_classes.sort();
        childrens_classes.dedup();

        let first_cost = costs.get(childrens_classes[0]).unwrap();

        if childrens_classes.contains(&cid)
            || (childrens_classes.len() == 1 && (node.cost + first_cost.total > best_cost))
        {
            // Shortcut. Can't be cheaper so return junk.
            return CostSet {
                costs: Default::default(),
                total: INFINITY,
                choice: node_idx,
            };
        }

        // Clone the biggest set and insert the others into it.
        let id_of_biggest = *childrens_classes
            .iter()
            .max_by_key(|&&s| costs.get(s).unwrap().costs.len())
            .unwrap();
        let mut result = costs.get(id_of_biggest).unwrap().costs.clone();
        for &child_cid in &childrens_classes {
            if child_cid == id_of_biggest {
                continue;
            }

            let next_cost = &costs.get(child_cid).unwrap().costs;
            for (&key, &value) in next_cost.iter() {
                result.insert(key, value);
            }
        }

        let contains = result.contains_key(cid);
        result.insert(cid, node.cost);

        let result_cost = if contains {
            INFINITY
        } else {
            result.values().sum()
        };

        CostSet {
            costs: result,
            total: result_cost,
            choice: node_idx,
        }
    }
}

impl Extractor for FasterGreedyDagOptimizedExtractor {
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
        let mut costs = FxHashMap::<&ClassId, CostSet<'_>>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );

        while let Some(node_idx) = analysis_pending.pop() {
            let (_, node) = egraph.nodes.get_index(node_idx as usize).unwrap();
            let class_id = &node.eclass;
            if node.children.iter().all(|c| costs.contains_key(n2c(c))) {
                let prev_cost = costs.get(class_id).map_or(INFINITY, |cs| cs.total);

                let cost_set = Self::calculate_cost_set(egraph, node_idx, node, &costs, prev_cost);
                if cost_set.total < prev_cost {
                    costs.insert(class_id, cost_set);
                    analysis_pending.extend(parents[class_id].iter().copied());
                }
            }
        }

        for (cid, cost_set) in costs {
            let (nid, _) = egraph.nodes.get_index(cost_set.choice as usize).unwrap();
            result.choose(cid.clone(), nid.clone());
        }

        result
    }
}
