use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/* This file was (almost) entirely generated with AI based off of the `prio-queue` extractor. */

use super::*;

/*
This extractor follows the same approach as the `prio-queue` extractor, but only visits those
e-nodes/e-classes reachable from `roots`. Accordingly, it is generally faster than `prio-queue`, but
especially so when the sub-graph reachable from `roots` is significantly smaller than the entire
e-graph.
*/

pub struct ReachPrioQueueExtractor;

impl Extractor for ReachPrioQueueExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let n2c = |nid: &NodeId| egraph.nid_to_cid(nid);

        let mut parents: IndexMap<ClassId, Vec<NodeId>> = IndexMap::default();
        let mut child_counter: IndexMap<NodeId, usize> = IndexMap::new();
        let mut analysis_pending: PrioQueue<NodeId, Cost> = PrioQueue::new();
        let mut result = ExtractionResult::default();
        let mut costs = FxHashMap::<ClassId, Cost>::default();

        let mut visited: FxHashSet<ClassId> = FxHashSet::default();
        let mut queue: VecDeque<ClassId> = VecDeque::new();
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

mod prio {
    use std::cmp::{Ord, Ordering};
    use std::collections::BinaryHeap;

    #[derive(PartialEq, Eq, Debug)]
    struct WithOrdRev<T: Eq, U: Ord>(pub T, pub U);

    impl<T: Eq, U: Ord> Ord for WithOrdRev<T, U> {
        fn cmp(&self, other: &Self) -> Ordering {
            other.1.cmp(&self.1)
        }
    }

    impl<T: Eq, U: Ord> PartialOrd for WithOrdRev<T, U> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    pub struct PrioQueue<T: Eq, C: Ord>(BinaryHeap<WithOrdRev<T, C>>);

    impl<T: Eq, C: Ord> PrioQueue<T, C> {
        pub fn new() -> Self {
            PrioQueue(BinaryHeap::new())
        }

        pub fn pop(&mut self) -> Option<(T, C)> {
            self.0.pop().map(|WithOrdRev(t, c)| (t, c))
        }

        pub fn insert(&mut self, t: T, c: C) {
            self.0.push(WithOrdRev(t, c));
        }
    }
}
use prio::*;
