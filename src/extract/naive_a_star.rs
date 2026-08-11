use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

struct NaiveAStarTopDownExtractor<'a> {
    egraph:        &'a EGraph,
    eqc_parents:   FxHashMap<&'a ClassId, Vec<&'a NodeId>>,
    node_delay:    FxHashMap<&'a NodeId, usize>,
    parent_cost:   FxHashMap<&'a NodeId, Cost>,
    node_cost:     FxHashMap<&'a NodeId, Cost>,
    enqueued_eqcs: FxHashSet<&'a ClassId>,
    queue:         PrioQueue<&'a NodeId, Cost>,
    leaves:        PrioQueue<&'a NodeId, Cost>
}

impl<'a> NaiveAStarTopDownExtractor<'a> {
    fn new(egraph: &'a EGraph) -> NaiveAStarTopDownExtractor<'a> {
        NaiveAStarTopDownExtractor {
            egraph,
            eqc_parents:   Default::default(),
            node_delay:    Default::default(),
            parent_cost:   Default::default(),
            node_cost:     Default::default(),
            enqueued_eqcs: Default::default(),
            queue:         PrioQueue::new(),
            leaves:        PrioQueue::new()
        }
    }

    fn set_node_delay(&mut self, node: &'a NodeId, delay: usize) {
        self.node_delay.insert(node, delay);
    }

    fn add_enqueued_eqc(&mut self, eqc: &'a ClassId) {
        self.enqueued_eqcs.insert(eqc);
    }

    fn is_enqueued_eqc(&self, eqc: &ClassId) -> bool {
        self.enqueued_eqcs.contains(eqc)
    }

    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: &'a NodeId) {
        self.eqc_parents.entry(eqc).or_insert_with(Vec::new).push(node);
    }

    fn dequeue(&mut self) -> Option<&'a NodeId> {
        self.queue.pop().map(|(node, _)| node)
    }

    fn enqueue_node(&mut self, node: &'a NodeId, parent_cost: Cost) {
        self.set_parent_cost(node, parent_cost);
        let td_cost = parent_cost + self.egraph[node].cost;
        self.queue.insert(node, td_cost);
    }

    fn enqueue_eqc(&mut self, eqc: &'a ClassId, parent_cost: Cost) {
        if !self.is_enqueued_eqc(eqc) {
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                self.enqueue_node(node, parent_cost);
            }
            self.add_enqueued_eqc(eqc);
        }
    }

    fn set_parent_cost(&mut self, node: &'a NodeId, parent_cost: Cost) {
        self.parent_cost.insert(node, parent_cost);
    }

    fn set_node_cost(&mut self, node: &'a NodeId, cost: Cost) {
        self.node_cost.insert(node, cost);
    }

    fn add_leaf(&mut self, node: &'a NodeId) {
        let top_down_cost = self.parent_cost[node];
        let bottom_up_cost = self.egraph[node].cost;
        self.set_node_cost(node, bottom_up_cost);
        let merit = top_down_cost + bottom_up_cost;
        self.leaves.insert(node, merit);
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_eqc(target, zero);
        while let Some(node) = self.dequeue() {
            if self.egraph[node].children.is_empty() {
            self.add_leaf(node);
            } else {
                let egraph = self.egraph;
                let td_cost = self.parent_cost[node] + self.egraph[node].cost;
                let mut unique_child_eqcs: FxHashSet<&'a ClassId> = FxHashSet::default();

                for child in &egraph[node].children {
                    let eqc = egraph.nid_to_cid(child);
                    if unique_child_eqcs.insert(eqc) {
                        self.add_eqc_parent(eqc, node);
                        self.enqueue_eqc(eqc, td_cost);
                    }
                }

                self.set_node_delay(node, unique_child_eqcs.len());
            }
        }
    }
}

struct NaiveAStarBottomUpExtractor<'a> {
    egraph:       &'a EGraph,
    eqc_parents:  FxHashMap<&'a ClassId, Vec<&'a NodeId>>,
    node_delay:   FxHashMap<&'a NodeId, usize>,
    parent_cost:  FxHashMap<&'a NodeId, Cost>,
    node_cost:    FxHashMap<&'a NodeId, Cost>,
    queue:        PrioQueue<&'a NodeId, Cost>,
    eqc_min_cost: FxHashMap<&'a ClassId, Cost>,
    eqc_min:      IndexMap<ClassId, NodeId>
}

impl<'a> NaiveAStarBottomUpExtractor<'a> {
    fn init(top_down: NaiveAStarTopDownExtractor<'a>) -> NaiveAStarBottomUpExtractor<'a> {
        NaiveAStarBottomUpExtractor {
            egraph:       top_down.egraph,
            eqc_parents:  top_down.eqc_parents,
            node_delay:   top_down.node_delay,
            parent_cost:  top_down.parent_cost,
            node_cost:    top_down.node_cost,
            queue:        top_down.leaves,
            eqc_min_cost: Default::default(),
            eqc_min:      IndexMap::new()
        }
    }

    fn eqc_has_min(&self, eqc: &ClassId) -> bool {
        self.eqc_min.contains_key(eqc)
    }

    fn set_eqc_min(&mut self, eqc: &'a ClassId, node: &'a NodeId) {
        let cost = self.node_cost[node];
        self.eqc_min_cost.entry(eqc).or_insert(cost);
        self.eqc_min.entry(eqc.clone()).or_insert_with(|| node.clone());
    }

    fn set_node_delay(&mut self, node: &'a NodeId, delay: usize) {
        self.node_delay.insert(node, delay);
    }

    fn erase_node_delay(&mut self, node: &'a NodeId) {
        self.node_delay.remove(node);
    }

    fn set_node_cost(&mut self, node: &'a NodeId, cost: Cost) {
        self.node_cost.insert(node, cost);
    }

    fn dequeue(&mut self) -> Option<&'a NodeId> {
        self.queue.pop().map(|(node, _)| node)
    }

    // Like `ExtractionResult::node_sum_cost`.
    fn get_min_node_cost(&self, node: &'a NodeId) -> Cost {
        let node = &self.egraph[node];
        let total_child_cost: Cost = node.children.iter().map(|child| {
            let eqc = self.egraph.nid_to_cid(child);
            self.eqc_min_cost.get(eqc).copied().unwrap()
        }).sum();
        node.cost + total_child_cost
    }

    fn enqueue(&mut self, node: &'a NodeId) {
        let top_down_cost = self.parent_cost.get(node).copied().unwrap();
        let bottom_up_cost = self.get_min_node_cost(node);
        self.set_node_cost(node, bottom_up_cost);
        let merit = top_down_cost + bottom_up_cost;
        self.queue.insert(node, merit);
    }

    fn run(&mut self) {
        while let Some(node) = self.dequeue() {
            let eqc = self.egraph.nid_to_cid(node);
            if self.eqc_has_min(eqc) { continue }
            self.set_eqc_min(eqc, node);
            let Some(parents) = self.eqc_parents.get(eqc) else { break };
            for parent in parents.iter().copied().collect::<Vec<&'a NodeId>>() {
                match self.node_delay.get(parent).copied() {
                    Some(1) => {
                        self.erase_node_delay(parent);
                        self.enqueue(parent)
                    },
                    Some(n) if n > 1 => {
                        self.set_node_delay(parent, n - 1)
                    },
                    _ => panic!("Reached bad path in `NaiveAStarBottomUpExtractor::main`."),
                }
            }
        }
    }
}

pub struct NaiveAStarExtractor;

impl Extractor for NaiveAStarExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // TODO: We currently assume there to be only a single root class from which we extract. Add
        //       a field to the `ExtractorDetail` in `main.rs` where this can be declared.
        assert_eq!(roots.len(), 1);
        let target = &roots[0];
        let mut top_down = NaiveAStarTopDownExtractor::new(egraph);
        top_down.run(target);
        let mut bottom_up = NaiveAStarBottomUpExtractor::init(top_down);
        bottom_up.run();
        ExtractionResult { choices: bottom_up.eqc_min }
    }
}

mod naive_a_star {
    use std::cmp::{Ord, Ordering};
    use std::collections::BinaryHeap;

    // Takes the `Ord` from U, but reverses it.
    #[derive(PartialEq, Eq, Debug)]
    struct WithOrdRev<T: Eq, U: Ord>(pub T, pub U);

    impl<T: Eq, U: Ord> Ord for WithOrdRev<T, U> {
        fn cmp(&self, other: &Self) -> Ordering {
            // It's the other way around, because we want a min-heap!
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
use naive_a_star::*;
