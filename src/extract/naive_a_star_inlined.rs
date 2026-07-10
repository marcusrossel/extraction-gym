use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

#[derive(PartialEq, Eq)]
enum Action<'a> {
    Visit(&'a NodeId),
    Assign(&'a ClassId, &'a NodeId)
}

struct NaiveAStarTopDownExtractor<'a> {
    egraph:        &'a EGraph,
    eqc_parents:   FxHashMap<&'a ClassId, Vec<&'a NodeId>>,
    node_delay:    FxHashMap<&'a NodeId, usize>,
    parent_cost:   FxHashMap<&'a NodeId, Cost>,
    node_cost:     FxHashMap<&'a NodeId, Cost>,
    enqueued_eqcs: FxHashSet<&'a ClassId>,
    queue:         PrioQueue<&'a NodeId, Cost>,
    leaves:        PrioQueue<Action<'a>, Cost>
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

    fn enqueue_eqc(&mut self, eqc: &'a ClassId, parent_cost: Cost) {
        // Inlined: is_enqueued_eqc
        if !self.enqueued_eqcs.contains(eqc) {
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                // Inlined: enqueue_node
                // Inlined: set_parent_cost
                self.parent_cost.insert(node, parent_cost);
                let td_cost = parent_cost + self.egraph[node].cost;
                self.queue.insert(node, td_cost);
            }
            // Inlined: add_enqueued_eqc
            self.enqueued_eqcs.insert(eqc);
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_eqc(target, zero);
        // Inlined: dequeue
        while let Some((node, _)) = self.queue.pop() {
            // Inlined: visit_node
            if self.egraph[node].children.is_empty() {
                // Inlined: add_leaf
                let top_down_cost = self.parent_cost[node];
                let bottom_up_cost = self.egraph[node].cost;
                // Inlined: set_node_cost
                self.node_cost.insert(node, bottom_up_cost);
                let merit = top_down_cost + bottom_up_cost;
                self.leaves.insert(Action::Visit(node), merit);
            } else {
                // Inlined: visit_branch_node
                let egraph = self.egraph;
                let td_cost = self.parent_cost[node] + self.egraph[node].cost;
                let mut unique_child_eqcs: FxHashSet<&'a ClassId> = FxHashSet::default();

                for child in &egraph[node].children {
                    let eqc = egraph.nid_to_cid(child);
                    if unique_child_eqcs.insert(eqc) {
                        // Inlined: add_eqc_parent
                        self.eqc_parents.entry(eqc).or_insert_with(Vec::new).push(node);
                        self.enqueue_eqc(eqc, td_cost);
                    }
                }

                // Inlined: set_node_delay
                self.node_delay.insert(node, unique_child_eqcs.len());
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
    queue:        PrioQueue<Action<'a>, Cost>,
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

    fn run(&mut self, target: &'a ClassId) {
        // Inlined: dequeue
        while let Some((action, merit)) = self.queue.pop() {
            // Inlined: run_action
            match action {
                Action::Visit(node) => {
                    // Inlined: visit_node
                    let eqc = self.egraph.nid_to_cid(node);
                    // Inlined: enqueue
                    self.queue.insert(Action::Assign(eqc, node), merit);
                },
                Action::Assign(eqc, node) => {
                    // Inlined: assign_eqc
                    // Inlined: eqc_has_min
                    if !self.eqc_min.contains_key(eqc) {
                        // Inlined: set_eqc_min
                        let cost = self.node_cost[node];
                        self.eqc_min_cost.entry(eqc).or_insert(cost);
                        self.eqc_min.entry(eqc.clone()).or_insert_with(|| node.clone());

                        // Inlined: update_eqc_parents
                        if let Some(parents) = self.eqc_parents.get(eqc) {
                            let parents = parents.clone();
                            for parent in parents {
                                match self.node_delay.get(parent).copied() {
                                    Some(1) => {
                                        // Inlined: enqueue_branch_node_visit
                                        let top_down_cost = self.parent_cost.get(parent).copied().unwrap();
                                        // Inlined: get_min_node_cost
                                        // Like `ExtractionResult::node_sum_cost`.
                                        let bottom_up_cost = {
                                            let node = &self.egraph[parent];
                                            let total_child_cost: Cost = node.children.iter().map(|child| {
                                                let eqc = self.egraph.nid_to_cid(child);
                                                self.eqc_min_cost.get(eqc).copied().unwrap()
                                            }).sum();
                                            node.cost + total_child_cost
                                        };
                                        // Inlined: set_node_cost
                                        self.node_cost.insert(parent, bottom_up_cost);
                                        let merit = top_down_cost + bottom_up_cost;
                                        // Inlined: enqueue
                                        self.queue.insert(Action::Visit(parent), merit);
                                    },
                                    Some(n) if n > 1 => {
                                        // Inlined: set_node_delay
                                        self.node_delay.insert(parent, n - 1);
                                    },
                                    _ => panic!("Reached bad path in `update_eqc_parents`."),
                                }
                            }
                        }
                    }
                },
            }
            // Inlined: eqc_has_min
            if self.eqc_min.contains_key(target) { break }
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
        bottom_up.run(target);
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
