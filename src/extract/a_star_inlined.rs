use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

#[derive(PartialEq, Eq)]
enum Action<'a> {
    Visit(&'a NodeId),
    Assign(&'a ClassId, &'a NodeId)
}

// TODO: Would it make sense to collapse some of the maps below when they are mapping from the same
//       type? E.g. `node_delay`, `parent_cost`, `node_cost`.
struct AStarExt<'a> {
    egraph:        &'a EGraph,
    queue:         PrioQueue<Action<'a>, Cost>,
    enqueued_eqcs: FxHashSet<&'a ClassId>,
    node_delay:    FxHashMap<&'a NodeId, usize>,
    eqc_parents:   FxHashMap<&'a ClassId, Vec<&'a NodeId>>,
    parent_cost:   FxHashMap<&'a NodeId, Cost>,
    node_cost:     FxHashMap<&'a NodeId, Cost>,
    eqc_min_cost:  FxHashMap<&'a ClassId, Cost>,
    eqc_min:       IndexMap<ClassId, NodeId>
}

impl<'a> AStarExt<'a> {
    fn new(egraph: &'a EGraph) -> AStarExt<'a> {
        AStarExt {
            egraph,
            queue:         PrioQueue::new(),
            enqueued_eqcs: Default::default(),
            node_delay:    Default::default(),
            eqc_parents:   Default::default(),
            parent_cost:   Default::default(),
            node_cost:     Default::default(),
            eqc_min_cost:  Default::default(),
            eqc_min:       Default::default()
        }
    }

    fn enqueue_assignment(&mut self, node: &'a NodeId, child_costs: Vec<Cost>) {
        // Like `ExtractionResult::node_sum_cost`.
        let bottom_up_cost = self.egraph[node].cost + child_costs.iter().sum::<NotNan<f64>>();
        // Inlined: set_node_cost
        self.node_cost.insert(node, bottom_up_cost);
        let top_down_cost = self.parent_cost[node];
        let merit = top_down_cost + bottom_up_cost;
        let eqc = self.egraph.nid_to_cid(node);
        let action = Action::Assign(eqc, node);
        // Inlined: enqueue
        self.queue.insert(action, merit);
    }

    fn enqueue_visit_eqc(&mut self, eqc: &'a ClassId, parent_cost: Cost) {
        // Inlined: is_enqueued_eqc
        if !self.enqueued_eqcs.contains(eqc) {
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                // Inlined: enqueue_visit_node
                // Inlined: set_parent_cost
                self.parent_cost.insert(node, parent_cost);
                let top_down_cost = parent_cost + self.egraph[node].cost;
                let action = Action::Visit(node);
                // Inlined: enqueue
                self.queue.insert(action, top_down_cost);
            }
            // Inlined: add_enqueued_eqc
            self.enqueued_eqcs.insert(eqc);
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_visit_eqc(target, zero);
        // Inlined: dequeue
        while let Some((action, _)) = self.queue.pop() {
            // Inlined: run_action
            match action {
                Action::Visit(node) => {
                    // Inlined: visit_node
                    if self.egraph[node].children.is_empty() {
                        self.enqueue_assignment(node, vec![]);
                    } else {
                        // Inlined: visit_branch_node
                        let mut child_costs = Vec::new();
                        let mut delayed_eqcs: FxHashSet<&'a ClassId> = FxHashSet::default();
                        let td_cost = self.parent_cost[node] + self.egraph[node].cost;
                        for child in &self.egraph[node].children {
                            let eqc = self.egraph.nid_to_cid(child);
                            // (1) If the child `eqc` is already resolved, remember its cost.
                            // (2) If `eqc` is not resolved, set the parent-child relationship, and enqueue `eqc`
                            //     (the node delay is set after the loop).
                            if let Some(&cost) = self.eqc_min_cost.get(eqc) {
                                child_costs.push(cost);
                            } else if !delayed_eqcs.contains(eqc) {
                                // It is important that we do not register the same e-class as delayed multiple
                                // times, as this would break the delay count.
                                delayed_eqcs.insert(eqc);
                                // Inlined: add_eqc_parent
                                self.eqc_parents.entry(eqc).or_insert_with(Vec::new).push(node);
                                self.enqueue_visit_eqc(eqc, td_cost);
                            }
                        }
                        if delayed_eqcs.is_empty() {
                            // If all of `node`'s children are resolved, we can add an assignment-action for it
                            // immediately.
                            self.enqueue_assignment(node, child_costs);
                        } else {
                            // Inlined: set_node_delay
                            self.node_delay.insert(node, delayed_eqcs.len());
                        }
                    }
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
                                        // Inlined: node_child_costs
                                        // TODO: If we want to optimize, we can immediately compute the sum here as we know that the
                                        //       cost function also just adds the child costs together.
                                        let child_costs = {
                                            let node = &self.egraph[parent];
                                            node.children.iter().map(|child| {
                                                let eqc = self.egraph.nid_to_cid(child);
                                                self.eqc_min_cost.get(eqc).copied().unwrap()
                                            }).collect()
                                        };
                                        self.enqueue_assignment(parent, child_costs);
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

pub struct AStarExtractor;

impl Extractor for AStarExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // TODO: We currently assume there to be only a single root class from which we extract. Add
        //       a field to the `ExtractorDetail` in `main.rs` where this can be declared.
        assert_eq!(roots.len(), 1);
        let target = &roots[0];
        let mut ext = AStarExt::new(egraph);
        ext.run(target);
        ExtractionResult { choices: ext.eqc_min }
    }
}

mod a_star {
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
use a_star::*;
