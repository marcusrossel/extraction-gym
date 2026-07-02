use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

#[derive(PartialEq, Eq)]
struct Merit {
    cost: Cost,
    path_cost: Cost
}

impl Merit {
    fn total(&self) -> Cost {
        self.cost + self.path_cost
    }
}

impl PartialOrd for Merit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.total().cmp(&other.total())
    }
}

#[derive(PartialEq, Eq)]
enum Action<'a> {
    Visit(&'a NodeId),
    Assign(&'a ClassId, &'a NodeId)
}

struct AStarExt<'a> {
    egraph:           &'a EGraph,
    queue:            PrioQueue<Action<'a>, Merit>,
    enqueued_eqcs:    FxHashSet<&'a ClassId>,
    node_delay:       FxHashMap<&'a NodeId, usize>,
    eqc_parents:      FxHashMap<&'a ClassId, Vec<(&'a NodeId, Cost)>>,
    eqc_min_cost:     FxHashMap<&'a ClassId, Cost>,
    eqc_min:          IndexMap<ClassId, NodeId>
}

impl<'a> AStarExt<'a> {
    fn new(egraph: &'a EGraph) -> AStarExt<'a> {
        let num_eqcs = egraph.classes().len();
        let num_nodes = egraph.nodes.len();
        AStarExt {
            egraph,
            queue:            PrioQueue::new(),
            enqueued_eqcs:    FxHashSet::with_capacity_and_hasher(num_eqcs, Default::default()),
            node_delay:       FxHashMap::with_capacity_and_hasher(num_nodes, Default::default()),
            eqc_parents:      FxHashMap::with_capacity_and_hasher(num_eqcs, Default::default()),
            eqc_min_cost:     Default::default(),
            eqc_min:          Default::default()
        }
    }

    fn eqc_has_min(&self, eqc: &ClassId) -> bool {
        self.eqc_min.contains_key(eqc)
    }

    fn set_eqc_min(&mut self, eqc: &'a ClassId, node: &'a NodeId, cost: Cost) {
        self.eqc_min_cost.entry(eqc).or_insert(cost);
        self.eqc_min.entry(eqc.clone()).or_insert_with(|| node.clone());
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

    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: &'a NodeId, path_cost: Cost) {
        self.eqc_parents.entry(eqc).or_insert_with(Vec::new).push((node, path_cost));
    }

    fn dequeue(&mut self) -> Option<(Action<'a>, Merit)> {
        self.queue.pop()
    }

    fn enqueue(&mut self, action: Action<'a>, merit: Merit) {
        self.queue.insert(action, merit);
    }

    fn enqueue_visit_node(&mut self, node: &'a NodeId, path_cost: Cost) {
        let merit = Merit { cost: self.egraph[node].cost, path_cost };
        let action = Action::Visit(node);
        self.enqueue(action, merit);
    }

    fn enqueue_assignment(&mut self, node: &'a NodeId, path_cost: Cost, child_costs: Vec<Cost>) {
        // Like `ExtractionResult::node_sum_cost`.
        let cost = self.egraph[node].cost + child_costs.iter().sum::<NotNan<f64>>();
        let merit = Merit { cost, path_cost }; 
        let eqc = self.egraph.nid_to_cid(node);
        let action = Action::Assign(eqc, node);
        self.enqueue(action, merit);
    }

    fn enqueue_visit_eqc(&mut self, eqc: &'a ClassId, eqc_path_cost: Cost) {
        if !self.is_enqueued_eqc(eqc) {
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                self.enqueue_visit_node(node, eqc_path_cost);
            }
            self.add_enqueued_eqc(eqc);
        }
    }

    fn visit_branch_node(&mut self, node: &'a NodeId, merit: Merit) {
        let mut child_costs = Vec::new();
        let mut delayed_eqcs: FxHashSet<&'a ClassId> = FxHashSet::default();
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
                self.add_eqc_parent(eqc, node, merit.path_cost);
                self.enqueue_visit_eqc(eqc, merit.path_cost);
            }
        }
        if delayed_eqcs.is_empty() {
            // If all of `node`'s children are resolved, we can add an assignment-action for it
            // immediately.
            self.enqueue_assignment(node, merit.path_cost, child_costs);
        } else {
            self.set_node_delay(node, delayed_eqcs.len());
        }
    }

    fn visit_node(&mut self, node: &'a NodeId, merit: Merit) {
        if self.egraph[node].children.is_empty() {
            self.enqueue_assignment(node, merit.path_cost, vec![]);
        } else {
            self.visit_branch_node(node, merit);
        }
    }

    // TODO: If we want to optimize, we can immediately compute the sum here as we know that the cost
    //       function also just adds the child costs together.
    fn node_child_costs(&self, node: &'a NodeId) -> Vec<Cost> {
        let node = &self.egraph[node];
        node.children.iter().map(|child| {
            let eqc = self.egraph.nid_to_cid(child);
            self.eqc_min_cost.get(eqc).copied().unwrap()
        }).collect()
    }

    fn update_eqc_parents(&mut self, eqc: &'a ClassId) {
        let Some(parents) = self.eqc_parents.get(eqc) else { return; };
        let parents = parents.clone();
        for (parent, path_cost) in parents {
            match self.node_delay.get(parent).copied() {
                Some(1) => {
                    let child_costs = self.node_child_costs(parent);
                    self.enqueue_assignment(parent, path_cost, child_costs);
                },
                Some(n) if n > 1 => {
                    self.set_node_delay(parent, n - 1)
                },
                _ => panic!("Reached bad path in `update_eqc_parents`."),
            }
        }
    }

    fn assign_eqc(&mut self, eqc: &'a ClassId, node: &'a NodeId, merit: Merit) {
        if !self.eqc_has_min(eqc) {
            self.set_eqc_min(eqc, node, merit.cost);
            self.update_eqc_parents(eqc);
        }
    }

    fn run_action(&mut self, action: Action<'a>, merit: Merit) {
        match action {
            Action::Visit(node)       => self.visit_node(node, merit),
            Action::Assign(eqc, node) => self.assign_eqc(eqc, node, merit),
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_visit_eqc(target, zero);
        while let Some((action, merit)) = self.dequeue() {
            self.run_action(action, merit);
            if self.eqc_has_min(target) { break }
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
