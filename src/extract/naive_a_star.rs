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

struct NaiveAStarTopDownExtractor<'a> {
    egraph:           &'a EGraph,
    eqc_parents:      FxHashMap<&'a ClassId, Vec<&'a NodeId>>,
    node_delay:       FxHashMap<&'a NodeId, usize>,
    branch_path_cost: FxHashMap<&'a NodeId, Cost>,
    enqueued_eqcs:    FxHashSet<&'a ClassId>,
    queue:            PrioQueue<&'a NodeId, Merit>, 
    leaves:           PrioQueue<Action<'a>, Merit>
}

impl<'a> NaiveAStarTopDownExtractor<'a> {
    fn new(egraph: &'a EGraph) -> NaiveAStarTopDownExtractor<'a> {
        let num_eqcs = egraph.classes().len();
        let num_nodes = egraph.nodes.len();
        NaiveAStarTopDownExtractor {
            egraph,
            eqc_parents: FxHashMap::with_capacity_and_hasher(num_eqcs, Default::default()),
            node_delay: FxHashMap::with_capacity_and_hasher(num_nodes, Default::default()),
            branch_path_cost: FxHashMap::with_capacity_and_hasher(num_nodes, Default::default()),
            enqueued_eqcs: FxHashSet::with_capacity_and_hasher(num_eqcs, Default::default()),
            queue: PrioQueue::new(),
            leaves: PrioQueue::new()
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

    fn dequeue(&mut self) -> Option<(&'a NodeId, Merit)> {
        self.queue.pop()
    }

    fn enqueue_node(&mut self, node: &'a NodeId, path_cost: Cost) {
        let cost = todo!(); // TODO: Top-down (lower bound) cost.
        let merit = Merit { cost, path_cost };
        self.queue.insert(node, merit);
    }

    fn enqueue_eqc(&mut self, eqc: &'a ClassId, eqc_path_cost: Cost) {
        if !self.is_enqueued_eqc(eqc) {
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                self.enqueue_node(node, eqc_path_cost);
            }
            self.add_enqueued_eqc(eqc);
        }
    }

    fn set_branch_path_cost(&mut self, node: &'a NodeId, path_cost: Cost) {
        self.branch_path_cost.insert(node, path_cost);
    }

    fn add_leaf(&mut self, node: &'a NodeId, merit: Merit) {
        self.leaves.insert(Action::Visit(node), merit);
    }

    fn visit_branch_node(&mut self, node: &'a NodeId, merit: Merit) {
        self.set_branch_path_cost(node, merit.path_cost);

        let egraph = self.egraph;
        let mut unique_child_eqcs: FxHashSet<&'a ClassId> = FxHashSet::default();

        for child in &egraph[node].children {
            let eqc = egraph.nid_to_cid(child);
            if unique_child_eqcs.insert(eqc) {
                self.add_eqc_parent(eqc, child);
                self.enqueue_eqc(eqc, merit.path_cost);
            }
        }

        self.set_node_delay(node, unique_child_eqcs.len());
    }

    fn visit_node(&mut self, node: &'a NodeId, merit: Merit) {
        if self.egraph[node].children.is_empty() {
            self.add_leaf(node, merit);
        } else {
            self.visit_branch_node(node, merit);
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_eqc(target, zero);
        while let Some((node, merit)) = self.dequeue() {
            self.visit_node(node, merit);
        }
    }
}

struct NaiveAStarBottomUpExtractor<'a> {
    egraph:           &'a EGraph,
    eqc_parents:      FxHashMap<&'a ClassId, Vec<&'a NodeId>>,
    node_delay:       FxHashMap<&'a NodeId, usize>,
    branch_path_cost: FxHashMap<&'a NodeId, Cost>,
    queue:            PrioQueue<Action<'a>, Merit>,
    eqc_min_cost:     FxHashMap<&'a ClassId, Cost>,
    eqc_min:          IndexMap<ClassId, NodeId>
}

impl<'a> NaiveAStarBottomUpExtractor<'a> {
    fn init(top_down: NaiveAStarTopDownExtractor<'a>) -> NaiveAStarBottomUpExtractor<'a> {
        NaiveAStarBottomUpExtractor {
            egraph: top_down.egraph,
            eqc_parents: top_down.eqc_parents,
            node_delay: top_down.node_delay,
            branch_path_cost: top_down.branch_path_cost,
            queue: top_down.leaves,
            eqc_min_cost: Default::default(),
            eqc_min: IndexMap::new()
        }
    }

    fn eqc_has_min(&self, eqc: &ClassId) -> bool {
        self.eqc_min.contains_key(eqc)
    }

    fn set_eqc_min_if_new(&mut self, eqc: &'a ClassId, node: &'a NodeId, cost: Cost) {
        self.eqc_min_cost.entry(eqc).or_insert(cost);
        self.eqc_min.entry(eqc.clone()).or_insert_with(|| node.clone());
    }

    fn set_node_delay(&mut self, node: &'a NodeId, delay: usize) {
        self.node_delay.insert(node, delay);
    }

    fn enqueue(&mut self, action: Action<'a>, merit: Merit) {
        self.queue.insert(action, merit);
    }

    fn dequeue(&mut self) -> Option<(Action<'a>, Merit)> {
        self.queue.pop()
    }

    fn get_min_node_cost(&self, node: &'a NodeId) -> Cost {
        let node = &self.egraph[node];
        let child_costs = node.children.iter().map(|child| {
            let eqc = self.egraph.nid_to_cid(child);
            self.eqc_min_cost.get(eqc).copied().unwrap()
        });
        todo!() // TODO: Bottom-up (exact) cost.
    }

    fn enqueue_branch_node_visit(&mut self, node: &'a NodeId) {
        let path_cost = self.branch_path_cost.get(node).copied().unwrap();
        let cost = self.get_min_node_cost(node);
        let merit = Merit { cost: cost, path_cost };
        self.enqueue(Action::Visit(node), merit);
    }

    fn update_eqc_parents(&mut self, eqc: &'a ClassId) {
        let Some(parents) = self.eqc_parents.get(eqc) else { return; };
        let parents: Vec<&'a NodeId> = parents.clone();
        for parent in parents {
            if let Some(n) = self.node_delay.get(parent).copied() {
                if n == 1 {
                    self.enqueue_branch_node_visit(parent);
                    return
                } else if n > 1 {
                    self.set_node_delay(parent, n - 1);
                    return
                }
            }
            panic!("Reached bad path in `update_eqc_parents`.")
        }
    }

    fn assign_eqc(&mut self, eqc: &'a ClassId, node: &'a NodeId, merit: Merit) {
        self.set_eqc_min_if_new(eqc, node, merit.cost);
        self.update_eqc_parents(eqc);
    }

    fn visit_node(&mut self, node: &'a NodeId, merit: Merit) {
        let eqc = self.egraph.nid_to_cid(node);
        self.enqueue(Action::Assign(eqc, node), merit);
    }

    fn run_action(&mut self, action: Action<'a>, merit: Merit) {
        match action {
            Action::Visit(node)       => self.visit_node(node, merit),
            Action::Assign(eqc, node) => self.assign_eqc(eqc, node, merit),
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        while let Some((action, merit)) = self.dequeue() {
            self.run_action(action, merit);
            if self.eqc_has_min(target) { break }
        }
    }
}

pub struct NaiveAStarExtractor;

impl Extractor for NaiveAStarExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // TODO: We currently assume there to be only a single root class from which we extract.
        let target = roots.last().unwrap();
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
