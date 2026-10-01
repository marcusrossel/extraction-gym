use std::mem;

use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

// The state associated with an e-node over the course of the extraction. As an e-node's bottom-up
// cost only becomes known once all of its child e-classes have been assigned a minimal e-node, the
// two states are mutually exclusive and can therefore share a single map.
enum NodeState {
    // The number of child e-classes which have not yet been assigned a minimal e-node. This is set
    // during the top-down phase and counted down during the bottom-up phase.
    Delay(usize),
    // The bottom-up cost of the e-node. Leaves obtain this state during the top-down phase, all
    // other e-nodes when their delay reaches zero during the bottom-up phase.
    Resolved(Cost)
}

impl NodeState {
    fn cost(&self) -> Cost {
        match self {
            NodeState::Resolved(cost) => *cost,
            NodeState::Delay(_)       => panic!("Accessed the cost of an unresolved e-node.")
        }
    }
}

// Whether an e-class has been resolved yet. As an e-class' parents are only needed at the very
// moment at which it is assigned a minimal e-node, the two states are mutually exclusive.
enum EqcStatus<'a> {
    // The e-nodes which have this e-class as a child. This is the status of every e-class during
    // the top-down phase.
    Parents(Vec<&'a NodeId>),
    // The bottom-up cost of the e-class' minimal e-node, which is set during the bottom-up phase.
    Resolved(Cost)
}

impl<'a> EqcStatus<'a> {
    fn cost(&self) -> Cost {
        match self {
            EqcStatus::Resolved(cost) => *cost,
            EqcStatus::Parents(_)     => panic!("Accessed the cost of an unresolved e-class.")
        }
    }

    fn parents_mut(&mut self) -> &mut Vec<&'a NodeId> {
        match self {
            EqcStatus::Parents(parents) => parents,
            EqcStatus::Resolved(_)      => panic!("Accessed the parents of a resolved e-class.")
        }
    }
}

// The state associated with a reached e-class over the course of the extraction. Keeping both parts
// in a single record means that everything about an e-class can be looked up at once.
struct EqcState<'a> {
    // The top-down cost by which the e-class was reached. This is set when the e-class is enqueued
    // during the top-down phase, and never changes afterwards.
    td_cost: Cost,
    status:  EqcStatus<'a>
}

struct TwoPhaseAStarTopDownExtractor<'a> {
    egraph:     &'a EGraph,
    eqc_state:  FxHashMap<&'a ClassId, EqcState<'a>>,
    node_state: FxHashMap<&'a NodeId, NodeState>,
    queue:      PrioQueue<&'a NodeId, Cost>,
    leaves:     PrioQueue<&'a NodeId, Cost>
}

impl<'a> TwoPhaseAStarTopDownExtractor<'a> {
    fn new(egraph: &'a EGraph) -> TwoPhaseAStarTopDownExtractor<'a> {
        TwoPhaseAStarTopDownExtractor {
            egraph,
            eqc_state:  Default::default(),
            node_state: Default::default(),
            queue:      PrioQueue::new(),
            leaves:     PrioQueue::new()
        }
    }

    fn set_node_delay(&mut self, node: &'a NodeId, delay: usize) {
        self.node_state.insert(node, NodeState::Delay(delay));
    }

    // Records `node` as a parent of `eqc`. If `eqc` has not been reached yet, this reaches it (with
    // `node` as its only parent) at the given top-down cost. Otherwise, the e-class keeps the
    // top-down cost it was reached by first.
    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: &'a NodeId, td_cost: Cost) {
        match self.eqc_state.get_mut(eqc) {
            Some(state) => state.status.parents_mut().push(node),
            None        => self.enqueue_eqc(eqc, td_cost, vec![node])
        }
    }

    fn dequeue(&mut self) -> Option<(&'a NodeId, Cost)> {
        self.queue.pop()
    }

    fn enqueue_node(&mut self, node: &'a NodeId, td_cost: Cost) {
        let td_cost = td_cost + self.egraph[node].cost;
        self.queue.insert(node, td_cost);
    }

    // Visits the given leaf: it records its (exact) bottom-up cost, and is added to `leaves` at its
    // merit. This reads the top-down cost of the leaf's e-class, which must thus be reached already.
    fn add_leaf(&mut self, node: &'a NodeId) {
        let eqc = self.egraph.nid_to_cid(node);
        let bottom_up_cost = self.egraph[node].cost;
        let top_down_cost = self.eqc_state[eqc].td_cost;
        let merit = top_down_cost + bottom_up_cost;
        self.node_state.insert(node, NodeState::Resolved(bottom_up_cost));
        self.leaves.insert(node, merit);
    }

    // Reaches the given e-class at the given top-down cost: records it with that cost and the given
    // parents, visits its leaves right away, and enqueues its branch e-nodes for a visit. This must
    // only be called for an e-class which has not been reached before (see `add_eqc_parent`).
    fn enqueue_eqc(&mut self, eqc: &'a ClassId, td_cost: Cost, parents: Vec<&'a NodeId>) {
        // This has to happen before the loop, as `add_leaf` reads the `td_cost` of `eqc`.
        self.eqc_state.insert(eqc, EqcState { td_cost, status: EqcStatus::Parents(parents) });
        let egraph = self.egraph;
        for node in &egraph.classes()[eqc].nodes {
            // Visiting a leaf does nothing but add it to `leaves`, so we do that directly instead
            // of enqueuing it for a visit. This is the same as in the (interleaved) A* extraction,
            // where it is also needed for correctness (see `AStarExt::enqueue_visit_eqc`).
            if egraph[node].children.is_empty() {
                self.add_leaf(node);
            } else {
                self.enqueue_node(node, td_cost);
            }
        }
    }

    fn visit_children (&mut self, node: &'a NodeId, td_cost : Cost) {
        let egraph = self.egraph;
        let mut unique_child_eqcs: FxHashSet<&'a ClassId> = FxHashSet::default();

        for child in &egraph[node].children {
            let eqc = egraph.nid_to_cid(child);
            if unique_child_eqcs.insert(eqc) {
                self.add_eqc_parent(eqc, node, td_cost);
            }
        }

        self.set_node_delay(node, unique_child_eqcs.len());
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_eqc(target, zero, Vec::new());
        // Only branch e-nodes are ever enqueued, as `enqueue_eqc` visits leaves right away.
        while let Some((node, td_cost)) = self.dequeue() {
            self.visit_children(node, td_cost);
        }
    }
}

struct TwoPhaseAStarBottomUpExtractor<'a> {
    egraph:     &'a EGraph,
    eqc_state:  FxHashMap<&'a ClassId, EqcState<'a>>,
    node_state: FxHashMap<&'a NodeId, NodeState>,
    queue:      PrioQueue<&'a NodeId, Cost>,
    eqc_min:    IndexMap<ClassId, NodeId>
}

// Like `ExtractionResult::node_sum_cost`. This is a free function, so that it can be called while
// `TwoPhaseAStarBottomUpExtractor::node_state` is mutably borrowed.
fn min_node_cost<'a>(
    egraph: &'a EGraph, eqc_state: &FxHashMap<&'a ClassId, EqcState<'a>>, node: &'a NodeId
) -> Cost {
    let node = &egraph[node];
    let total_child_cost: Cost = node.children.iter().map(|child| {
        eqc_state[egraph.nid_to_cid(child)].status.cost()
    }).sum();
    node.cost + total_child_cost
}

impl<'a> TwoPhaseAStarBottomUpExtractor<'a> {
    fn init(top_down: TwoPhaseAStarTopDownExtractor<'a>) -> TwoPhaseAStarBottomUpExtractor<'a> {
        TwoPhaseAStarBottomUpExtractor {
            egraph:     top_down.egraph,
            eqc_state:  top_down.eqc_state,
            node_state: top_down.node_state,
            queue:      top_down.leaves,
            eqc_min:    IndexMap::new()
        }
    }

    // Accesses the fields directly (instead of via methods), so that the borrow checker sees the
    // accesses to the maps as disjoint. This allows `node_state` to be updated in place, while the
    // other maps are being read.
    fn update_parents(&mut self, parents: Vec<&'a NodeId>) {
        let egraph = self.egraph;
        for parent in parents {
            let state = self.node_state.get_mut(parent).expect(BAD_PATH);
            match state {
                NodeState::Delay(1) => {
                    // If the parent's e-class has already been resolved (by another of its
                    // e-nodes), the parent cannot become its minimal e-node, so there's no need to
                    // enqueue it. Its state then simply stays at `Delay(1)`, as it is never looked
                    // at again.
                    let parent_eqc_state = &self.eqc_state[egraph.nid_to_cid(parent)];
                    if let EqcStatus::Resolved(_) = parent_eqc_state.status { continue }
                    let top_down_cost = parent_eqc_state.td_cost;
                    let bottom_up_cost = min_node_cost(egraph, &self.eqc_state, parent);
                    *state = NodeState::Resolved(bottom_up_cost);
                    self.queue.insert(parent, top_down_cost + bottom_up_cost);
                },
                NodeState::Delay(delay) if *delay > 1 => *delay -= 1,
                _ => panic!("{}", BAD_PATH)
            }
        }
    }

    fn run(&mut self, target: &ClassId) {
        let egraph = self.egraph;
        while let Some((node, _)) = self.queue.pop() {
            let eqc = egraph.nid_to_cid(node);
            // Every e-class which was reached during the top-down phase has a state.
            let state = self.eqc_state.get_mut(eqc).expect(BAD_PATH);
            // Taking the parents out of the status is what marks the e-class as resolved, so an
            // e-class which already has the `Cost` status has been assigned a minimal e-node
            // before. Thus, no separate map is needed to detect this. If the node's e-class is
            // already resolved, there's nothing to do.
            let parents = match &mut state.status {
                EqcStatus::Resolved(_)      => continue,
                EqcStatus::Parents(parents) => mem::take(parents)
            };
            state.status = EqcStatus::Resolved(self.node_state[node].cost());
            self.eqc_min.insert(eqc.clone(), node.clone());
            if eqc == target { break }
            self.update_parents(parents);
        }
    }
}

const BAD_PATH: &str = "Reached bad path in `TwoPhaseAStarBottomUpExtractor::run`.";

pub struct TwoPhaseAStarExtractor;

impl Extractor for TwoPhaseAStarExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // TODO: We currently assume there to be only a single root class from which we extract. Add
        //       a field to the `ExtractorDetail` in `main.rs` where this can be declared.
        assert_eq!(roots.len(), 1);
        let target = &roots[0];
        let mut top_down = TwoPhaseAStarTopDownExtractor::new(egraph);
        top_down.run(target);
        let mut bottom_up = TwoPhaseAStarBottomUpExtractor::init(top_down);
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
