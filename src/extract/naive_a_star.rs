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
    Cost(Cost)
}

impl NodeState {
    fn cost(&self) -> Cost {
        match self {
            NodeState::Cost(cost) => *cost,
            NodeState::Delay(_)   => panic!("Accessed the cost of an unresolved e-node.")
        }
    }
}

// The state associated with an e-class over the course of the extraction. As an e-class' parents
// are only needed at the very moment at which it is assigned a minimal e-node, the two states are
// mutually exclusive and can therefore share a single map.
enum EqcState<'a> {
    // The e-nodes which have this e-class as a child. This is the state of every e-class during the
    // top-down phase.
    Parents(Vec<&'a NodeId>),
    // The bottom-up cost of the e-class' minimal e-node, which is set during the bottom-up phase.
    Cost(Cost)
}

impl<'a> EqcState<'a> {
    fn cost(&self) -> Cost {
        match self {
            EqcState::Cost(cost) => *cost,
            EqcState::Parents(_) => panic!("Accessed the cost of an unresolved e-class.")
        }
    }

    fn parents_mut(&mut self) -> &mut Vec<&'a NodeId> {
        match self {
            EqcState::Parents(parents) => parents,
            EqcState::Cost(_)          => panic!("Accessed the parents of a resolved e-class.")
        }
    }
}

struct NaiveAStarTopDownExtractor<'a> {
    egraph:     &'a EGraph,
    eqc_state:  FxHashMap<&'a ClassId, EqcState<'a>>,
    node_state: FxHashMap<&'a NodeId, NodeState>,
    td_cost:    FxHashMap<&'a ClassId, Cost>,
    queue:      PrioQueue<&'a NodeId, Cost>,
    leaves:     PrioQueue<&'a NodeId, Cost>
}

impl<'a> NaiveAStarTopDownExtractor<'a> {
    fn new(egraph: &'a EGraph) -> NaiveAStarTopDownExtractor<'a> {
        NaiveAStarTopDownExtractor {
            egraph,
            eqc_state:  Default::default(),
            node_state: Default::default(),
            td_cost:    Default::default(),
            queue:      PrioQueue::new(),
            leaves:     PrioQueue::new()
        }
    }

    fn set_node_delay(&mut self, node: &'a NodeId, delay: usize) {
        self.node_state.insert(node, NodeState::Delay(delay));
    }

    fn set_td_cost(&mut self, eqc: &'a ClassId, cost: Cost) {
        self.td_cost.insert(eqc, cost);
    }

    // Determines whether a given e-class has already been enqueued via `enqueue_eqc`. As
    // `enqueue_eqc` always sets `td_cost` for the given e-class, we use membership in this map
    // as the indicator.
    fn is_enqueued_eqc(&self, eqc: &ClassId) -> bool {
        self.td_cost.contains_key(eqc)
    }

    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: &'a NodeId) {
        let state = self.eqc_state.entry(eqc).or_insert_with(|| EqcState::Parents(Vec::new()));
        state.parents_mut().push(node);
    }

    fn dequeue(&mut self) -> Option<(&'a NodeId, Cost)> {
        self.queue.pop()
    }

    fn enqueue_node(&mut self, node: &'a NodeId, td_cost: Cost) {
        let td_cost = td_cost + self.egraph[node].cost;
        self.queue.insert(node, td_cost);
    }

    fn enqueue_eqc(&mut self, eqc: &'a ClassId, td_cost: Cost) {
        if !self.is_enqueued_eqc(eqc) {
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                self.enqueue_node(node, td_cost);
            }
            self.set_td_cost(eqc, td_cost);
        }
    }

    fn add_leaf(&mut self, node: &'a NodeId) {
        let eqc = self.egraph.nid_to_cid(node);
        let bottom_up_cost = self.egraph[node].cost;
        let top_down_cost = self.td_cost[eqc];
        let merit = top_down_cost + bottom_up_cost;
        self.node_state.insert(node, NodeState::Cost(bottom_up_cost));
        self.leaves.insert(node, merit);
    }

    fn visit_children (&mut self, node: &'a NodeId, td_cost : Cost) {
        let egraph = self.egraph;
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

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_eqc(target, zero);
        while let Some((node, td_cost)) = self.dequeue() {
            if self.egraph[node].children.is_empty() {
                self.add_leaf(node);
            } else {
                self.visit_children(node, td_cost);
            }
        }
    }
}

struct NaiveAStarBottomUpExtractor<'a> {
    egraph:     &'a EGraph,
    eqc_state:  FxHashMap<&'a ClassId, EqcState<'a>>,
    node_state: FxHashMap<&'a NodeId, NodeState>,
    td_cost:    FxHashMap<&'a ClassId, Cost>,
    queue:      PrioQueue<&'a NodeId, Cost>,
    eqc_min:    IndexMap<ClassId, NodeId>
}

// Like `ExtractionResult::node_sum_cost`. This is a free function, so that it can be called while
// `NaiveAStarBottomUpExtractor::node_state` is mutably borrowed.
fn min_node_cost<'a>(
    egraph: &'a EGraph, eqc_state: &FxHashMap<&'a ClassId, EqcState<'a>>, node: &'a NodeId
) -> Cost {
    let node = &egraph[node];
    let total_child_cost: Cost = node.children.iter().map(|child| {
        eqc_state[egraph.nid_to_cid(child)].cost()
    }).sum();
    node.cost + total_child_cost
}

impl<'a> NaiveAStarBottomUpExtractor<'a> {
    fn init(top_down: NaiveAStarTopDownExtractor<'a>) -> NaiveAStarBottomUpExtractor<'a> {
        NaiveAStarBottomUpExtractor {
            egraph:     top_down.egraph,
            eqc_state:  top_down.eqc_state,
            node_state: top_down.node_state,
            td_cost:    top_down.td_cost,
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
                    let bottom_up_cost = min_node_cost(egraph, &self.eqc_state, parent);
                    let top_down_cost = self.td_cost[egraph.nid_to_cid(parent)];
                    *state = NodeState::Cost(bottom_up_cost);
                    self.queue.insert(parent, top_down_cost + bottom_up_cost);
                },
                NodeState::Delay(delay) if *delay > 1 => *delay -= 1,
                _ => panic!("{}", BAD_PATH)
            }
        }
    }

    fn run(&mut self) {
        let egraph = self.egraph;
        while let Some((node, _)) = self.queue.pop() {
            let eqc = egraph.nid_to_cid(node);
            let Some(state) = self.eqc_state.get_mut(eqc) else {
                // An e-class without parents can only be the target e-class, so we are done.
                self.eqc_min.insert(eqc.clone(), node.clone());
                break
            };
            // Taking the parents out of the state is what marks the e-class as resolved, so an
            // e-class which is already in the `Cost` state has been assigned a minimal e-node
            // before. Thus, no separate map is needed to detect this.
            let parents = match state {
                EqcState::Cost(_)          => continue,
                EqcState::Parents(parents) => mem::take(parents)
            };
            *state = EqcState::Cost(self.node_state[node].cost());
            self.eqc_min.insert(eqc.clone(), node.clone());
            self.update_parents(parents);
        }
    }
}

const BAD_PATH: &str = "Reached bad path in `NaiveAStarBottomUpExtractor::run`.";

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
