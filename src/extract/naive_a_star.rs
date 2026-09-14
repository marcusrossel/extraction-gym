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

struct NaiveAStarTopDownExtractor<'a> {
    egraph:      &'a EGraph,
    eqc_parents: FxHashMap<&'a ClassId, Vec<&'a NodeId>>,
    node_state:  FxHashMap<&'a NodeId, NodeState>,
    parent_cost: FxHashMap<&'a ClassId, Cost>,
    queue:       PrioQueue<&'a NodeId, Cost>,
    leaves:      PrioQueue<&'a NodeId, Cost>
}

impl<'a> NaiveAStarTopDownExtractor<'a> {
    fn new(egraph: &'a EGraph) -> NaiveAStarTopDownExtractor<'a> {
        NaiveAStarTopDownExtractor {
            egraph,
            eqc_parents: Default::default(),
            node_state:  Default::default(),
            parent_cost: Default::default(),
            queue:       PrioQueue::new(),
            leaves:      PrioQueue::new()
        }
    }

    fn set_node_delay(&mut self, node: &'a NodeId, delay: usize) {
        self.node_state.insert(node, NodeState::Delay(delay));
    }

    fn set_parent_cost(&mut self, eqc: &'a ClassId, cost: Cost) {
        self.parent_cost.insert(eqc, cost);
    }

    // Determines whether a given e-class has already been enqueued via `enqueue_eqc`. As
    // `enqueue_eqc` always sets `parent_cost` for the given e-class, we use membership in this map
    // as the indicator.
    fn is_enqueued_eqc(&self, eqc: &ClassId) -> bool {
        self.parent_cost.contains_key(eqc)
    }

    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: &'a NodeId) {
        self.eqc_parents.entry(eqc).or_insert_with(Vec::new).push(node);
    }

    fn dequeue(&mut self) -> Option<&'a NodeId> {
        self.queue.pop().map(|(node, _)| node)
    }

    fn enqueue_node(&mut self, node: &'a NodeId, parent_cost: Cost) {
        let td_cost = parent_cost + self.egraph[node].cost;
        self.queue.insert(node, td_cost);
    }

    fn enqueue_eqc(&mut self, eqc: &'a ClassId, parent_cost: Cost) {
        if !self.is_enqueued_eqc(eqc) {
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                self.enqueue_node(node, parent_cost);
            }
            self.set_parent_cost(eqc, parent_cost);
        }
    }

    fn add_leaf(&mut self, node: &'a NodeId) {
        let eqc = self.egraph.nid_to_cid(node);
        let bottom_up_cost = self.egraph[node].cost;
        let top_down_cost = self.parent_cost[eqc];
        let merit = top_down_cost + bottom_up_cost;
        self.node_state.insert(node, NodeState::Cost(bottom_up_cost));
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
                let eqc = egraph.nid_to_cid(node);
                let td_cost = self.parent_cost[eqc] + self.egraph[node].cost;
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
    node_state:   FxHashMap<&'a NodeId, NodeState>,
    parent_cost:  FxHashMap<&'a ClassId, Cost>,
    queue:        PrioQueue<&'a NodeId, Cost>,
    eqc_min_cost: FxHashMap<&'a ClassId, Cost>,
    eqc_min:      IndexMap<ClassId, NodeId>
}

// Like `ExtractionResult::node_sum_cost`. This is a free function, so that it can be called while
// `NaiveAStarBottomUpExtractor::node_state` is mutably borrowed.
fn min_node_cost<'a>(
    egraph: &'a EGraph, eqc_min_cost: &FxHashMap<&'a ClassId, Cost>, node: &'a NodeId
) -> Cost {
    let node = &egraph[node];
    let total_child_cost: Cost = node.children.iter().map(|child| {
        eqc_min_cost[egraph.nid_to_cid(child)]
    }).sum();
    node.cost + total_child_cost
}

impl<'a> NaiveAStarBottomUpExtractor<'a> {
    fn init(top_down: NaiveAStarTopDownExtractor<'a>) -> NaiveAStarBottomUpExtractor<'a> {
        NaiveAStarBottomUpExtractor {
            egraph:       top_down.egraph,
            eqc_parents:  top_down.eqc_parents,
            node_state:   top_down.node_state,
            parent_cost:  top_down.parent_cost,
            queue:        top_down.leaves,
            eqc_min_cost: Default::default(),
            eqc_min:      IndexMap::new()
        }
    }

    fn run(&mut self) {
        let egraph = self.egraph;
        // The remaining fields are destructured, so that the borrow checker sees the accesses to the
        // maps below as disjoint. This allows `node_state` to be updated in place, while the other
        // maps are being read.
        let Self { eqc_parents, node_state, parent_cost, queue, eqc_min_cost, eqc_min, .. } = self;

        while let Some((node, _)) = queue.pop() {
            let eqc = egraph.nid_to_cid(node);
            // Determines whether a given e-class has already been assigned a minimal e-node. As
            // `eqc_min_cost` and `eqc_min` are always set in tandem, we use membership in the
            // former as the indicator, as it uses the faster hasher.
            if eqc_min_cost.contains_key(eqc) { continue }
            eqc_min_cost.insert(eqc, node_state[node].cost());
            eqc_min.insert(eqc.clone(), node.clone());

            let Some(parents) = eqc_parents.get(eqc) else { break };
            for parent in parents.iter().copied() {
                let state = node_state.get_mut(parent).expect(BAD_PATH);
                match state {
                    NodeState::Delay(1) => {
                        let bottom_up_cost = min_node_cost(egraph, eqc_min_cost, parent);
                        let top_down_cost = parent_cost[egraph.nid_to_cid(parent)];
                        *state = NodeState::Cost(bottom_up_cost);
                        queue.insert(parent, top_down_cost + bottom_up_cost);
                    },
                    NodeState::Delay(delay) if *delay > 1 => *delay -= 1,
                    _ => panic!("{}", BAD_PATH)
                }
            }
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
