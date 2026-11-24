use std::mem;

use rustc_hash::FxHashMap;

use super::*;

// State of a visited branch e-node waiting for its child e-classes to be resolved during the
// bottom-up phase. `delay` counts remaining unresolved child occurrences (not deduplicated), and
// `cost` accumulates `node.cost` plus resolved child costs.
struct PendingNode {
    delay: u32,
    cost: Cost,
}

// Whether an e-class has been resolved yet. As an e-class' parents are only needed at the moment
// it is assigned a minimal e-node, the two states are mutually exclusive.
enum EqcStatus {
    Parents(Vec<u32>),
    Resolved,
}

// The state associated with a reached e-class over the course of the extraction.
struct EqcState {
    // The top-down cost by which the e-class was reached during the top-down phase.
    td_cost: Cost,
    status: EqcStatus,
}

struct TwoPhaseAStarTopDownExtractor<'a> {
    egraph: &'a EGraph,
    eqc_state: FxHashMap<&'a ClassId, EqcState>,
    node_state: FxHashMap<u32, PendingNode>,
    queue: PrioQueue<u32, Cost>,
    leaves: PrioQueue<(u32, Cost), Cost>,
}

impl<'a> TwoPhaseAStarTopDownExtractor<'a> {
    fn new(egraph: &'a EGraph) -> TwoPhaseAStarTopDownExtractor<'a> {
        TwoPhaseAStarTopDownExtractor {
            egraph,
            eqc_state: Default::default(),
            node_state: Default::default(),
            queue: PrioQueue::new(),
            leaves: PrioQueue::new(),
        }
    }

    fn enqueue_eqc(&mut self, eqc: &'a ClassId, td_cost: Cost, parents: Vec<u32>) {
        self.eqc_state.insert(
            eqc,
            EqcState {
                td_cost,
                status: EqcStatus::Parents(parents),
            },
        );
        let egraph = self.egraph;
        for node_id in &egraph.classes()[eqc].nodes {
            let (idx, _, node) = egraph.nodes.get_full(node_id).unwrap();
            let node_idx = idx as u32;
            if node.children.is_empty() {
                self.leaves
                    .insert((node_idx, node.cost), td_cost + node.cost);
            } else {
                self.queue.insert(node_idx, td_cost + node.cost);
            }
        }
    }

    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: u32, td_cost: Cost) {
        match self.eqc_state.get_mut(eqc) {
            Some(state) => match &mut state.status {
                EqcStatus::Parents(parents) => parents.push(node),
                EqcStatus::Resolved => unreachable!(),
            },
            None => self.enqueue_eqc(eqc, td_cost, vec![node]),
        }
    }

    fn visit_children(&mut self, node: u32, td_cost: Cost) {
        let egraph = self.egraph;
        let (_, n) = egraph.nodes.get_index(node as usize).unwrap();
        for child in &n.children {
            let eqc = egraph.nid_to_cid(child);
            self.add_eqc_parent(eqc, node, td_cost);
        }
        self.node_state.insert(
            node,
            PendingNode {
                delay: n.children.len() as u32,
                cost: n.cost,
            },
        );
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_eqc(target, zero, Vec::new());
        while let Some((node, td_cost)) = self.queue.pop() {
            self.visit_children(node, td_cost);
        }
    }
}

struct TwoPhaseAStarBottomUpExtractor<'a> {
    egraph: &'a EGraph,
    eqc_state: FxHashMap<&'a ClassId, EqcState>,
    node_state: FxHashMap<u32, PendingNode>,
    queue: PrioQueue<(u32, Cost), Cost>,
    eqc_min: IndexMap<ClassId, NodeId>,
}

impl<'a> TwoPhaseAStarBottomUpExtractor<'a> {
    fn init(top_down: TwoPhaseAStarTopDownExtractor<'a>) -> TwoPhaseAStarBottomUpExtractor<'a> {
        TwoPhaseAStarBottomUpExtractor {
            egraph: top_down.egraph,
            eqc_state: top_down.eqc_state,
            node_state: top_down.node_state,
            queue: top_down.leaves,
            eqc_min: IndexMap::new(),
        }
    }

    fn update_parents(&mut self, parents: Vec<u32>, resolved_cost: Cost) {
        for parent in parents {
            let pending = self.node_state.get_mut(&parent).expect(BAD_PATH);
            pending.cost += resolved_cost;
            pending.delay -= 1;
            if pending.delay == 0 {
                let (_, parent_node) = self.egraph.nodes.get_index(parent as usize).unwrap();
                let parent_eqc_state = &self.eqc_state[&parent_node.eclass];
                if let EqcStatus::Resolved = parent_eqc_state.status {
                    continue;
                }
                let bottom_up_cost = pending.cost;
                self.queue.insert(
                    (parent, bottom_up_cost),
                    parent_eqc_state.td_cost + bottom_up_cost,
                );
            }
        }
    }

    fn run(&mut self, target: &ClassId) {
        let egraph = self.egraph;
        while let Some(((node, resolved_cost), _)) = self.queue.pop() {
            let (nid, n) = egraph.nodes.get_index(node as usize).unwrap();
            let eqc = &n.eclass;
            let state = self.eqc_state.get_mut(eqc).expect(BAD_PATH);
            if matches!(state.status, EqcStatus::Resolved) {
                continue;
            }
            let prev = mem::replace(&mut state.status, EqcStatus::Resolved);
            self.eqc_min.insert(eqc.clone(), nid.clone());
            if eqc == target {
                break;
            }
            if let EqcStatus::Parents(parents) = prev {
                self.update_parents(parents, resolved_cost);
            }
        }
    }
}

const BAD_PATH: &str = "Reached bad path in `TwoPhaseAStarBottomUpExtractor::run`.";

pub struct TwoPhaseAStarOptimizedExtractor;

impl Extractor for TwoPhaseAStarOptimizedExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // TODO: We currently assume there to be only a single root class from which we extract. Add
        //       a field to the `ExtractorDetail` in `main.rs` where this can be declared.
        assert_eq!(roots.len(), 1);
        let target = &roots[0];
        let mut top_down = TwoPhaseAStarTopDownExtractor::new(egraph);
        top_down.run(target);
        let mut bottom_up = TwoPhaseAStarBottomUpExtractor::init(top_down);
        bottom_up.run(target);
        ExtractionResult {
            choices: bottom_up.eqc_min,
        }
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
