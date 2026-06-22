use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

#[derive(PartialEq, Eq, Clone, Copy)]
enum ECost {
    Normal(Cost),
    Inf
}

impl PartialOrd for ECost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ECost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (ECost::Inf, ECost::Inf) => std::cmp::Ordering::Equal,
            (ECost::Inf, ECost::Normal(_)) => std::cmp::Ordering::Greater,
            (ECost::Normal(_), ECost::Inf) => std::cmp::Ordering::Less,
            (ECost::Normal(c1), ECost::Normal(c2)) => c1.cmp(c2),
        }
    }
}

impl std::ops::Add for ECost {
    type Output = ECost;

    fn add(self, rhs: ECost) -> ECost {
        match (self, rhs) {
            (ECost::Normal(c1), ECost::Normal(c2)) => ECost::Normal(c1 + c2),
            _ => ECost::Inf
        }
    }
}

#[derive(PartialEq, Eq)]
struct Merit {
    cost: ECost,
    path_cost: ECost
}

impl Merit {
    fn total(&self) -> ECost {
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
enum Action {
    Visit(NodeId),
    Assign(ClassId, NodeId)
}

struct NaiveAStarExtractorState {
    eqc_parents:   FxHashMap<ClassId, Vec<NodeId>>,
    node_delay:    FxHashMap<NodeId, u8>,
    app_path_cost: FxHashMap<NodeId, ECost>,
    enqueued_eqcs: FxHashSet<ClassId>,
    down_queue:    PrioQueue<NodeId, ECost>, // `queue` in Lean
    up_queue:      PrioQueue<Action, Merit>, // `leaves` in Lean
    eqc_min:       FxHashMap<ClassId, (NodeId, Cost)> // TODO Split this into two maps if necessary.
}

impl NaiveAStarExtractorState {
    fn new(num_eqcs: usize, num_nodes: usize) -> NaiveAStarExtractorState {
        NaiveAStarExtractorState {
            eqc_parents: FxHashMap::with_capacity_and_hasher(num_eqcs, Default::default()),
            node_delay: FxHashMap::with_capacity_and_hasher(num_nodes, Default::default()),
            app_path_cost: FxHashMap::with_capacity_and_hasher(num_nodes, Default::default()),
            enqueued_eqcs: FxHashSet::with_capacity_and_hasher(num_eqcs, Default::default()),
            down_queue: PrioQueue::new(),
            up_queue: PrioQueue::new(),
            eqc_min: FxHashMap::with_capacity_and_hasher(num_nodes, Default::default()),
        }
    }
}

pub struct NaiveAStarExtractor;

impl Extractor for NaiveAStarExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        let mut state = NaiveAStarExtractorState::new(egraph.classes().len(), egraph.nodes.len());
        todo!()
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
