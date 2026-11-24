use std::collections::hash_map::Entry;
use std::mem;

use rustc_hash::FxHashMap;

use super::*;

// An action scheduled by the queue, carrying a dense `u32` index into `egraph.nodes`.
// Assignments carry their exact `bottom_up_cost` directly so leaves and ready branch e-nodes
// never need to write or read a `Resolved` entry in `node_state`.
#[derive(PartialEq, Eq)]
enum Action {
    Visit(u32),
    Assign(u32, Cost),
}

// State of a visited branch e-node waiting for one or more child e-classes to be resolved.
// `delay` counts remaining unresolved child occurrences (not deduplicated), and `cost`
// accumulates `node.cost` plus the costs of already-resolved child occurrences.
struct PendingNode {
    delay: u32,
    cost: Cost,
}

// Inline parent list avoiding heap allocations for e-classes with 0 or 1 parent.
enum Parents {
    Empty,
    One(u32),
    Many(Vec<u32>),
}

impl Parents {
    #[inline]
    fn push(&mut self, node: u32) {
        match self {
            Parents::Empty => *self = Parents::One(node),
            Parents::One(first) => *self = Parents::Many(vec![*first, node]),
            Parents::Many(vec) => vec.push(node),
        }
    }
}

// Whether an e-class has been resolved yet. As an e-class' parents are only needed at the moment
// it is assigned a minimal e-node, the two states are mutually exclusive.
enum EqcStatus {
    Parents(Parents),
    Resolved { cost: Cost, choice: u32 },
}

// The state associated with a reached e-class over the course of the extraction.
struct EqcState {
    // The top-down cost by which the e-class was reached. Set once when first reached.
    td_cost: Cost,
    status: EqcStatus,
}

struct InterleavedAStarExt<'a> {
    egraph: &'a EGraph,
    queue: PrioQueue<Action, Cost>,
    node_state: FxHashMap<u32, PendingNode>,
    eqc_state: FxHashMap<&'a ClassId, EqcState>,
}

impl<'a> InterleavedAStarExt<'a> {
    fn new(egraph: &'a EGraph) -> InterleavedAStarExt<'a> {
        InterleavedAStarExt {
            egraph,
            queue: PrioQueue::new(),
            node_state: Default::default(),
            eqc_state: Default::default(),
        }
    }

    fn enqueue_eqc_nodes(&mut self, eqc: &ClassId, td_cost: Cost) {
        let egraph = self.egraph;
        for node_id in &egraph.classes()[eqc].nodes {
            let (idx, _, node) = egraph.nodes.get_full(node_id).unwrap();
            let node_idx = idx as u32;
            if node.children.is_empty() {
                self.queue
                    .insert(Action::Assign(node_idx, node.cost), td_cost + node.cost);
            } else {
                self.queue
                    .insert(Action::Visit(node_idx), td_cost + node.cost);
            }
        }
    }

    // Visits `node`, whose e-class is reached at `td_cost`.
    fn visit_node(&mut self, node: u32, n: &'a Node, td_cost: Cost) {
        let egraph = self.egraph;
        let node_td_cost = td_cost + n.cost;
        let mut delay = 0u32;
        let mut acc_cost = n.cost;

        for child in &n.children {
            let child = egraph.nid_to_cid(child);
            match self.eqc_state.entry(child) {
                Entry::Occupied(mut e) => match &mut e.get_mut().status {
                    EqcStatus::Resolved { cost, .. } => acc_cost += *cost,
                    EqcStatus::Parents(parents) => {
                        parents.push(node);
                        delay += 1;
                    }
                },
                Entry::Vacant(e) => {
                    e.insert(EqcState {
                        td_cost: node_td_cost,
                        status: EqcStatus::Parents(Parents::One(node)),
                    });
                    self.enqueue_eqc_nodes(child, node_td_cost);
                    delay += 1;
                }
            }
        }

        if delay == 0 {
            self.queue
                .insert(Action::Assign(node, acc_cost), td_cost + acc_cost);
        } else {
            self.node_state.insert(
                node,
                PendingNode {
                    delay,
                    cost: acc_cost,
                },
            );
        }
    }

    #[inline]
    fn update_parent(&mut self, parent: u32, resolved_cost: Cost) {
        let pending = self.node_state.get_mut(&parent).expect(BAD_PATH);
        pending.cost += resolved_cost;
        pending.delay -= 1;
        if pending.delay == 0 {
            let (_, parent_node) = self.egraph.nodes.get_index(parent as usize).unwrap();
            let parent_eqc_state = &self.eqc_state[&parent_node.eclass];
            if let EqcStatus::Resolved { .. } = parent_eqc_state.status {
                return;
            }
            let bottom_up_cost = pending.cost;
            self.queue.insert(
                Action::Assign(parent, bottom_up_cost),
                parent_eqc_state.td_cost + bottom_up_cost,
            );
        }
    }

    fn update_parents(&mut self, parents: Parents, resolved_cost: Cost) {
        match parents {
            Parents::Empty => {}
            Parents::One(parent) => self.update_parent(parent, resolved_cost),
            Parents::Many(parents) => {
                for parent in parents {
                    self.update_parent(parent, resolved_cost);
                }
            }
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        let egraph = self.egraph;
        let zero = NotNan::new(0.0).unwrap();
        self.eqc_state.insert(
            target,
            EqcState {
                td_cost: zero,
                status: EqcStatus::Parents(Parents::Empty),
            },
        );
        self.enqueue_eqc_nodes(target, zero);

        while let Some((action, _)) = self.queue.pop() {
            let node = match action {
                Action::Visit(node) | Action::Assign(node, _) => node,
            };
            let (_, n) = egraph.nodes.get_index(node as usize).unwrap();
            let eqc = &n.eclass;
            let state = self.eqc_state.get_mut(eqc).expect(BAD_PATH);
            if matches!(state.status, EqcStatus::Resolved { .. }) {
                continue;
            }
            match action {
                Action::Visit(node) => {
                    let td_cost = state.td_cost;
                    self.visit_node(node, n, td_cost);
                }
                Action::Assign(node, resolved_cost) => {
                    let prev = mem::replace(
                        &mut state.status,
                        EqcStatus::Resolved {
                            cost: resolved_cost,
                            choice: node,
                        },
                    );
                    if eqc == target {
                        break;
                    }
                    if let EqcStatus::Parents(parents) = prev {
                        self.update_parents(parents, resolved_cost);
                    }
                }
            }
        }
    }

    // Reconstructs `ExtractionResult::choices` for only the e-classes reachable from `target`.
    fn build_choices(&self, target: &'a ClassId) -> IndexMap<ClassId, NodeId> {
        let mut choices = IndexMap::new();
        let mut stack: Vec<&'a ClassId> = vec![target];
        while let Some(eqc) = stack.pop() {
            if choices.contains_key(eqc) {
                continue;
            }
            let choice = match self.eqc_state[eqc].status {
                EqcStatus::Resolved { choice, .. } => choice,
                EqcStatus::Parents(_) => panic!("{}", BAD_PATH),
            };
            let (nid, n) = self.egraph.nodes.get_index(choice as usize).unwrap();
            choices.insert(eqc.clone(), nid.clone());
            for child in &n.children {
                stack.push(self.egraph.nid_to_cid(child));
            }
        }
        choices
    }
}

const BAD_PATH: &str = "Reached bad path in `InterleavedAStarExt::run`.";

pub struct InterleavedAStarOptimizedExtractor;

impl Extractor for InterleavedAStarOptimizedExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // TODO: We currently assume there to be only a single root class from which we extract. Add
        //       a field to the `ExtractorDetail` in `main.rs` where this can be declared.
        assert_eq!(roots.len(), 1);
        let target = &roots[0];
        let mut ext = InterleavedAStarExt::new(egraph);
        ext.run(target);
        ExtractionResult {
            choices: ext.build_choices(target),
        }
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
