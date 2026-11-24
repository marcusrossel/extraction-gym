use std::collections::BTreeMap;
use std::mem;

use rustc_hash::FxHashMap;

use super::*;

// An action scheduled by the queue, carrying a dense `u32` index into `egraph.nodes`.
// Assignments carry their exact `bottom_up_cost` directly so leaves and ready branch e-nodes
// never need to write or read a `Resolved` entry in `node_state`.
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

// Whether an e-class has been resolved yet. As an e-class' parents are only needed at the moment
// it is assigned a minimal e-node, the two states are mutually exclusive.
enum EqcStatus {
    Parents(Vec<u32>),
    Resolved(Cost),
}

// The state associated with a reached e-class over the course of the extraction.
struct EqcState {
    // The top-down cost by which the e-class was reached. Set once when first reached.
    td_cost: Cost,
    status: EqcStatus,
}

struct InterleavedAStarExt<'a> {
    egraph: &'a EGraph,
    queue: BTreeMap<Cost, Vec<Action>>,
    node_state: FxHashMap<u32, PendingNode>,
    eqc_state: FxHashMap<&'a ClassId, EqcState>,
    eqc_min: IndexMap<ClassId, NodeId>,
}

impl<'a> InterleavedAStarExt<'a> {
    fn new(egraph: &'a EGraph) -> InterleavedAStarExt<'a> {
        InterleavedAStarExt {
            egraph,
            queue: BTreeMap::new(),
            node_state: Default::default(),
            eqc_state: Default::default(),
            eqc_min: Default::default(),
        }
    }

    fn enqueue(&mut self, action: Action, cost: Cost) {
        self.queue.entry(cost).or_default().push(action);
    }

    fn dequeue(&mut self) -> Option<Action> {
        let mut entry = self.queue.first_entry()?;
        let action = entry.get_mut().pop().unwrap();
        if entry.get().is_empty() {
            entry.remove();
        }
        Some(action)
    }

    fn enqueue_visit_eqc(&mut self, eqc: &'a ClassId, td_cost: Cost, parents: Vec<u32>) {
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
                self.enqueue(Action::Assign(node_idx, node.cost), td_cost + node.cost);
            } else {
                self.enqueue(Action::Visit(node_idx), td_cost + node.cost);
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
            match self.eqc_state.get_mut(child) {
                Some(state) => match &mut state.status {
                    EqcStatus::Resolved(cost) => acc_cost += *cost,
                    EqcStatus::Parents(parents) => {
                        parents.push(node);
                        delay += 1;
                    }
                },
                None => {
                    self.enqueue_visit_eqc(child, node_td_cost, vec![node]);
                    delay += 1;
                }
            }
        }

        if delay == 0 {
            self.enqueue(Action::Assign(node, acc_cost), td_cost + acc_cost);
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

    fn update_parents(&mut self, parents: Vec<u32>, resolved_cost: Cost) {
        for parent in parents {
            let pending = self.node_state.get_mut(&parent).expect(BAD_PATH);
            pending.cost += resolved_cost;
            pending.delay -= 1;
            if pending.delay == 0 {
                let (_, parent_node) = self.egraph.nodes.get_index(parent as usize).unwrap();
                let parent_eqc_state = &self.eqc_state[&parent_node.eclass];
                if let EqcStatus::Resolved(_) = parent_eqc_state.status {
                    continue;
                }
                let bottom_up_cost = pending.cost;
                self.enqueue(
                    Action::Assign(parent, bottom_up_cost),
                    parent_eqc_state.td_cost + bottom_up_cost,
                );
            }
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        let egraph = self.egraph;
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_visit_eqc(target, zero, Vec::new());

        while let Some(action) = self.dequeue() {
            let node = match action {
                Action::Visit(node) | Action::Assign(node, _) => node,
            };
            let (nid, n) = egraph.nodes.get_index(node as usize).unwrap();
            let eqc = &n.eclass;
            let state = self.eqc_state.get_mut(eqc).expect(BAD_PATH);
            if matches!(state.status, EqcStatus::Resolved(_)) {
                continue;
            }
            match action {
                Action::Visit(node) => {
                    let td_cost = state.td_cost;
                    self.visit_node(node, n, td_cost);
                }
                Action::Assign(_, resolved_cost) => {
                    let prev = mem::replace(&mut state.status, EqcStatus::Resolved(resolved_cost));
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
            choices: ext.eqc_min,
        }
    }
}
