use std::mem;

use rustc_hash::FxHashMap;

use super::common::{BucketQueue, EqcStatus, node_cost};
use super::*;

// An action scheduled by the queue, carrying a dense `u32` index into `egraph.nodes`.
// Assignments carry their exact `bottom_up_cost` directly so leaves and ready branch e-nodes
// never need to write or read a `Resolved` entry in `node_delay`.
enum Action {
    Visit(u32),
    Assign(u32, Cost),
}

// The state associated with a reached e-class over the course of the extraction.
struct EqcState {
    // The top-down cost by which the e-class was reached. Set once when first reached.
    td_cost: Cost,
    status: EqcStatus,
}

struct InterleavedAStarExt<'a> {
    egraph: &'a EGraph,
    queue: BucketQueue<Action>,
    node_delay: FxHashMap<u32, u32>,
    eqc_state: FxHashMap<&'a ClassId, EqcState>,
    eqc_min: IndexMap<ClassId, NodeId>,
}

impl<'a> InterleavedAStarExt<'a> {
    fn new(egraph: &'a EGraph) -> InterleavedAStarExt<'a> {
        InterleavedAStarExt {
            egraph,
            queue: BucketQueue::new(),
            node_delay: Default::default(),
            eqc_state: Default::default(),
            eqc_min: Default::default(),
        }
    }

    fn enqueue(&mut self, action: Action, cost: Cost) {
        self.queue.push(cost, action);
    }

    fn dequeue(&mut self) -> Option<Action> {
        self.queue.pop()
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
                let leaf_cost = node_cost(egraph, node, |_| unreachable!());
                self.enqueue(Action::Assign(node_idx, leaf_cost), td_cost + leaf_cost);
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

        for child in &n.children {
            let child = egraph.nid_to_cid(child);
            match self.eqc_state.get_mut(child) {
                Some(state) => match &mut state.status {
                    EqcStatus::Resolved(_) => {}
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
            let bottom_up_cost =
                node_cost(egraph, n, |cid| self.eqc_state[cid].status.resolved_cost());
            self.enqueue(
                Action::Assign(node, bottom_up_cost),
                td_cost + bottom_up_cost,
            );
        } else {
            self.node_delay.insert(node, delay);
        }
    }

    fn update_parents(&mut self, parents: Vec<u32>) {
        for parent in parents {
            let delay = self.node_delay.get_mut(&parent).expect(BAD_PATH);
            *delay -= 1;
            if *delay == 0 {
                let (_, parent_node) = self.egraph.nodes.get_index(parent as usize).unwrap();
                let parent_eqc_state = &self.eqc_state[&parent_node.eclass];
                if let EqcStatus::Resolved(_) = parent_eqc_state.status {
                    continue;
                }
                let bottom_up_cost = node_cost(self.egraph, parent_node, |cid| {
                    self.eqc_state[cid].status.resolved_cost()
                });
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
                        self.update_parents(parents);
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
