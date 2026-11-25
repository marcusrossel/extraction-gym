use std::mem;

use rustc_hash::FxHashMap;

use super::common::{BucketQueue, EqcStatus, node_cost};
use super::*;

// The state associated with a reached e-class over the course of the extraction.
struct EqcState {
    // The top-down cost by which the e-class was reached during the top-down phase.
    td_cost: Cost,
    status: EqcStatus,
}

struct TwoPhaseAStarTopDownExtractor<'a> {
    egraph: &'a EGraph,
    eqc_state: FxHashMap<&'a ClassId, EqcState>,
    node_delay: FxHashMap<u32, u32>,
    queue: BucketQueue<u32>,
    leaves: BucketQueue<(u32, Cost)>,
}

impl<'a> TwoPhaseAStarTopDownExtractor<'a> {
    fn new(egraph: &'a EGraph) -> TwoPhaseAStarTopDownExtractor<'a> {
        TwoPhaseAStarTopDownExtractor {
            egraph,
            eqc_state: Default::default(),
            node_delay: Default::default(),
            queue: BucketQueue::new(),
            leaves: BucketQueue::new(),
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
                let leaf_cost = node_cost(egraph, node, |_| unreachable!());
                self.leaves.push(td_cost + leaf_cost, (node_idx, leaf_cost));
            } else {
                self.queue.push(td_cost + node.cost, node_idx);
            }
        }
    }

    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: u32, td_cost: Cost) {
        match self.eqc_state.get_mut(eqc) {
            Some(state) => match &mut state.status {
                EqcStatus::Parents(parents) => parents.push(node),
                EqcStatus::Resolved(_) => unreachable!(),
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
        self.node_delay.insert(node, n.children.len() as u32);
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_eqc(target, zero, Vec::new());
        while let Some((node, td_cost)) = self.queue.pop_with_cost() {
            self.visit_children(node, td_cost);
        }
    }
}

struct TwoPhaseAStarBottomUpExtractor<'a> {
    egraph: &'a EGraph,
    eqc_state: FxHashMap<&'a ClassId, EqcState>,
    node_delay: FxHashMap<u32, u32>,
    queue: BucketQueue<(u32, Cost)>,
    eqc_min: IndexMap<ClassId, NodeId>,
}

impl<'a> TwoPhaseAStarBottomUpExtractor<'a> {
    fn init(top_down: TwoPhaseAStarTopDownExtractor<'a>) -> TwoPhaseAStarBottomUpExtractor<'a> {
        TwoPhaseAStarBottomUpExtractor {
            egraph: top_down.egraph,
            eqc_state: top_down.eqc_state,
            node_delay: top_down.node_delay,
            queue: top_down.leaves,
            eqc_min: IndexMap::new(),
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
                self.queue.push(
                    parent_eqc_state.td_cost + bottom_up_cost,
                    (parent, bottom_up_cost),
                );
            }
        }
    }

    fn run(&mut self, target: &ClassId) {
        let egraph = self.egraph;
        while let Some((node, resolved_cost)) = self.queue.pop() {
            let (nid, n) = egraph.nodes.get_index(node as usize).unwrap();
            let eqc = &n.eclass;
            let state = self.eqc_state.get_mut(eqc).expect(BAD_PATH);
            if matches!(state.status, EqcStatus::Resolved(_)) {
                continue;
            }
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
