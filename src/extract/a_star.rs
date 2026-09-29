use std::mem;

use rustc_hash::{FxHashMap, FxHashSet};

use super::*;

#[derive(PartialEq, Eq)]
enum Action<'a> {
    Visit(&'a NodeId),
    Assign(&'a ClassId, &'a NodeId)
}

// The state associated with an e-node over the course of the extraction. As an e-node's bottom-up
// cost only becomes known once all of its child e-classes have been assigned a minimal e-node, the
// two states are mutually exclusive and can therefore share a single map.
enum NodeState {
    // The number of child e-classes which have not yet been assigned a minimal e-node.
    Delay(usize),
    // The bottom-up cost of the e-node, which is set when its assignment-action is enqueued.
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
    // The e-nodes which have this e-class as a child, which are accumulated while the e-class is
    // unresolved.
    Parents(Vec<&'a NodeId>),
    // The bottom-up cost of the e-class' minimal e-node, which is set when the e-class is assigned
    // one.
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

struct AStarExt<'a> {
    egraph:     &'a EGraph,
    queue:      PrioQueue<Action<'a>, Cost>,
    node_state: FxHashMap<&'a NodeId, NodeState>,
    eqc_state:  FxHashMap<&'a ClassId, EqcState<'a>>,
    td_cost:    FxHashMap<&'a ClassId, Cost>,
    eqc_min:    IndexMap<ClassId, NodeId>
}

// Like `ExtractionResult::node_sum_cost`.
fn node_cost(egraph: &EGraph, node: &NodeId, child_costs: &[Cost]) -> Cost {
    egraph[node].cost + child_costs.iter().sum::<Cost>()
}

// The costs of the minimal e-nodes of `node`'s child e-classes, which requires all of them to be
// resolved. This is a free function, so that it can be called while `AStarExt::node_state` is
// mutably borrowed.
fn node_child_costs<'a>(
    egraph: &'a EGraph, eqc_state: &FxHashMap<&'a ClassId, EqcState<'a>>, node: &'a NodeId
) -> Vec<Cost> {
    egraph[node].children.iter().map(|child| {
        eqc_state[egraph.nid_to_cid(child)].cost()
    }).collect()
}

impl<'a> AStarExt<'a> {
    fn new(egraph: &'a EGraph) -> AStarExt<'a> {
        AStarExt {
            egraph,
            queue:      PrioQueue::new(),
            node_state: Default::default(),
            eqc_state:  Default::default(),
            td_cost:    Default::default(),
            eqc_min:    Default::default()
        }
    }

    // Determines whether the e-class has already been assigned a minimal e-node.
    fn is_resolved_eqc(&self, eqc: &ClassId) -> bool {
        matches!(self.eqc_state.get(eqc), Some(EqcState::Cost(_)))
    }

    fn set_node_delay(&mut self, node: &'a NodeId, delay: usize) {
        self.node_state.insert(node, NodeState::Delay(delay));
    }

    fn set_td_cost(&mut self, eqc: &'a ClassId, cost: Cost) {
        self.td_cost.insert(eqc, cost);
    }

    // Determines whether a given e-class has already been enqueued via `enqueue_visit_eqc`. As
    // `enqueue_visit_eqc` always sets `td_cost` for the given e-class, we use membership in
    // this map as the indicator.
    fn is_enqueued_eqc(&self, eqc: &ClassId) -> bool {
        self.td_cost.contains_key(eqc)
    }

    fn add_eqc_parent(&mut self, eqc: &'a ClassId, node: &'a NodeId) {
        let state = self.eqc_state.entry(eqc).or_insert_with(|| EqcState::Parents(Vec::new()));
        state.parents_mut().push(node);
    }

    fn dequeue(&mut self) -> Option<Action<'a>> {
        self.queue.pop().map(|(action, _)| action)
    }

    fn enqueue(&mut self, action: Action<'a>, merit: Cost) {
        self.queue.insert(action, merit);
    }

    fn enqueue_visit_node(&mut self, node: &'a NodeId, td_cost: Cost) {
        let top_down_cost = td_cost + self.egraph[node].cost;
        let action = Action::Visit(node);
        self.enqueue(action, top_down_cost);
    }

    fn enqueue_assignment(&mut self, node: &'a NodeId, child_costs: Vec<Cost>) {
        let bottom_up_cost = node_cost(self.egraph, node, &child_costs);
        self.node_state.insert(node, NodeState::Cost(bottom_up_cost));
        let eqc = self.egraph.nid_to_cid(node);
        let top_down_cost = self.td_cost[eqc];
        let merit = top_down_cost + bottom_up_cost;
        let action = Action::Assign(eqc, node);
        self.enqueue(action, merit);
    }

    fn enqueue_visit_eqc(&mut self, eqc: &'a ClassId, td_cost: Cost) {
        if !self.is_enqueued_eqc(eqc) {
            // This has to happen before the loop, as `enqueue_assignment` reads the `td_cost` of
            // `eqc`.
            self.set_td_cost(eqc, td_cost);
            let egraph = self.egraph;
            for node in &egraph.classes()[eqc].nodes {
                // Visiting a leaf would do nothing but enqueue its assignment-action, so we enqueue
                // that directly and never enqueue a visit-action for a leaf in the first place.
                if egraph[node].children.is_empty() {
                    self.enqueue_assignment(node, vec![]);
                } else {
                    self.enqueue_visit_node(node, td_cost);
                }
            }
        }
    }

    fn visit_node(&mut self, node: &'a NodeId) {
        let mut delayed_eqcs: FxHashSet<&'a ClassId> = FxHashSet::default();
        let eqc = self.egraph.nid_to_cid(node);
        let td_cost = self.td_cost[eqc] + self.egraph[node].cost;
        for child in &self.egraph[node].children {
            let child = self.egraph.nid_to_cid(child);
            // (1) If the child `eqc` is already resolved, there's nothing to do (its cost is
            //     fetched after the loop, if needed).
            // (2) If `eqc` is not resolved, set the parent-child relationship, and enqueue `eqc`
            //     (the node delay is set after the loop).
            if self.is_resolved_eqc(child) { continue }
            // It is important that we do not register the same e-class as delayed multiple times,
            // as this would break the delay count.
            if !delayed_eqcs.contains(child) {
                delayed_eqcs.insert(child);
                self.add_eqc_parent(child, node);
                self.enqueue_visit_eqc(child, td_cost);
            }
        }
        if delayed_eqcs.is_empty() {
            // If all of `node`'s children are resolved, we can add an assignment-action for it
            // immediately.
            let child_costs = node_child_costs(self.egraph, &self.eqc_state, node);
            self.enqueue_assignment(node, child_costs);
        } else {
            self.set_node_delay(node, delayed_eqcs.len());
        }
    }

    // Notifies the parents of a just-resolved e-class that one of their child e-classes has been
    // resolved: a parent which was only waiting for this one e-class is now fully resolved itself
    // and gets an assignment-action enqueued, and any other parent merely has its delay decremented.
    fn update_parents(&mut self, parents: Vec<&'a NodeId>) {
        let egraph = self.egraph;
        // The remaining fields are destructured, so that the borrow checker sees the accesses to
        // the maps below as disjoint. This allows `node_state` to be updated in place, while the
        // other maps are being read.
        let Self { queue, node_state, eqc_state, td_cost, .. } = self;
        for parent in parents {
            let state = node_state.get_mut(parent).expect(BAD_PATH);
            match state {
                NodeState::Delay(1) => {
                    // If the parent's e-class has already been resolved (by another of its
                    // e-nodes), the parent cannot become its minimal e-node, so there's no need to
                    // enqueue an assignment-action for it. Its state then simply stays at
                    // `Delay(1)`, as it is never looked at again.
                    let parent_eqc = egraph.nid_to_cid(parent);
                    if matches!(eqc_state.get(parent_eqc), Some(EqcState::Cost(_))) { continue }
                    let child_costs = node_child_costs(egraph, eqc_state, parent);
                    let bottom_up_cost = node_cost(egraph, parent, &child_costs);
                    let merit = td_cost[parent_eqc] + bottom_up_cost;
                    *state = NodeState::Cost(bottom_up_cost);
                    queue.insert(Action::Assign(parent_eqc, parent), merit);
                },
                NodeState::Delay(delay) if *delay > 1 => *delay -= 1,
                _ => panic!("{}", BAD_PATH)
            }
        }
    }

    fn run(&mut self, target: &'a ClassId) {
        let zero = NotNan::new(0.0).unwrap();
        self.enqueue_visit_eqc(target, zero);
        while let Some(action) = self.dequeue() {
            match action {
                Action::Visit(node) => {
                    // If `node`'s e-class has already been resolved (by another of its e-nodes),
                    // `node` cannot become its minimal e-node, so there's no need to explore below
                    // `node`. This does not affect which e-nodes get assigned: the minimality of an
                    // assignment only depends on the visits of e-nodes of *unresolved* e-classes.
                    // It does mean that `td_cost` is not necessarily the cost of the cheapest path
                    // to an e-class, as that path may run through an e-node skipped here.
                    if self.is_resolved_eqc(self.egraph.nid_to_cid(node)) { continue }
                    // Only branch nodes get visit-actions (see `enqueue_visit_eqc`).
                    self.visit_node(node)
                },
                Action::Assign(eqc, node) => {
                    if eqc == target {
                        self.eqc_min.insert(eqc.clone(), node.clone());
                        break
                    }
                    // The only e-class which may not have a parent is the target, which we already
                    // handled above.
                    let state = self.eqc_state.get_mut(eqc).expect(BAD_PATH);
                    // Taking the parents out of the state is what marks the e-class as resolved, so
                    // an e-class which is already in the `Cost` state has been assigned a minimal
                    // e-node before. Thus, no separate map is needed to detect this. If the node's
                    // e-class is already resolved, there's nothing to do.
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
    }
}

const BAD_PATH: &str = "Reached bad path in `AStarExt::run`.";

pub struct AStarExtractor;

impl Extractor for AStarExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // TODO: We currently assume there to be only a single root class from which we extract. Add
        //       a field to the `ExtractorDetail` in `main.rs` where this can be declared.
        assert_eq!(roots.len(), 1);
        let target = &roots[0];
        let mut ext = AStarExt::new(egraph);
        ext.run(target);
        ExtractionResult { choices: ext.eqc_min }
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
