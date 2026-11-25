use std::cmp::{Ord, Ordering};
use std::collections::{BTreeMap, BinaryHeap, VecDeque};

use rustc_hash::FxHashSet;

use super::{ClassId, Cost, EGraph, Node};

// Takes the `Ord` from U, but reverses it.
#[derive(PartialEq, Eq, Debug)]
struct WithOrdRev<T: Eq, U: Ord>(pub T, pub U);

impl<T: Eq, U: Ord> Ord for WithOrdRev<T, U> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // It's the other way around, because we want a min-heap!
        other.1.cmp(&self.1)
    }
}

impl<T: Eq, U: Ord> PartialOrd for WithOrdRev<T, U> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) struct PrioQueue<T: Eq, C: Ord>(BinaryHeap<WithOrdRev<T, C>>);

impl<T: Eq, C: Ord> PrioQueue<T, C> {
    #[inline]
    pub fn new() -> Self {
        PrioQueue(BinaryHeap::new())
    }

    #[inline]
    pub fn pop(&mut self) -> Option<(T, C)> {
        self.0.pop().map(|WithOrdRev(t, c)| (t, c))
    }

    #[inline]
    pub fn insert(&mut self, t: T, c: C) {
        self.0.push(WithOrdRev(t, c));
    }
}

/// A bucketed min-priority queue backed by `BTreeMap<Cost, Vec<T>>`.
pub(crate) struct BucketQueue<T>(BTreeMap<Cost, Vec<T>>);

impl<T> Default for BucketQueue<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BucketQueue<T> {
    #[inline]
    pub fn new() -> Self {
        BucketQueue(BTreeMap::new())
    }

    #[inline]
    pub fn push(&mut self, cost: Cost, item: T) {
        self.0.entry(cost).or_default().push(item);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let mut entry = self.0.first_entry()?;
        let item = entry.get_mut().pop().unwrap();
        if entry.get().is_empty() {
            entry.remove();
        }
        Some(item)
    }

    #[inline]
    pub fn pop_with_cost(&mut self) -> Option<(T, Cost)> {
        let mut entry = self.0.first_entry()?;
        let cost = *entry.key();
        let item = entry.get_mut().pop().unwrap();
        if entry.get().is_empty() {
            entry.remove();
        }
        Some((item, cost))
    }
}

/// Evaluates the bottom-up cost of `node` given a lookup for its child e-class costs.
/// Centralized here so all optimized extractors share a single cost-function hook.
#[inline]
pub(crate) fn node_cost(
    egraph: &EGraph,
    node: &Node,
    mut child_cost: impl FnMut(&ClassId) -> Cost,
) -> Cost {
    node.cost
        + node
            .children
            .iter()
            .map(|c| child_cost(egraph.nid_to_cid(c)))
            .sum::<Cost>()
}

/// Whether an e-class has been resolved yet. As an e-class' parents are only needed at the moment
/// it is assigned a minimal e-node, the two states are mutually exclusive.
pub(crate) enum EqcStatus {
    Parents(Vec<u32>),
    Resolved(Cost),
}

impl EqcStatus {
    #[inline]
    pub fn resolved_cost(&self) -> Cost {
        match self {
            EqcStatus::Resolved(cost) => *cost,
            EqcStatus::Parents(_) => panic!("Accessed the cost of an unresolved e-class."),
        }
    }
}

/** A data structure to maintain a queue of unique elements.

Notably, insert/pop operations have O(1) expected amortized runtime complexity.

Thanks @Bastacyclop for the implementation!
*/
#[derive(Clone)]
pub(crate) struct UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    set: FxHashSet<T>, // hashbrown::
    queue: VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: Default::default(),
            queue: VecDeque::new(),
        }
    }
}

impl<T> UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter.into_iter() {
            self.insert(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        res.as_ref().map(|t| self.set.remove(t));
        res
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}
