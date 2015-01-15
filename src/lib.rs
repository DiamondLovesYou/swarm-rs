//! An impl of a number of common CRDTs.
//!
//! This is a WIP, so expect that it might change in incompatible ways.

#![experimental]

use std::borrow::BorrowFrom;
use std::collections::BTreeSet;
use std::default::Default;

pub struct State<T: Default + Send, Merger: Fn(T, T) -> T> {
    state:  Option<T>,
    merger: Merger,
}
impl<T: Default + Send, Merger: Fn(T, T) -> T> State<T, Merger> {
    #[stable]
    pub fn new(merger: Merger) -> State<T, Merger> {
        State {
            state: Some(Default::default()),
            merger: merger,
        }
    }

    pub fn query<F, U>(&self, f: F) -> U where F: FnOnce(&T) -> U {
        assert!(self.state.is_some());
        f(self.state.as_ref().unwrap())
    }

    // Mutate the state locally. Returns a ref for downstream replicas.
    pub fn update<F>(&mut self, f: F) -> &T where F: FnOnce(&mut T) {
        assert!(self.state.is_some());
        f(self.state.as_mut().unwrap());
        self.state.as_ref().unwrap()
    }
    // Merge the current state with an external state.
    pub fn merge(&mut self, right_state: T) {
        assert!(self.state.is_some());
        let left_state = self.state.take().unwrap();
        let new_state = (self.merger)(left_state, right_state);
        self.state = Some(new_state);
    }
}


type GOSSMerger<Atom> = fn(BTreeSet<Atom>, BTreeSet<Atom>) -> BTreeSet<Atom>;
pub struct GrowOnlySetState<Atom: Ord + Send>(State<BTreeSet<Atom>, GOSSMerger<Atom>>);

impl<Atom> GrowOnlySetState<Atom> where Atom: Ord + Send {
    pub fn new() -> GrowOnlySetState<Atom> {

        fn merger<Atom>(mut left: BTreeSet<Atom>, right: BTreeSet<Atom>) -> BTreeSet<Atom>
            where Atom: Ord + Send
        {
            left.extend(right.into_iter());
            left
        }

        GrowOnlySetState(State::new(merger))
    }

    fn inner_imm(&self) -> &State<BTreeSet<Atom>, GOSSMerger<Atom>> {
        let &GrowOnlySetState(ref inner) = self;
        inner
    }
    fn inner_mut(&mut self) -> &mut State<BTreeSet<Atom>, GOSSMerger<Atom>> {
        let &mut GrowOnlySetState(ref mut inner) = self;
        inner
    }

    pub fn add(&mut self, v: Atom) -> &BTreeSet<Atom> {
        self.inner_mut()
            .update(move |: state: &mut BTreeSet<Atom>| {
                state.insert(v);
            })
    }

    pub fn lookup<T: ?Sized>(&self, v: &T) -> bool where T: BorrowFrom<Atom>, T: Ord {
        self.inner_imm()
            .query(move |&: state: &BTreeSet<Atom>| {
                state.contains(v)
            })
    }

    pub fn merge(&mut self, right: BTreeSet<Atom>) {
        self.inner_mut()
            .merge(right)
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn grow_only_set_state() {
        let mut left = GrowOnlySetState::new();
        let left_state = left.add(1u64)
            .clone();

        let mut right = GrowOnlySetState::new();
        let right_state = right.add(2u64)
            .clone();

        left.merge(right_state);
        assert!(left.lookup(&1u64));
        assert!(left.lookup(&2u64));

        right.merge(left_state);
        assert!(right.lookup(&1u64));
        assert!(right.lookup(&2u64));
    }

}
