//! An impl of a number of common CRDTs.
//!
//! This is a WIP, so expect that it might change in incompatible ways.

#![experimental]
#![allow(unstable)]

use std::borrow::BorrowFrom;
use std::collections::BTreeSet;
use std::default::Default;
use std::error;

pub mod set;

pub trait Log<Op> {
    type Error;
    // Once this function returns, it is expected that all changes to
    // the log have been made durable. Additionally, this crate
    // assumes in order delivery, ie operations will be delivered in
    // the same order as this API provides them.
    fn apply_downstream(&mut self, op: Op) -> Result<(), <Self as Log<Op>>::Error>;
    // Add to the local log, but don't try to apply downstream.
    // As with Log::apply_downstream, Log::add_to_log must make log changes durable.
    fn add_to_log(&mut self, op: Op) -> Result<(), <Self as Log<Op>>::Error>;
}
pub trait Operation<S> {
    type Error;
    fn apply_to_state(&self,
                      state: &mut S) -> Result<(), <Self as Operation<S>>::Error>; 
}
#[derive(Show)]
pub enum UpdateError<L, O>
    where L: error::Error, O: error::Error,
{
    Log(L),
    Op(O),
}
impl<L, O> error::Error for UpdateError<L, O>
    where L: error::Error, O: error::Error,
{
    fn description(&self) -> &str {
        match self {
            &UpdateError::Log(ref e) => e.description(),
            &UpdateError::Op(ref e) => e.description(),
        }
    }
    fn detail(&self) -> Option<String> {
        match self {
            &UpdateError::Log(ref e) => e.detail(),
            &UpdateError::Op(ref e) => e.detail(),
        }
    }
    fn cause(&self) -> Option<&error::Error> {
        match self {
            &UpdateError::Log(ref e) => Some(e as &error::Error),
            &UpdateError::Op(ref e) => Some(e as &error::Error),
        }
    }
}

pub struct StateBasedReplica<M, S>
    where M: Fn(S, S) -> S;
pub struct OpBasedReplica<O, S>
    where O: Operation<S>;

pub trait ReplicationKind: Default {}
impl<M, S> ReplicationKind for StateBasedReplica<M, S>
    where M: Fn(S, S) -> S, {}
impl<O, S> ReplicationKind for OpBasedReplica<O, S>
    where O: Operation<S>, {}

impl<M, S> Default for StateBasedReplica<M, S>
    where M: Fn(S, S) -> S,
{
    fn default() -> StateBasedReplica<M, S> {
        StateBasedReplica
    }
}
impl<O, S> Default for OpBasedReplica<O, S>
    where O: Operation<S>, {
        fn default() -> OpBasedReplica<O, S> {
            OpBasedReplica
        }
}

// A local replica of a state. Can be Op or State based.
#[derive(Show)]
pub struct Replica<S: Default, T>
    where T: ReplicationKind,
{
    state: Option<S>,
    _type: T,
}

impl<S: Default, T: ReplicationKind> Default for Replica<S, T> {
    fn default() -> Replica<S, T> {
        Replica::new()
    }
}

impl<S, O> Replica<S, OpBasedReplica<O, S>>
    where O: Operation<S>,
          S: Default,
          <O as Operation<S>>::Error: error::Error,
{
    pub fn update(&mut self, op: &O) -> Result<(), <O as Operation<S>>::Error> {
        assert!(self.state.is_some());
        op.apply_to_state(self.state.as_mut().unwrap())
    }
}
impl<S: Default, T: ReplicationKind> Replica<S, T> {
    pub fn new() -> Replica<S, T> {
        Replica {
            state: Some(Default::default()),
            _type: Default::default(),
        }
    }
    pub fn query<'a, F, U>(&'a self, f: F) -> U where F: FnOnce(&'a S) -> U {
        f(self.state.as_ref().unwrap())
    }
}

pub trait OpLogStatePair<L, S> {
    fn log_imm(&self) -> &L;
    fn log_mut(&mut self) -> &mut L;
    fn state_imm(&self) -> &S;
}

impl<S: Default + Send, M: Fn(S, S) -> S> Replica<S, StateBasedReplica<M, S>> {
    // Mutate the state locally. Returns a ref for downstream replicas.
    pub fn mutate<F>(&mut self, f: F) -> &S where F: FnOnce(&mut S) {
        assert!(self.state.is_some());
        f(self.state.as_mut().unwrap());
        self.state.as_ref().unwrap()
    }
    // Merge the current state with an external state.
    pub fn merge(&mut self, right_state: S, merger: M) {
        assert!(self.state.is_some());
        let left_state = self.state.take().unwrap();
        let new_state = merger(left_state, right_state);
        self.state = Some(new_state);
    }
}


type GOSSMerger<Atom> = fn(BTreeSet<Atom>, BTreeSet<Atom>) -> BTreeSet<Atom>;
type GOSSReplica<A: Ord + Send> =
    Replica<BTreeSet<A>, StateBasedReplica<GOSSMerger<A>, BTreeSet<A>>>;
pub struct GrowOnlySetState<Atom: Ord + Send>(GOSSReplica<Atom>);

fn goss_merger<Atom>(mut left: BTreeSet<Atom>, right: BTreeSet<Atom>) -> BTreeSet<Atom>
    where Atom: Ord + Send
{
    left.extend(right.into_iter());
    left
}

impl<Atom> GrowOnlySetState<Atom> where Atom: Ord + Send {
    pub fn new() -> GrowOnlySetState<Atom> {
        GrowOnlySetState(Replica::new())
    }

    fn inner_imm(&self) -> &GOSSReplica<Atom> {
        let &GrowOnlySetState(ref inner) = self;
        inner
    }
    fn inner_mut(&mut self) -> &mut GOSSReplica<Atom> {
        let &mut GrowOnlySetState(ref mut inner) = self;
        inner
    }

    pub fn add(&mut self, v: Atom) -> &BTreeSet<Atom> {
        self.inner_mut()
            .mutate(move |: state: &mut BTreeSet<Atom>| {
                state.insert(v);
            })
    }

    pub fn lookup<T: ?Sized>(&self, v: &T) -> bool where T: BorrowFrom<Atom>, T: Ord {
        self.inner_imm()
            .query(move |&: state: &BTreeSet<Atom>| {
                state.contains(v)
            })
    }

    pub fn len(&self) -> usize {
        self.inner_imm()
            .query(move |: state: &BTreeSet<Atom>| {
                state.len()
            })
    }

    pub fn merge(&mut self, right: BTreeSet<Atom>) {
        self.inner_mut()
            .merge(right, goss_merger)
    }
}


#[cfg(test)]
mod test {
    mod grow_only_set_state {
        use super::super::*;
        #[test]
        fn merge() {
            let mut left = GrowOnlySetState::new();
            let left_state = left.add(1u64)
                .clone();

            let mut right = GrowOnlySetState::new();
            let right_state = right.add(2u64)
                .clone();

            left.merge(right_state);
            assert!(left.lookup(&1u64));
            assert!(left.lookup(&2u64));
            assert_eq!(left.len(), 2);

            right.merge(left_state);
            assert!(right.lookup(&1u64));
            assert!(right.lookup(&2u64));
            assert_eq!(right.len(), 2);
        }

        #[test]
        fn concurrent_add() {
            let mut left = GrowOnlySetState::new();
            let left_state = left.add(1u64)
                .clone();

            let mut right = GrowOnlySetState::new();
            right.add(1u64);
            right.merge(left_state);
            assert!(right.lookup(&1u64));
            assert_eq!(right.len(), 1);
        }
    }
}
