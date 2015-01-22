//! An impl of a number of common CRDTs.
//!
//! This is a WIP, so expect that it might change in incompatible ways.

#![experimental]
#![allow(unstable)]

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
pub trait Operation<S>: Clone {
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
#[derive(Show, Clone)]
pub struct StateBasedReplica<M, S>
    where M: Fn(S, S) -> S;
#[derive(Show, Clone)]
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
#[derive(Show, Clone)]
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
