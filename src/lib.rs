//! An impl of a number of common CRDTs.
//!
//! This is a WIP, so expect that it might change in incompatible ways.

#![experimental]
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
