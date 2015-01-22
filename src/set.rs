use super::{Operation, Replica, UpdateError};
use super::{OpBasedReplica, StateBasedReplica};
use std::borrow::BorrowFrom;
use std::collections::{btree_map, BTreeMap, BTreeSet};
use std::error::Error;
use std::iter;

#[derive(Clone, Show)]
pub enum Op<Atom> {
    Add(Atom),
    Remove(Atom),
}
impl<Atom> Op<Atom> {
    pub fn as_ref(&self) -> &Atom {
        match self {
            &Op::Add(ref v) | &Op::Remove(ref v) => v,
        }
    }
}

// TODO partial order for Op. The order should represent a requisite delivery order.

#[derive(Copy, Clone, Show)]
pub enum OpError {
    ValueAlreadyPresent,
    ValueNotPresent,
    ValueTombstoned,
}
impl Error for OpError {
    fn description(&self) -> &str {
        match self {
            &OpError::ValueAlreadyPresent => "that value is already present in the set",
            &OpError::ValueNotPresent => "that value isn't present within the set",
            &OpError::ValueTombstoned => "that value is marked with a tombstone; \
                                          it is impossible to re-add it to the set",
        }
    }
}

impl<A: Clone + Ord> Operation<UniqueState<A>> for Op<A> {
    type Error = OpError;
    fn apply_to_state(&self,
                      state: &mut UniqueState<A>) -> Result<(), OpError> {
        match self {
            &Op::Add(ref v) => {
                match state.get(v) {
                    Some(has_tombstone) if !(*has_tombstone) => {
                        return Err(OpError::ValueAlreadyPresent);
                    },
                    Some(has_tombstone) if *has_tombstone => {
                        return Err(OpError::ValueTombstoned);
                    }
                    Some(_) => unreachable!(),
                    None => {},
                }
                assert!(state.insert(v.clone(), false).is_none());
            }
            &Op::Remove(ref v) => {
                match state.get_mut(v) {
                    Some(ref mut has_tombstone) if !(**has_tombstone) => {
                        **has_tombstone = true;
                        return Ok(());
                    },
                    Some(ref has_tombstone) if **has_tombstone => {
                        return Ok(());
                    }
                    Some(_) => unreachable!(),
                    None => {
                        return Err(OpError::ValueNotPresent);
                    },
                }
                assert!(state.insert(v.clone(), true).is_none());
            }
        }
        return Ok(());
    }
}

// If the mapped value is true, it is a tombstone.
pub type UniqueState<A> = BTreeMap<A, bool>;
pub type UniqueReplica<A> =
    Replica<UniqueState<A>, OpBasedReplica<Op<A>, UniqueState<A>>>;
// An Op-based CRDT.
#[derive(Clone, Show)]
pub struct Unique<L, A>
    where L: super::Log<Op<A>>, A: Ord + Clone,
{
    state: UniqueReplica<A>,
    log:   L,
}
pub type UniqueError<L, A> = UpdateError<<L as super::Log<Op<A>>>::Error, OpError>;

impl<LHSL, RHSL, A> PartialEq<Unique<RHSL, A>> for Unique<LHSL, A>
    where LHSL: super::Log<Op<A>>, RHSL: super::Log<Op<A>>,
          A: PartialEq + Ord + Clone,
{
    fn eq(&self, rhs: &Unique<RHSL, A>) -> bool {
        self.state.query(move |&: lhs| {
            lhs.eq(rhs.state.query(move |: state| state ))
        })
    }
}

impl<Log, Atom> Unique<Log, Atom>
    where Log: super::Log<Op<Atom>>,
          Atom: Ord + Clone,
          <Log as super::Log<Op<Atom>>>::Error: Error,
          <Op<Atom> as Operation<UniqueState<Atom>>>::Error: Error,
{
    pub fn new_with_log(log: Log) -> Unique<Log, Atom> {
        Unique {
            state: Replica::new(),
            log: log,
        }
    }

    pub fn lookup(&self, v: &Atom) -> bool {
        self.state
            .query(move |&: state: &UniqueState<Atom>| {
                match state.get(v) {
                    Some(tomb) => !(*tomb),
                    None => false,
                }
            })
    }

    // Return a forward only iterator over the none tombstoned elements in this set.
    pub fn iter(&self) -> iter::FilterMap<(&Atom, &bool), &Atom, btree_map::Iter<Atom, bool>, for<'a> fn((&'a Atom, &bool)) -> Option<&'a Atom>> {
        fn filter_map<'a, A>((a, &tomb): (&'a A, &bool)) -> Option<&'a A> {
            if !tomb { Some(a) }
            else     { None    }
        }

        self.state
            .query(|state| {
                state.iter()
                    .filter_map(filter_map as for<'a> fn((&'a Atom, &bool)) -> Option<&'a Atom>)
            })
    }

    pub fn add(&mut self, v: Atom) -> Result<(), UniqueError<Log, Atom>> {
        let op = Op::Add(v);
        let res = self.state.update(&op);
        match res {
            Ok(()) => self.log
                .apply_downstream(op)
                .map_err(|err| UpdateError::Log(err) ),
            Err(err) => Err(UpdateError::Op(err)),
        }
    }
    pub fn remove(&mut self, v: Atom) -> Result<(), UniqueError<Log, Atom>> {
        let op = Op::Remove(v);
        let res = self.state.update(&op);
        match res {
            Ok(()) => self.log
                .apply_downstream(op)
                .map_err(|err| UpdateError::Log(err) ),
            Err(err) => Err(UpdateError::Op(err)),
        }
    }

    // Note this is not what you would expect, it will also include dead set members.
    pub fn len(&self) -> usize {
        self.state.query(move |: state: &UniqueState<Atom>| state.len() )
    }

    // Deliver an upstream operation to this replica. Returns the
    // number of operations successfully processed.
    pub fn deliver(&mut self, ops: &[Op<Atom>]) -> (usize,
                                                    Result<(), UniqueError<Log, Atom>>)
    {
        let mut successful: usize = 0;
        for op in ops.iter() {
            let op = op.clone();
            let res = self.state.update(&op);
            let res = match res {
                Ok(()) | Err(OpError::ValueAlreadyPresent) => {
                    self.log
                        .add_to_log(op)
                        .map_err(|err| UpdateError::Log(err) )
                },
                Err(err) => Err(UpdateError::Op(err)),
            };
            if res.is_ok() {
                successful = successful + 1;
            } else {
                return (successful, res);
            }
        }
        return (successful, Ok(()));
    }

    pub fn log_imm<'a>(&'a self) -> &'a Log { &self.log }
    pub fn log_mut<'a>(&'a mut self) -> &'a mut Log { &mut self.log }
}

impl<Atom: Clone + Ord> Operation<BTreeMap<Atom, (u64, u64)>> for Op<Atom> {
    type Error = OpError;
    fn apply_to_state(&self,
                      state: &mut BTreeMap<Atom, (u64, u64)>) -> Result<(), OpError> {
        match self {
            &Op::Add(ref v) => {
                match state.get_mut(v) {
                    Some(&mut (ref mut added, _)) => {
                        *added = *added + 1;
                        return Ok(());
                    },
                    None => {},
                }
                assert!(state.insert(v.clone(), (1, 0)).is_none());
            }
            &Op::Remove(ref v) => {
                match state.get_mut(v) {
                    Some(&mut (_, ref mut removed)) => {
                        *removed = *removed + 1;
                        return Ok(());
                    },
                    None => {},
                }
                assert!(state.insert(v.clone(), (0, 1)).is_none());
            }
        }
        return Ok(());
    }
}

pub type PNState<A> = BTreeMap<A, (u64, u64)>;
pub type PNReplica<A> =
    Replica<PNState<A>, OpBasedReplica<Op<A>, PNState<A>>>;
// A multi value set.
#[derive(Clone, Show)]
pub struct PN<L, A>
    where L: super::Log<Op<A>>,
          A: Ord + Clone,
{
    state: PNReplica<A>,
    log: L,
}

impl<LHSL, RHSL, A> PartialEq<PN<RHSL, A>> for PN<LHSL, A>
    where LHSL: super::Log<Op<A>>, RHSL: super::Log<Op<A>>,
          A: PartialEq + Ord + Clone,
{
    fn eq(&self, rhs: &PN<RHSL, A>) -> bool {
        self.state.query(move |&: lhs| {
            lhs.eq(rhs.state.query(move |: state| state ))
        })
    }
}
pub type PNError<L, A> = UpdateError<<L as super::Log<Op<A>>>::Error, OpError>;

impl<L, A> PN<L, A>
    where L: super::Log<Op<A>>,
          A: Ord + Clone,
          <L as super::Log<Op<A>>>::Error: Error,
          <Op<A> as Operation<PNState<A>>>::Error: Error,
{
    pub fn new_with_log(log: L) -> PN<L, A> {
        PN {
            state: super::Replica::new(),
            log: log,
        }
    }

    pub fn log_imm(&self) -> &L { &self.log }
    pub fn log_mut(&mut self) -> &mut L { &mut self.log }

    pub fn lookup(&self, k: &A) -> bool {
        self.count(k) > 0
    }
    pub fn count(&self, k: &A) -> i64 {
        self.state
            .query(move |&: state| {
                match state.get(k) {
                    None => 0,
                    Some(&(ref added, ref removed)) => *added as i64 - *removed as i64,
                }
            })
    }

    // Return a forward only iterator over the none tombstoned elements in this set.
    pub fn iter(&self) -> iter::FilterMap<(&A, &(u64, u64)), &A, btree_map::Iter<A, (u64, u64)>, for<'a> fn((&'a A, &(u64, u64))) -> Option<&'a A>> {
        fn filter_map<'a, A>((a, &(added, removed)): (&'a A, &(u64, u64))) -> Option<&'a A> {
            if added > removed { Some(a) }
            else               { None    }
        }

        self.state
            .query(|state| {
                state.iter()
                    .filter_map(filter_map as for<'a> fn((&'a A, &(u64, u64))) -> Option<&'a A>)
            })
    }

    fn update(&mut self, op: Op<A>) -> Result<(), PNError<L, A>> {
        let res = self.state.update(&op);
        match res {
            Ok(()) => self.log
                .apply_downstream(op)
                .map_err(|err| UpdateError::Log(err) ),
            Err(err) => Err(UpdateError::Op(err)),
        }
    }

    pub fn add(&mut self, v: A) -> Result<(), PNError<L, A>> {
        let op = Op::Add(v);
        self.update(op)
    }
    pub fn remove(&mut self, v: A) -> Result<(), PNError<L, A>> {
        let op = Op::Remove(v);
        self.update(op)
    }

    // Note this count includes elements that are technically removed.
    pub fn len(&self) -> usize {
        self.state
            .query(move |: state| {
                state.len()
            })
    }

    pub fn deliver(&mut self,
                   ops: &[Op<A>]) -> (usize, Result<(), PNError<L, A>>) {
        let mut successful: usize = 0;
        for op in ops.iter() {
            let res = self.state.update(op);
            let res = match res {
                Ok(()) => self.log
                    .add_to_log(op.clone())
                    .map_err(|err| UpdateError::Log(err) ),
                Err(err) => Err(UpdateError::Op(err)),
            };
            if res.is_ok() {
                successful = successful + 1;
            } else {
                return (successful, res);
            }
        }
        return (successful, Ok(()));
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
#[allow(dead_code)]
mod test {
    use {Log, Operation, UpdateError};
    use super::*;
    use std::error::Error;

    // Our mock DumbLog can't fail:
    #[derive(Show)]
    pub struct NullError;
    impl Error for NullError {
        fn description(&self) -> &str { panic!("this error should never happen"); }
    }
    pub type UniqueSet = Unique<DumbLog<Op<u64>, UniqueState<u64>>, u64>;
    pub type TestPN = super::PN<DumbLog<Op<u64>, PNState<u64>>, u64>;
    #[derive(Default, Clone, Show)]
    pub struct DumbLog<O, S> where O: Operation<S> {
        log: Vec<O>,

        downstreamed: usize,
    }
    pub type TestDumbLog = DumbLog<Op<u64>, UniqueState<u64>>;
    macro_rules! impl_deliver {
        ($state:ty, $set:ty => $func_name:ident) => {
            impl DumbLog<Op<u64>, $state> {
                fn $func_name(&self,
                              start: usize,
                              end: Option<usize>,
                              to: &mut $set) -> (usize,
                                                 Result<(), UpdateError<NullError, OpError>>)
                {
                    let ops = self.log.as_slice()
                        .slice(start, end.unwrap_or(self.len()));
                    to.deliver(ops)
                }
            }
        }
    }
    impl_deliver! { UniqueState<u64>, UniqueSet => deliver_unique }
    impl_deliver! { PNState<u64>, TestPN => deliver_pn }
    impl<O, S> DumbLog<O, S> {
        fn len(&self) -> usize { self.log.len() }
        fn downstreamed(&self) -> usize {
            self.downstreamed
        }
        fn not_downstreamed(&self) -> usize {
            self.len() - self.downstreamed()
        }
    }
        
    macro_rules! new_dumb_log(
        () => {
            DumbLog {
                log: Vec::new(),
                downstreamed: 0,
            }
        }
    );

    macro_rules! impl_log_for_state {
        ($for_type:ty) => {
            impl Log<Op<u64>> for DumbLog<Op<u64>, $for_type> {
                type Error = NullError;
                fn apply_downstream(&mut self, op: Op<u64>) -> Result<(), NullError> {
                    // for tests we ignore the 'must be durable' stipulation.
                    self.downstreamed = self.downstreamed + 1;
                    self.add_to_log(op)
                }
                fn add_to_log(&mut self, op: Op<u64>) -> Result<(), NullError> {
                    self.log.push(op);
                    Ok(())
                }
            }
        }
    }
    impl_log_for_state! { UniqueState<u64> }
    impl_log_for_state! { PNState<u64> }

    pub fn new_unique() -> UniqueSet {
        Unique::new_with_log(new_dumb_log!())
    }

    pub fn new_pn() -> TestPN {
        PN::new_with_log(new_dumb_log!())
    }

    mod unique {
        use {UpdateError};
        use super::*;
        use super::super::OpError;
        #[test]
        fn op_log_len() {
            let mut local = new_unique();
            assert!(local.add(6u64).is_ok());
            assert_eq!(local.log_imm().len(), 1);
            assert!(local.remove(6u64).is_ok());
            assert_eq!(local.log_imm().len(), 2);
            assert_eq!(local.log_imm().downstreamed(), 2);
        }
        #[test]
        fn commute() {
            let mut right = new_unique();
            assert!(right.add(6u64).is_ok());
            assert!(right.remove(6u64).is_ok());
            let mut left = new_unique();
            right.log_imm().deliver_unique(0, None, &mut left);
            assert_eq!(left.len(), 1);
            assert!(!left.lookup(&6u64))
        }
        #[test]
        fn add_twice() {
            let mut right = new_unique();
            assert!(right.add(6u64).is_ok());
            assert!(match right.add(6u64) {
                Err(UpdateError::Op(OpError::ValueAlreadyPresent)) => true,
                _ => false,
            });
        }
        #[test]
        fn remove_first() {
            let mut right = new_unique();
            assert!(right.remove(6u64).is_err());
        }
        #[test]
        fn concurrent_adds_commute() {
            let mut right = new_unique();
            assert!(right.add(6u64).is_ok());
            assert!(right.add(5u64).is_ok());

            let mut left = new_unique();
            assert!(left.add(4u64).is_ok());
            assert!(left.add(3u64).is_ok());

            assert_eq!(left.log_imm().len(), 2);
            assert!(!left.lookup(&6u64));
            assert!(!left.lookup(&5u64));

            let left_log = left.log_imm().clone();
            right.log_imm().deliver_unique(0, None, &mut left);

            assert_eq!(left.log_imm().len(), 4);
            assert!(left.lookup(&6u64));
            assert!(left.lookup(&5u64));

            left_log.deliver_unique(0, None, &mut right);
            assert_eq!(right.log_imm().len(), 4);
            assert!(right.lookup(&4u64));
            assert!(right.lookup(&3u64));
        }
        #[test]
        fn concurrent_removes_commute() {
            let mut right = new_unique();
            assert!(right.add(6u64).is_ok());
            assert!(right.remove(6u64).is_ok());

            let mut left = new_unique();
            assert!(left.add(6u64).is_ok());

            assert!(left.lookup(&6u64));

            right.log_imm().deliver_unique(0, None, &mut left);

            assert!(!left.lookup(&6u64));
        }
        #[test]
        fn delivery_order_insensitive() {
            let mut right = new_unique();
            assert!(right.add(6u64).is_ok());

            let mut left = new_unique();
            assert!(left.add(6u64).is_ok());
            assert!(left.remove(6u64).is_ok());

            let left_log = left.log_imm().clone();
            right.log_imm().deliver_unique(0, None, &mut left);
            left_log.deliver_unique(0, None, &mut right);
            
            assert_eq!(left, right);
        }
    }
    mod pn {
        use super::*;
        #[test]
        fn log_len() {
            let mut l = new_pn();
            assert!(l.add(6u64).is_ok());
            assert_eq!(l.log_imm().len(), 1);
            assert!(l.remove(6u64).is_ok());
            assert_eq!(l.log_imm().len(), 2);
        }
        #[test]
        fn add_twice() {
            let mut l = new_pn();
            assert!(l.add(6u64).is_ok());
            assert!(l.add(6u64).is_ok());
            assert_eq!(l.count(&6u64), 2);
        }
        #[test]
        fn add_twice_replicated() {
            let mut l = new_pn();
            assert!(l.add(6u64).is_ok());
            assert!(l.add(6u64).is_ok());
            assert_eq!(l.count(&6u64), 2);
            
            let mut r = new_pn();
            l.log_imm().deliver_pn(0, None, &mut r);
            assert_eq!(r.count(&6u64), 2);
            assert_eq!(r.log_imm().not_downstreamed(), 2);
        }
        #[test]
        fn remove_twice_add_once() {
            let mut l = new_pn();
            assert!(l.remove(6u64).is_ok());
            assert!(l.remove(6u64).is_ok());
            assert!(!l.lookup(&6u64));
        }
        #[test]
        fn remove_twice_counted() {
            let mut l = new_pn();
            assert!(l.remove(6u64).is_ok());
            assert_eq!(l.count(&6u64), -1);
            assert!(l.remove(6u64).is_ok());
            assert_eq!(l.count(&6u64), -2);
        }
        #[test]
        fn remove_twice_logged() {
            let mut l = new_pn();
            assert!(l.remove(6u64).is_ok());
            assert!(l.remove(6u64).is_ok());
            assert_eq!(l.log_imm().len(), 2);
            assert_eq!(l.log_imm().downstreamed(), 2);
        }
        #[test]
        fn commute() {
            let mut l = new_pn();
            assert!(l.remove(6u64).is_ok());
            assert!(l.remove(6u64).is_ok());

            let mut r = new_pn();
            assert!(r.add(6u64).is_ok());

            let ll = l.log_imm().clone();
            r.log_imm().deliver_pn(0, None, &mut l);

            ll.deliver_pn(0, None, &mut r);
            assert_eq!(r.count(&6u64), -1);
            assert_eq!(l.count(&6u64), -1);
        }
        #[test]
        fn delivery_order_insensitive() {
            let mut right = new_pn();
            assert!(right.add(6u64).is_ok());

            let mut left = new_pn();
            assert!(left.add(6u64).is_ok());
            assert!(left.remove(6u64).is_ok());

            let left_log = left.log_imm().clone();
            right.log_imm().deliver_pn(0, None, &mut left);
            left_log.deliver_pn(0, None, &mut right);

            assert_eq!(left, right);
        }
    }
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
