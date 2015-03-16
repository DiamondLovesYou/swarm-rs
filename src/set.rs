use super::{Operation, Replica, UpdateError};
use super::{OpBasedReplica, StateBasedReplica};
use std::borrow::Borrow;
use std::collections::{btree_map, BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::iter;
use std::cmp::Ordering;
use std::ops::Deref;
use uuid;

#[derive(Clone, Debug)]
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
#[derive(Copy, Clone, Debug)]
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
impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.description())
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
#[derive(Clone, Debug)]
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
        self.state.query(move |lhs| {
            lhs.eq(rhs.state.query(move |state| state ))
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
            .query(move |state: &UniqueState<Atom>| {
                match state.get(v) {
                    Some(tomb) => !(*tomb),
                    None => false,
                }
            })
    }

    // Return a forward only iterator over the none tombstoned elements in this set.
    pub fn iter(&self) -> iter::FilterMap<btree_map::Iter<Atom, bool>, for<'a> fn((&'a Atom, &bool)) -> Option<&'a Atom>> {
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
        self.state.query(move |state: &UniqueState<Atom>| state.len() )
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
#[derive(Clone, Debug)]
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
        self.state.query(move |lhs| {
            lhs.eq(rhs.state.query(move |state| state ))
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
            .query(move |state| {
                match state.get(k) {
                    None => 0,
                    Some(&(ref added, ref removed)) => *added as i64 - *removed as i64,
                }
            })
    }

    // Return a forward only iterator over the none tombstoned elements in this set.
    pub fn iter(&self) -> iter::FilterMap<btree_map::Iter<A, (u64, u64)>, for<'a> fn((&'a A, &(u64, u64))) -> Option<&'a A>> {
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
            .query(move |state| {
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
type GOSSReplica<A> = Replica<BTreeSet<A>, StateBasedReplica<GOSSMerger<A>, BTreeSet<A>>>;
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
            .mutate(move |state: &mut BTreeSet<Atom>| {
                state.insert(v);
            })
    }

    pub fn lookup<T: ?Sized>(&self, v: &T) -> bool where Atom: Borrow<T>, T: Ord {
        self.inner_imm()
            .query(move |state: &BTreeSet<Atom>| {
                state.contains(v)
            })
    }

    pub fn len(&self) -> usize {
        self.inner_imm()
            .query(move |state: &BTreeSet<Atom>| {
                state.len()
            })
    }

    pub fn merge(&mut self, right: BTreeSet<Atom>) {
        self.inner_mut()
            .merge(right, goss_merger)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ORElement<A>(A, uuid::Uuid);
impl<A> Deref for ORElement<A> {
    type Target = A;
    fn deref<'a>(&'a self) -> &'a A {
        let &ORElement(ref inner, _) = self;
        inner
    }
}
impl<A: fmt::Debug> fmt::Debug for ORElement<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &ORElement(ref v, ref id) = self;

        write!(f, "ORElement({:?}, {})", v, id.to_hyphenated_string())
    }
}


impl<A: PartialEq + Ord> PartialOrd for ORElement<A> {
    fn partial_cmp(&self, rhs: &ORElement<A>) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl<A: Eq + Ord> Ord for ORElement<A> {
    fn cmp(&self, rhs: &ORElement<A>) -> Ordering {
        fn get_uuid_u128(uuid: &uuid::Uuid) -> (u64, u64) {
            let mut i = uuid.as_bytes().iter().map(|&v| v as u64 );
            let s = &mut i;
            fn n<I>(iter: &mut I) -> <I as Iterator>::Item
                where I: Iterator,
                      <I as Iterator>::Item: Clone,
            {
                iter.next().unwrap().clone()
            }
            (n(s) << 0  | n(s) << 8  | n(s) << 16 | n(s) << 24 |
             n(s) << 32 | n(s) << 40 | n(s) << 48 | n(s) << 56,

             n(s) << 0  | n(s) << 8  | n(s) << 16 | n(s) << 24 |
             n(s) << 32 | n(s) << 40 | n(s) << 48 | n(s) << 56)
        }
        let &ORElement(ref l, ref lid) = self;
        let &ORElement(ref r, ref rid) = rhs;

        match l.cmp(r) {
            Ordering::Equal => {
                let (vl1, vl2) = get_uuid_u128(lid);
                let (vr1, vr2) = get_uuid_u128(rid);
                match vl1.cmp(&vr1) {
                    Ordering::Equal => {},
                    cmp => { return cmp; },
                }
                return vl2.cmp(&vr2);
            },
            cmp => { return cmp; }
        }
    }
}

pub type ORState<A> = BTreeSet<ORElement<A>>;
pub type OROp<A> = Op<ORElement<A>>;

impl<A: Clone + Ord> Operation<ORState<A>> for Op<ORElement<A>> {
    type Error = OpError;
    fn apply_to_state(&self,
                      state: &mut ORState<A>) -> Result<(), OpError> {
        match self {
            &Op::Add(ref v) => {
                if !state.insert(v.clone()) {
                    Err(OpError::ValueAlreadyPresent)
                } else {
                    Ok(())
                }
            }
            &Op::Remove(ref v) => {
                if !state.remove(v) {
                    Err(OpError::ValueNotPresent)
                } else {
                    Ok(())
                }
            }
        }
    }
}

pub type ORError<L, A> =
    UpdateError<<L as super::Log<OROp<A>>>::Error, OpError>;

pub type ORReplica<A> =
    Replica<ORState<A>, OpBasedReplica<OROp<A>, ORState<A>>>;
// An observe only set.
#[derive(Clone, Debug)]
pub struct OR<L, A>
    where L: super::Log<OROp<A>>,
          A: Ord + Clone,
{
    state: ORReplica<A>,
    log: L,
}
impl<L, A> OR<L, A>
    where L: super::Log<OROp<A>>,
          A: Ord + Clone,
          <L as super::Log<OROp<A>>>::Error: Error,
          <OROp<A> as Operation<ORState<A>>>::Error: Error,
{
    pub fn new_with_log(log: L) -> OR<L, A> {
        OR {
            state: super::Replica::new(),
            log: log,
        }
    }

    pub fn log_imm(&self) -> &L { &self.log }
    pub fn log_mut(&mut self) -> &mut L { &mut self.log }

    pub fn lookup(&self, element: &ORElement<A>) -> bool {
        self.state
            .query(move |s| s.contains(element) )
    }

    fn update(&mut self, op: OROp<A>) -> Result<(), ORError<L, A>> {
        let res = self.state
            .update(&op)
            .map_err(|err| UpdateError::Op(err) );
        match res {
            Ok(()) => self.log_mut()
                .apply_downstream(op)
                .map_err(|err| UpdateError::Log(err) ),
            err => err,
        }
    }

    pub fn insert(&mut self, v: A) -> Result<ORElement<A>, ORError<L, A>> {
        let id = uuid::Uuid::new_v4();
        let element = ORElement(v, id);
        let op = Op::Add(element.clone());
        self.update(op)
            .map(move |()| element )
    }
    pub fn remove(&mut self, element: ORElement<A>) -> Result<(), ORError<L, A>> {
        let op = Op::Remove(element);
        self.update(op)
    }
    pub fn deliver(&mut self,
                   ops: &[OROp<A>]) -> (usize, Result<(), ORError<L, A>>) {
        let mut successful: usize = 0;
        for op in ops.iter() {
            let res = self.state
                .update(op);
            let res = match res {
                Ok(()) => self.log
                    .add_to_log(op.clone())
                    .map_err(|err| UpdateError::Log(err) ),
                Err(err) => Err(UpdateError::Op(err)),
            };
            if res.is_ok() {
                successful += 1;
            } else {
                return (successful, res);
            }
        }
        return (successful, Ok(()));
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod test {
    use {Log, UpdateError, NullError};
    use super::*;
    use test_helpers::*;

    pub type UniqueSet = Unique<DumbLog<Op<u64>>, u64>;
    pub type TestPN = super::PN<DumbLog<Op<u64>>, u64>;
    pub type TestDumbLog = DumbLog<Op<u64>>;
    pub type TestOR = super::OR<DumbLog<OROp<u64>>, u64>;
    impl_deliver! { UniqueState<u64>, UniqueSet => deliver_unique }
    impl_deliver! { PNState<u64>, TestPN => deliver_pn }
    impl_deliver! { ORState<u64>, OROp<u64>, TestOR => deliver_or }
    impl_log_for_state! { Op<u64> }
    impl_log_for_state! { OROp<u64> }

    pub fn new_unique() -> UniqueSet {
        Unique::new_with_log(new_dumb_log!())
    }

    pub fn new_pn() -> TestPN {
        PN::new_with_log(new_dumb_log!())
    }
    pub fn new_or() -> TestOR {
        OR::new_with_log(new_dumb_log!())
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
            deliver_unique(right.log_imm(), 0, None, &mut left);
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
            deliver_unique(right.log_imm(), 0, None, &mut left);

            assert_eq!(left.log_imm().len(), 4);
            assert!(left.lookup(&6u64));
            assert!(left.lookup(&5u64));

            deliver_unique(&left_log, 0, None, &mut right);
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

            deliver_unique(right.log_imm(), 0, None, &mut left);

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
            deliver_unique(right.log_imm(), 0, None, &mut left);
            deliver_unique(&left_log, 0, None, &mut right);

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
            deliver_pn(l.log_imm(), 0, None, &mut r);
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
            deliver_pn(r.log_imm(), 0, None, &mut l);

            deliver_pn(&ll, 0, None, &mut r);
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
            deliver_pn(right.log_imm(), 0, None, &mut left);
            deliver_pn(&left_log, 0, None, &mut right);

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

    mod or {
        use {UpdateError};
        use super::*;
        use super::super::*;
        use test_helpers::DumbLog;

        type TestORError = ORError<DumbLog<OROp<u64>>, u64>;
        type TestORResult = Result<(), TestORError>;

        #[test]
        fn insert() {
            let mut l = new_or();
            let _e1 = l.insert(6u64).unwrap();
            let _e2 = l.insert(6u64).unwrap();
        }

        #[test]
        fn inserts_commute() {
            let mut l = new_or();
            let le1 = l.insert(6u64).unwrap();
            let ll = l.log_imm().clone();

            let mut r = new_or();
            let re1 = r.insert(6u64).unwrap();
            let rl = r.log_imm().clone();

            let (_, res) = deliver_or(&ll, 0, None, &mut r);
            res.unwrap();
            let (_, res) = deliver_or(&rl, 0, None, &mut l);
            res.unwrap();

            assert!(l.lookup(&re1));
            assert!(l.lookup(&le1));
            assert!(r.lookup(&re1));
            assert!(r.lookup(&le1));
        }

        #[test]
        fn inserts_are_unique() {
            let mut l = new_or();
            let le1 = l.insert(6u64).unwrap();
            let ll = l.log_imm().clone();

            let mut r = new_or();
            let re1 = r.insert(6u64).unwrap();
            let rl = r.log_imm().clone();

            assert!(match r.remove(le1.clone()) {
                Err(UpdateError::Op(OpError::ValueNotPresent)) => true,
                _ => false,
            });
            assert!(match l.remove(re1.clone()) {
                Err(UpdateError::Op(OpError::ValueNotPresent)) => true,
                _ => false,
            });

            let (_, res) = deliver_or(&ll, 0, None, &mut r);
            res.unwrap();
            let (_, res) = deliver_or(&rl, 0, None, &mut l);
            res.unwrap();

            l.remove(le1.clone()).unwrap();
            assert!(l.lookup(&re1));
            r.remove(re1.clone()).unwrap();
            assert!(r.lookup(&le1));
        }

        #[test]
        fn remove() {
            let mut l = new_or();
            let v = l.insert(6u64).unwrap();
            l.remove(v).unwrap();
        }

        #[test]
        fn removes_commute() {
            let mut l = new_or();
            let le1 = l.insert(6u64).unwrap();

            let mut r = new_or();
            let re1 = r.insert(6u64).unwrap();

            let ll = l.log_imm().clone();
            let rl = r.log_imm().clone();

            let (_, res) = deliver_or(&ll, 0, None, &mut r);
            res.unwrap();
            let (_, res) = deliver_or(&rl, 0, None, &mut l);
            res.unwrap();

            let offset = l.log_imm().len();

            println!("l: {:?}", l);
            println!("r: {:?}", r);

            l.remove(re1.clone()).unwrap();
            let (_, res) = deliver_or(l.log_imm(), offset, None, &mut r);
            res.unwrap();

            println!("l: {:?}", l);
            println!("r: {:?}", r);

            assert!(!l.lookup(&re1));
            assert!(l.lookup(&le1));
            assert!(!r.lookup(&re1));
            assert!(r.lookup(&le1));
        }
    }
}
