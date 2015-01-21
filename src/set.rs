use super::{Operation, Replica, UpdateError};
use super::{OpBasedReplica};
use std::collections::{btree_map, BTreeMap};
use std::error::Error;
use std::marker::{ContravariantLifetime};

#[derive(Clone)]
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
pub struct Unique<L, A>
    where L: super::Log<Op<A>>, A: Ord + Clone,
{
    state: UniqueReplica<A>,
    log:   L,
}
pub type UniqueError<L, A> = UpdateError<<L as super::Log<Op<A>>>::Error, OpError>;

// An in order iterator over the non-tombstoned elements in the unique set.
pub struct UniqueIter<'a, A: 'a> {
    inner_iter: btree_map::Iter<'a, A, bool>,
    _m1: ContravariantLifetime<'a>,
}
impl<'a, A> Iterator for UniqueIter<'a, A> {
    type Item = &'a A;
    fn next(&mut self) -> Option<&'a A> {
        loop {
            let this = self.inner_iter.next();
            if this.is_none() { return None; }

            let (a, &tombstone) = this.unwrap();
            if tombstone { continue; }
            else         { return Some(a); }
        }
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

    pub fn iter(&self) -> UniqueIter<Atom> {
        self.state.query(|state| UniqueIter {
                inner_iter: state.iter(),
                _m1: ContravariantLifetime,
            } )
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
                    Some(&mut (ref mut added, ref mut removed)) if *added > *removed => {
                        *removed = *removed + 1;
                        return Ok(());
                    },
                    Some(&mut (ref mut added, ref mut removed)) if *added <= *removed => {
                        return Err(OpError::ValueNotPresent);
                    },
                    Some(_) => unreachable!(),
                    None => {},
                }
                assert!(state.insert(v.clone(), (0, 1)).is_none());
            }
        }
        return Ok(());
    }
}

pub type TwoPSetState<A> = BTreeMap<A, (u64, u64)>;
pub type TwoPSetReplica<A> =
    Replica<TwoPSetState<A>, OpBasedReplica<Op<A>, TwoPSetState<A>>>;
// A multi value set.
pub struct TwoPSet<L, A>
    where L: super::Log<Op<A>>,
          A: Ord + Clone,
{
    state: TwoPSetReplica<A>,
    log: L,
}

pub type TwoPSetError<L, A> = UpdateError<<L as super::Log<Op<A>>>::Error, OpError>;

impl<L, A> TwoPSet<L, A>
    where L: super::Log<Op<A>>,
          A: Ord + Clone,
          <L as super::Log<Op<A>>>::Error: Error,
          <Op<A> as Operation<TwoPSetState<A>>>::Error: Error,
{
    pub fn new_with_log(log: L) -> TwoPSet<L, A> {
        TwoPSet {
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

    fn update(&mut self, op: Op<A>) -> Result<(), TwoPSetError<L, A>> {
        let res = self.state.update(&op);
        match res {
            Ok(()) => self.log
                .apply_downstream(op)
                .map_err(|err| UpdateError::Log(err) ),
            Err(err) => Err(UpdateError::Op(err)),
        }
    }

    pub fn add(&mut self, v: A) -> Result<(), TwoPSetError<L, A>> {
        let op = Op::Add(v);
        self.update(op)
    }
    pub fn remove(&mut self, v: A) -> Result<(), TwoPSetError<L, A>> {
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
                   ops: &[Op<A>]) -> (usize, Result<(), TwoPSetError<L, A>>) {
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

#[cfg(test)]
#[allow(dead_code)]
mod test {
    use {Log, Operation};
    use super::*;
    use std::error::Error;

    #[derive(Show)]
    pub struct NullError;
    impl Error for NullError {
        fn description(&self) -> &str { panic!("this error should never happen"); }
    }
    pub type UniqueSet = Unique<DumbLog<Op<u64>, UniqueState<u64>>, u64>;
    #[derive(Default, Clone)]
    pub struct DumbLog<O, S> where O: Operation<S> {
        log: Vec<O>,

        downstreamed: usize,
    }
    pub type TestDumbLog = DumbLog<Op<u64>, UniqueState<u64>>;
    impl DumbLog<Op<u64>, UniqueState<u64>> {
        fn deliver(&self, start: usize, end: Option<usize>, to: &mut UniqueSet) -> (usize, Result<(), UniqueError<TestDumbLog, u64>>) {
            let ops = self.log.as_slice()
                .slice(start, end.unwrap_or(self.len()));
            to.deliver(ops)
        }
    }
    impl<O, S> DumbLog<O, S> {
        fn len(&self) -> usize { self.log.len() }
        fn downstreamed(&self) -> usize {
            self.downstreamed
        }
        fn not_downstreamed(&self) -> usize {
            self.len() - self.downstreamed()
        }
    }
        
    pub fn new_dumb_log() -> TestDumbLog {
        DumbLog {
            log: Vec::new(),
            downstreamed: 0,
        }
    }
    impl Log<Op<u64>> for DumbLog<Op<u64>, UniqueState<u64>> {
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
    pub fn new_unique() -> UniqueSet {
        Unique::new_with_log(new_dumb_log())
    }

    pub fn identical_abstract_states(left: &UniqueSet, right: &UniqueSet) -> bool {
        let mut li = left.iter();
        let mut ri = right.iter();
        loop {
            let l = li.next();
            let r = ri.next();
            if l.is_none() && r.is_some() ||
                l.is_some() && r.is_none() ||
                l != r {
                    return false;
                }
            else if l.is_none() && r.is_none() { return true; }
        }
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
            right.log_imm().deliver(0, None::<usize>, &mut left);
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
            right.log_imm().deliver(0, None, &mut left);

            assert_eq!(left.log_imm().len(), 4);
            assert!(left.lookup(&6u64));
            assert!(left.lookup(&5u64));

            left_log.deliver(0, None, &mut right);
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

            right.log_imm().deliver(0, None, &mut left);

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
            right.log_imm().deliver(0, None, &mut left);
            left_log.deliver(0, None, &mut right);
            
            assert!(identical_abstract_states(&left, &right));
        }
    }
}
