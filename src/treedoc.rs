use {OpBasedReplica, UpdateError};
use std::collections::bitv;
use std::collections::{Bitv, BTreeMap};
use std::collections::btree_map::{self, Range};
use std::collections::Bound::{Included, Excluded, Unbounded};
use std::cmp::{Ordering};
use std::default::Default;

pub use super::set::OpError;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SiteId(pub u64);

pub trait SiteIdentifier {
    fn site_id(&self) -> SiteId;
}

#[derive(Eq, Ord, PartialEq, PartialOrd, Clone, Copy, Debug, Hash)]
pub struct Disambiguator {
    counter: u64,
    site: SiteId,
}
impl Disambiguator {
    #[cfg(test)]
    pub fn new_raw(counter: u64, site: SiteId) -> Disambiguator {
        Disambiguator {
            counter: counter,
            site: site,
        }
    }
    pub fn site_id(&self) -> SiteId { self.site }
}
#[derive(Debug, Clone, Hash)]
pub struct Path {
    // False is left, true is right.
    prefix: Bitv,
    next: Option<(Disambiguator, Option<Box<Path>>)>,
}
impl Path {
    pub fn new_empty() -> Path {
        Path {
            prefix: Bitv::new(),
            next: None,
        }
    }

    #[cfg(test)]
    pub fn new_raw(prefix: Bitv,
                   next: Option<(Disambiguator, Option<Box<Path>>)>) -> Path {
        if let Some((_, Some(ref p))) = next {
            assert!(!p.empty());
        }

        Path {
            prefix: prefix,
            next: next,
        }
    }

    pub fn ends_with_disambiguator(&self) -> bool {
        match self.next.as_ref() {
            None => false,
            Some(&(_, None)) => true,
            Some(&(_, Some(ref next))) => next.ends_with_disambiguator(),
        }
    }

    pub fn next_imm(&self) -> Option<&Path> {
        self.next
            .as_ref()
            .and_then(move |: &(_, ref v)| {
                v.as_ref()
                    .map(move |: v| &(**v) )
            })
    }
    fn next_mut(&mut self) -> Option<&mut Path> {
        self.next
            .as_mut()
            .and_then(move |: &mut (_, ref mut p)| {
                p.as_mut()
                    .map(move |: b| &mut (**b) )
            })
    }

    pub fn disambiguator(&self) -> Option<&Disambiguator> {
        self.next
            .as_ref()
            .map(move |: &(ref dis, _)| dis )
    }
    pub fn last_disambiguator(&self) -> Option<&Disambiguator> {
        self.next
            .as_ref()
            .and_then(move |: &(ref dis, ref n)| {
                if let &Some(ref n) = n {
                    n.last_disambiguator()
                } else {
                    Some(dis)
                }
            })
    }

    fn take_last_disambiguator(&mut self) -> Option<Disambiguator> {
        self.next
            .as_mut()
            .and_then(move |: &mut (_, ref mut n)| {
                n.as_mut()
                    .and_then(move |: n| n.take_last_disambiguator() )
            })
            .or_else(move |:| {
                self.next
                    .take()
                    .map(move |: (dis, _)| dis )
            })
    }

    fn push_prefix(&mut self, prefix: &Bitv) {
        if let Some(next) = self.next_mut() {
            next.push_prefix(prefix);
            return;
        }

        if let Some(&dis) = self.disambiguator() {
            // create a new path:
            let next = Path {
                prefix: prefix.clone(),
                next: None,
            };
            self.next = Some((dis, Some(box next)));
        } else {
            self.prefix.extend(prefix.iter());
        }
    }
    fn set_last_disambiguator(&mut self, dis: Disambiguator, p: Option<Box<Path>>) {
        match self.next {
            None | Some((_, None)) => {
                debug_assert!(p.as_ref().map(move |: p| !p.empty() ).unwrap_or(true));
                self.next = Some((dis, p));
            }
            Some((_, Some(ref mut next))) => {
                next.set_last_disambiguator(dis, p)
            }
        }
    }

    /// Assuming this Path starts at the root, return the height of the tree.
    pub fn height(&self) -> usize {
        let mut this_opt = Some(self);
        let mut count: usize = 0;
        loop {
            let this = match this_opt {
                None => break,
                Some(this) => this,
            };
            count += this.prefix.len();
            this_opt = this.next_imm();
        }
        return count;
    }

    /// Get the depth of disambiguators
    pub fn links(&self) -> usize {
        let rest = self.next_imm()
            .map(|n| n.links() )
            .unwrap_or(0);
        let own = self.disambiguator()
            .map(|_| 1 )
            .unwrap_or(0);

        rest + own
    }

    /// a.k.a. is this the root?
    pub fn empty(&self) -> bool {
        self.prefix.len() == 0
    }

    pub fn iter<'a>(&'a self) -> PathEntries<'a> {
        let &Path {
            ref prefix,
            ref next,
        } = self;
        PathEntries {
            prefix_iter: prefix.iter(),
            next:        next.as_ref(),
        }
    }

    pub fn distance(&self, rhs: &Path) -> usize {
        let mut liter = self.iter();
        let mut riter = rhs.iter();

        let mut diverged = false;
        let mut distance: usize = 0;

        loop {
            let l = liter.next();
            let r = riter.next();

            match (l, r) {
                (None, None) => return distance,
                (Some(ref lv), Some(ref rv)) if lv == rv => {
                    if diverged { distance += 2; }
                },
                (Some(ref lv), Some(ref rv)) if lv != rv => {
                    diverged = true;
                    distance += 2;
                },

                (None, Some(_)) | (Some(_), None) => {
                    diverged = true;
                    distance += 1;
                },
                _ => unreachable!(),
            }
        }
    }

    pub fn last_bit(&self) -> Option<bool> {
        self.next_imm()
            .and_then(|n| n.last_bit() )
            .or_else(|| {
                if self.prefix.len() == 0 {
                    None
                } else {
                    Some(self.prefix[self.prefix.len() - 1])
                }
            })
    }

    fn last_path_mut(&mut self) -> &mut Path {
        match self.next {
            None | Some((_, None)) => {},
            Some((_, Some(ref mut next))) => {
                return next.last_path_mut();
            },
        }
        self
    }
}
pub trait ChildPath {
    fn get_child(self, side: Side) -> Path;
}
impl ChildPath for Option<Path> {
    fn get_child(self, side: Side) -> Path {
        match self {
            None => Path {
                prefix: Bitv::from_elem(1, side.to_bit()),
                next: None,
            },
            Some(p) => p.get_child(side),
        }
    }
}
impl ChildPath for Path {
    fn get_child(mut self, side: Side) -> Path {
        let prefix = Bitv::from_elem(1, side.to_bit());
        self.push_prefix(&prefix);
        self
    }
}

impl Eq for Path {}
impl PartialEq for Path {
    fn eq(&self, rhs: &Path) -> bool {
        match self.cmp(rhs) {
            Ordering::Equal => true,
            _ => false,
        }
    }
}
impl PartialOrd for Path {
    fn partial_cmp(&self, rhs: &Path) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Path {
    fn cmp(&self, rhs: &Path) -> Ordering {
        let mut left_iter = self.iter().peekable();
        let mut right_iter = rhs.iter().peekable();

        loop {
            let left = left_iter.next();
            let right = right_iter.next();

            match (left, right) {
                (None, None) => { return Ordering::Equal; },

                (None, Some(PathEntry::Ambiguity(_))) |
                (Some(PathEntry::Ambiguity(_)), None) => {
                    if left_iter.peek().is_some() || right_iter.peek().is_some() {
                        panic!("Impossible branched reached; \
                                did delivery order violate causal order?")
                    }
                    return Ordering::Equal;
                }

                (Some(PathEntry::Bit(l)), Some(PathEntry::Bit(r))) => {
                    let cmp = l.cmp(&r);
                    if cmp == Ordering::Equal {
                        continue;
                    } else {
                        return cmp;
                    }
                },
                (Some(PathEntry::Ambiguity(l)), Some(PathEntry::Ambiguity(r))) => {
                    let cmp = l.cmp(r);
                    if cmp == Ordering::Equal {
                        continue;
                    } else {
                        return cmp;
                    }
                },


                (Some(PathEntry::Bit(false)), Some(PathEntry::Ambiguity(_))) |
                (Some(PathEntry::Ambiguity(_)), Some(PathEntry::Bit(true)))  => {
                    return Ordering::Less;
                },
                (Some(PathEntry::Bit(true)),  Some(PathEntry::Ambiguity(_))) |
                (Some(PathEntry::Ambiguity(_)), Some(PathEntry::Bit(false))) => {
                    return Ordering::Greater;
                },


                (Some(PathEntry::Bit(false)), None) |
                (None, Some(PathEntry::Bit(true))) => { return Ordering::Less; },
                (Some(PathEntry::Bit(true)),  None) |
                (None, Some(PathEntry::Bit(false))) => { return Ordering::Greater; },
            }
        } // Loop.
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum PathEntry<'a> {
    Bit(bool),
    Ambiguity(&'a Disambiguator),
}
pub struct PathEntries<'a> {
    prefix_iter: bitv::Iter<'a>,
    next: Option<&'a (Disambiguator, Option<Box<Path>>)>,
}
impl<'a> Iterator for PathEntries<'a> {
    type Item = PathEntry<'a>;
    fn next(&mut self) -> Option<PathEntry<'a>> {
        let next_prefix = self.prefix_iter.next();
        match next_prefix {
            Some(bit) => Some(PathEntry::Bit(bit)),
            None => {
                let next = self.next.take();
                match next {
                    None => None,
                    Some(&(ref dis, ref next_path)) => {
                        let ret = PathEntry::Ambiguity(dis);
                        if let Some(ref next_path) = next_path.as_ref() {
                            self.prefix_iter = next_path.prefix.iter();
                            self.next = next_path.next.as_ref();
                        }
                        Some(ret)
                    }
                }
            }
        }
    }
}
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Side {
    Left,
    Right,
}
impl Side {
    fn to_bit(&self) -> bool {
        match self {
            &Side::Left => false,
            &Side::Right => true,
        }
    }
    pub fn opposite_side(self) -> Side {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }
}
#[derive(Clone)]
pub enum Op<A> {
    /// Atoms are always inserted with disambiguators; however, paths don't
    /// require disambiguators in order to have total order
    Insert {
        at: Path,
        node: A,
    },
    Delete(Path),
}
impl<A: Clone> super::Operation<State<A>> for Op<A> {
    type Error = OpError;
    fn apply_to_state(&self,
                      state: &mut State<A>) -> Result<(), OpError> {
        match self {
            &Op::Insert {
                at: ref at,
                node: ref n,
            } => {
                debug_assert!(at.ends_with_disambiguator());
                match state.entry(at.clone()) {
                    btree_map::Entry::Vacant(view) => {
                        view.insert(n.clone());
                        Ok(())
                    }
                    btree_map::Entry::Occupied(_) => {
                        Err(OpError::ValueAlreadyPresent)
                    }
                }
            },
            &Op::Delete(ref at) => {
                match state.entry(at.clone()) {
                    btree_map::Entry::Vacant(_) => {
                        Err(OpError::ValueNotPresent)
                    }
                    btree_map::Entry::Occupied(view) => {
                        view.remove();
                        Ok(())
                    }
                }
            },
        }
    }
}
pub type State<A> = BTreeMap<Path, A>;
pub type Replica<A> = super::Replica<State<A>, OpBasedReplica<Op<A>, State<A>>>;

pub type TreedocError<L, A> = UpdateError<<L as super::Log<Op<A>>>::Error, OpError>;

pub struct Treedoc<L, A>
    where L: super::Log<Op<A>>,
          A: Clone,
{
    // a.k.a. the tree root
    state: Replica<A>,
    site: SiteId,
    disambiguator_counter: u64,
    log: L,
}
impl<L, A> Treedoc<L, A>
    where L: super::Log<Op<A>>,
          A: Clone,
          <L as super::Log<Op<A>>>::Error: ::std::error::Error,
{
    pub fn new(log: L, site: SiteId) -> Treedoc<L, A> {
        Treedoc {
            state: Default::default(),
            site: site,
            disambiguator_counter: 0,
            log: log,
        }
    }

    pub fn log_mut<'a>(&'a mut self) -> &'a mut L { &mut self.log }
    pub fn log_imm<'a>(&'a self) -> &'a L { &self.log }

    pub fn next_disambiguator(&mut self) -> Disambiguator {
        let discounter = self.disambiguator_counter;
        self.disambiguator_counter += 1;

        Disambiguator {
            site: self.site.clone(),
            counter: discounter,
        }
    }

    fn update(&mut self, op: Op<A>) -> Result<(), TreedocError<L, A>> {
        let res = self.state
            .update(&op)
            .map_err(|err| UpdateError::Op(err) );
        match res {
            Ok(()) => {
                self.log_mut()
                    .apply_downstream(op)
                    .map_err(|err| UpdateError::Log(err) )
            },
            err => err,
        }
    }

    fn next_path(&mut self, prev: Path, side_first: Side) -> Path {
        /* TODO
        struct PathInsertionOrderIter {
            cache_depth: u64,
            side_first: Side,
            root: Path,

            current: Path,
            last_side: Side,
        }

        impl<'a> PathInsertionOrderIter {
        }



        let mut insertion_iter = PathInsertionOrderIter {
            cache_depth: (self.len() as f64).log2().ceil() as u64 + 1,

        };
        let mut iter = self.state
            .query(move |: state| {
                state.range(Included(prev), Unbounded)
            });
        let (lower_bound_size, _) = iter.size_hint();
        let mut empty = Vec::with_capacity(lower_bound_size);
        for (path, _) in*/

        let mut next = prev.get_child(side_first);
        next.set_last_disambiguator(self.next_disambiguator(), None);
        self.prune_unambiguous(next)
    }

    /// March through all of `p`'s paths, removing disambiguators which don't
    /// disambiguate (ie there isn't any ambiguity).
    pub fn prune_unambiguous(&self, Path { prefix, next }: Path) -> Path {
        if next.is_none() { return Path { prefix: prefix, next: next }; }

        let mut result = Path::new_empty();
        result.push_prefix(&prefix);

        let (dis, next) = next.unwrap();
        prune(self, &mut result, dis, next);
        return result;

        fn prune<L, A>(this: &Treedoc<L, A>,
                       result: &mut Path,
                       dis: Disambiguator,
                       mut current: Option<Box<Path>>)
            where L: super::Log<Op<A>>, A: Clone,
                  <L as super::Log<Op<A>>>::Error: ::std::error::Error,
        {
            debug_assert!(!result.ends_with_disambiguator());
            let has_siblings = current.is_none() || this
                .mini_siblings_of(&*result)
                .count() > 1;

            let next = current
                .as_mut()
                .and_then(move |: c| c.next.take() );

            if has_siblings {
                result.set_last_disambiguator(dis, current);
            } else {
                match current {
                    Some(box Path {
                        prefix, next: None,
                    }) => result.push_prefix(&prefix),
                    _ => unreachable!(),
                }
            }

            if let Some((dis, next)) = next {
                prune(this, result, dis, next);
            }
        }

    }

    pub fn mini_siblings_of(&self, p: &Path) -> Range<Path, A> {
        let mut p = p.clone();
        p.take_last_disambiguator();

        let l_bound = p.clone()
            .get_child(Side::Left);
        let r_bound = p.get_child(Side::Right);
        self.state
            .query(move |: state| {
                state.range(Excluded(&l_bound),
                            Excluded(&r_bound))
            })
    }

    /// Return an existing path that is n positions right of `path`.
    pub fn get_path_n_right(&self, path: &Path, n: usize) -> Option<&Path> {
        self.get_n_right(path, n)
            .map(|(p, _)| p )
    }
    /// Return a reference to an existing path with its associated value n positions right.
    /// Returns `None` if there is no such position.
    pub fn get_n_right(&self, path: &Path, n: usize) -> Option<(&Path, &A)> {
        self.state
            .query(move |: state| {
                state.range(Included(path), Unbounded)
                    .nth(n)
            })
    }
    /// Return an existing path that is n positions right of `path`.
    pub fn get_path_n_left(&self, path: &Path, n: usize) -> Option<&Path> {
        self.get_n_left(path, n)
            .map(|(p, _)| p )
    }
    /// Return a reference to an existing path with its associated value n positions right.
    /// Returns `None` if there is no such position.
    pub fn get_n_left(&self, path: &Path, n: usize) -> Option<(&Path, &A)> {
        self.state
            .query(move |: state| {
                // reverse_in_place is messing up, so we have to do this manually.
                let mut iter = state
                    .range(Unbounded, Included(path));
                let mut n = n;
                loop {
                    let v = iter.next_back();
                    if v.is_none() { return None; }
                    else if n == 0 { return v; }
                    else { n -= 1; }
                }
            })
    }

    pub fn insert_at(&mut self, at: Option<Path>,
                     value: A) -> Result<Path, TreedocError<L, A>> {
        let at = at
            .unwrap_or_else(|| {
                let mut p = Path::new_empty();
                p.set_last_disambiguator(self.next_disambiguator(), None);
                p
            });

        let op = Op::Insert {
            at: at.clone(),
            node: value,
        };
        self.update(op)
            .map(move |: ()| at )
    }
    pub fn insert_right(&mut self,
                        left: Path, value: A) -> Result<Path, TreedocError<L, A>> {
        let new_path = self.next_path(left, Side::Right);
        self.insert_at(Some(new_path.clone()), value)
            .map(move |: _| new_path )
    }
    pub fn insert_left(&mut self,
                       right: Path, value: A) -> Result<Path, TreedocError<L, A>> {
        let new_path = self.next_path(right, Side::Left);
        self.insert_at(Some(new_path.clone()), value)
            .map(move |: _| new_path )
    }

    // TODO: return the value deleted.
    pub fn delete(&mut self, at: Option<Path>) -> Result<(), TreedocError<L, A>> {
        let at = at
            .unwrap_or_else(|| {
                let mut p = Path::new_empty();
                p.set_last_disambiguator(self.next_disambiguator(), None);
                p
            });
        let op = Op::Delete(at);
        self.update(op)
    }

    pub fn iter(&self) -> btree_map::Values<Path, A> {
        self.state.query(move |: state| state.values() )
    }

    pub fn deliver(&mut self,
                   ops: &[Op<A>]) -> (usize, Result<(), TreedocError<L, A>>) {
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
#[allow(unused_variables)]
mod test {
    use super::*;
    use std::cell::Cell;
    thread_local! { static SITE_COUNTER: Cell<u64> = Cell::new(0) }
    pub fn next_site_id() -> SiteId {
        SiteId(SITE_COUNTER.with(|v| {
            v.set(v.get() + 1);
            v.get()
        }))
    }

    pub trait NextDis {
        fn next_dis(&self) -> Disambiguator;
    }
    impl NextDis for SiteId {
        fn next_dis(&self) -> Disambiguator {
            Disambiguator::new_raw(next_dis_counter(),
                                   self.clone())
        }
    }

    thread_local! { static DIS_COUNTER: Cell<u64> = Cell::new(0) }
    pub fn next_dis_counter() -> u64 {
        DIS_COUNTER.with(|v| {
            v.set(v.get() + 1);
            v.get()
        })
    }
    pub fn next_dis() -> Disambiguator {
        const SITE: SiteId = SiteId(20);
        Disambiguator::new_raw(next_dis_counter(),
                               SITE.clone())
    }

    pub fn debug_path(p: Path) -> Path {
        println!("");
        println!("path iters:");
        for i in p.iter() {
            println!("{:?}", i);
        }
        p
    }

    pub fn debug_paths(l: &Path, r: &Path) {
        println!("");
        println!("path iters:");
        let mut li = l.iter();
        let mut ri = r.iter();
        let mut i: usize = 0;
        loop {
            let l = li.next();
            let r = ri.next();
            println!("##{}: left: {:?} right: {:?}", i, l, r);
            if l.is_none() && r.is_none() { break; }
            i += 1;
        }
    }
    mod treedoc {
        use {UpdateError, Log};
        use super::*;
        use super::super::*;
        use test_helpers::*;
        use std::collections::Bitv;
        use std::slice::SliceConcatExt;

        type TestTD = Treedoc<DumbLog<Op<String>, State<String>>, String>;

        impl_deliver! { State<String>, Op<String>, TestTD => deliver_td }
        impl_log_for_state! { State<String>, Op<String> }

        fn new_td() -> TestTD {
            Treedoc::new(new_dumb_log!(),
                         next_site_id())
        }

        fn check_td(td: &TestTD, str: &'static str) {
            let result = {
                let v: Vec<String> = td.iter()
                    .map(|v| v.to_string() )
                    .collect();
                v.connect(" ")
            };
            assert_eq!(result, str);
        }

        #[test]
        fn insert_at_root() {
            let mut c = new_td();
            c.insert_at(None, "test".to_string()).unwrap();
            check_td(&c, "test");
        }
        #[test]
        fn insert_right() {
            let mut c = new_td();
            let p = c.insert_at(None, "test".to_string()).unwrap();
            c.insert_right(p, "test2".to_string()).unwrap();
            check_td(&c, "test test2");
        }
        #[test]
        fn insert_left() {
            let mut c = new_td();
            let p = c.insert_at(None, "test".to_string()).unwrap();
            c.insert_left(p, "test2".to_string()).unwrap();
            check_td(&c, "test2 test");
        }
        #[test]
        fn delete() {
            let mut c = new_td();
            let p1 = c.insert_at(None, "test".to_string()).unwrap();
            let p2 = c.insert_left(p1.clone(), "test2".to_string()).unwrap();
            c.delete(Some(p1)).unwrap();
            check_td(&c, "test2");
        }
        #[test]
        fn inserts_commute() {
            let mut r = new_td();
            let mut l = new_td();

            r.insert_at(None, "test_right".to_string()).unwrap();
            l.insert_at(None, "test_left".to_string()).unwrap();

            let ll = l.log_imm().clone();
            deliver_td(r.log_imm(), 0, None, &mut l);
            deliver_td(&ll, 0, None, &mut r);

            const RESULT: &'static str = "test_right test_left";

            check_td(&r, RESULT);
            check_td(&l, RESULT);
        }
        #[test]
        fn prune_unambiguous() {
            let mut r = new_td();
            let mut l = new_td();

            let p1 = r.insert_at(None, "test_right".to_string())
                .unwrap();
            l.insert_at(None, "test_left".to_string()).unwrap();

            let ll = l.log_imm().clone();
            deliver_td(r.log_imm(), 0, None, &mut l);
            deliver_td(&ll, 0, None, &mut r);

            const RESULT: &'static str = "test_right test_left";

            check_td(&r, RESULT);
            check_td(&l, RESULT);

            let p2 = r.insert_right(p1, "test_right-right".to_string()).unwrap();
            assert_eq!(p2.links(), 2);
            check_td(&r, "test_right test_right-right test_left");
        }
        #[test]
        fn get_n_left_or_right() {
            // left:
            {
                let mut c = new_td();
                let p1 = c.insert_at(None, "center".to_string())
                    .unwrap();
                let p2 = c.insert_left(p1.clone(), "left1".to_string()).unwrap();
                let p3 = c.insert_left(p2.clone(), "left2".to_string()).unwrap();
                let p4 = c.insert_left(p3, "left3".to_string()).unwrap();

                let right_2 = c.get_n_right(&p4, 2)
                    .map(|(p, v)| (p, v.as_slice()) );
                assert_eq!(right_2, Some((&p2, "left1")));
                let p2_b = right_2.map(|(p, _)| p ).unwrap();
                let p1_b = c.get_path_n_right(p2_b, 1).unwrap();
                assert_eq!(p1_b, &p1);

                assert_eq!(c.get_n_right(&p4, 10), None);
            }

            // right:
            {
                let mut c = new_td();
                let p1 = c.insert_at(None, "center".to_string())
                    .unwrap();
                let p2 = c.insert_right(p1.clone(), "right1".to_string()).unwrap();
                let p3 = c.insert_right(p2.clone(), "right2".to_string()).unwrap();
                let p4 = c.insert_right(p3, "right3".to_string()).unwrap();

                let left_2 = c.get_n_left(&p4, 2)
                    .map(|(p, v)| (p, v.as_slice()) );
                assert_eq!(left_2, Some((&p2, "right1")));
                let p2_b = left_2.map(|(p, _)| p ).unwrap();
                let p1_b = c.get_path_n_left(p2_b, 1).unwrap();
                assert_eq!(p1_b, &p1);

                assert_eq!(c.get_n_left(&p4, 10), None);
            }
        }

        #[test]
        fn delete_before_insert() {
            let mut c = new_td();
            assert!(match c.delete(None) {
                Err(UpdateError::Op(OpError::ValueNotPresent)) => true,
                _ => false,
            });
        }
    }
    mod path {
        use super::*;
        use super::super::*;
        use std::collections::Bitv;

        #[test]
        pub fn simple_height() {
            let mut prefix = Bitv::from_elem(3, false);
            prefix.set(2, true);
            let p = Path::new_raw
                (prefix, None);
            assert_eq!(p.height(), 3);
        }
        #[test]
        pub fn disambiguator_height() {
            let mut prefix = Bitv::from_elem(3, false);
            prefix.set(1, true);
            let dis = Disambiguator::new_raw(10, SiteId(20));
            let p = Path::new_raw(prefix, Some((dis, None)));
            assert_eq!(p.height(), 3);
        }
        #[test]
        pub fn nested_disambiguator_height() {
            let prefix = Bitv::from_elem(3, false);
            let dis1 = Disambiguator::new_raw(0, SiteId(20));
            let dis2 = Disambiguator::new_raw(1, SiteId(20));
            let nested = box Path::new_raw(prefix.clone(), Some((dis2, None)));
            let l = Path::new_raw(prefix.clone(), Some((dis1, Some(nested))));
            assert_eq!(l.height(), 6);
        }
        #[test]
        pub fn prefix_cmp_neq() {
            let l = {
                let prefix = Bitv::from_elem(3, false);
                Path::new_raw(prefix, None)
            };
            let r = {
                let prefix = Bitv::from_elem(3, true);
                Path::new_raw(prefix, None)
            };
            assert!(l != r);
        }
        #[test]
        pub fn prefix_cmp_eq() {
            let l = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, None)
            };
            let r = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, None)
            };
            println!("{:?} == {:?}", l, r);
            assert!(l == r);
        }
        #[test]
        pub fn prefix_cmp() {
            let l = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, None)
            };
            let r = {
                let prefix = Bitv::from_elem(3, false);
                Path::new_raw(prefix, None)
            };

            assert!(l > r);

            let r = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(1, true);
                Path::new_raw(prefix, None)
            };

            assert!(l < r);
        }
        #[test]
        pub fn prefix_disambiguator_cmp() {
            let l = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, None)
            };
            let r = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, Some((next_dis(), None)))
            };

            assert_eq!(l, r);

        }

        // test that disambiguators are in the 'center':
        #[test]
        pub fn disambiguator_cmp_center() {
            let l = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, Some((next_dis(), None)))
            };
            let r = {
                let mut prefix = Bitv::from_elem(4, false);
                prefix.set(2, true);
                Path::new_raw(prefix, None)
            };
            println!("{:?} < {:?}", r, l);
            assert!(r < l);

            let l = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, Some((next_dis(), None)))
            };
            let r = {
                let mut prefix = Bitv::from_elem(4, false);
                prefix.set(2, true);
                prefix.set(3, true);
                Path::new_raw(prefix, None)
            };

            println!("{:?} < {:?}", l, r);
            assert!(l < r);
        }
        #[test]
        #[should_fail]
        pub fn disambiguator_with_suffix() {
            // it's impossible to insert a child node before the parent mininode.
            let l = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, None)
            };
            let r = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                let suffix = Bitv::from_elem(2, false);
                let suffix = Path::new_raw(suffix, None);
                Path::new_raw(prefix, Some((next_dis(), Some(box suffix))))
            };
            let _ = l == r;
        }
        #[test]
        pub fn disambiguator_cmp() {
            let c_dis = next_dis();
            let c = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                Path::new_raw(prefix, Some((c_dis.clone(), None)))
            };

            let l = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                let inner = box Path::new_raw(Bitv::from_elem(1, false),
                                              None);
                Path::new_raw(prefix, Some((c_dis.clone(), Some(inner))))
            };

            debug_paths(&l, &c);
            println!("{:?} < {:?}", l, c);
            assert!(l < c);

            let r = {
                let mut prefix = Bitv::from_elem(3, false);
                prefix.set(2, true);
                let inner = box Path::new_raw(Bitv::from_elem(1, true),
                                              None);
                Path::new_raw(prefix, Some((c_dis.clone(), Some(inner))))
            };

            println!("{:?} < {:?}", c, r);
            assert!(c < r);
        }
        #[test]
        fn ends_with_disambiguator() {
            let c = {
                let prefix = Bitv::from_elem(3, false);
                Path::new_raw(prefix, Some((next_dis(), None)))
            };
            assert!(c.ends_with_disambiguator());
        }
        #[test]
        fn nested_ends_with_disambiguator() {
            let c = {
                let prefix = Bitv::from_elem(3, false);
                let inner = Path::new_raw(Bitv::from_elem(1, true),
                                          Some((next_dis(), None)));
                let inner = box inner;
                Path::new_raw(prefix, Some((next_dis(),
                                            Some(inner))))
            };
            assert!(c.ends_with_disambiguator());
        }
        #[test]
        fn one_way_distance() {
            let l = {
                let prefix = Bitv::from_elem(6, false);
                Path::new_raw(prefix, None)
            };
            let r = {
                let prefix = Bitv::from_elem(3, false);
                Path::new_raw(prefix, None)
            };

            assert_eq!(l.distance(&r), 3);
            assert_eq!(r.distance(&l), 3);
        }
        #[test]
        fn two_way_distance() {
            let l = {
                let mut prefix = Bitv::from_elem(6, false);
                prefix.set(3, true);
                Path::new_raw(prefix, None)
            };
            let r = {
                let prefix = Bitv::from_elem(7, false);
                Path::new_raw(prefix, None)
            };
            assert_eq!(l.distance(&r), 7);
            assert_eq!(r.distance(&l), 7);
        }

        #[test]
        fn last_bit() {
            let c = {
                let mut prefix = Bitv::from_elem(2, false);
                prefix.set(1, true);
                Path::new_raw(prefix, None)
            };
            assert_eq!(c.last_bit(), Some(true));

            let c = {
                let prefix = Bitv::from_elem(2, false);
                let inner = Path::new_raw(Bitv::from_elem(2, true), None);
                Path::new_raw(prefix, Some((next_dis(), Some(box inner))))
            };
            assert_eq!(c.last_bit(), Some(true));
        }
    }
}
