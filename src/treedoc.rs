use {OpBasedReplica, UpdateError};
use std::collections::bitv;
use std::collections::{Bitv, BTreeMap};
use std::cmp::{Ordering};
use std::default::Default;
pub use super::set::OpError;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SiteId(pub u64);

pub trait SiteIdentifier {
    fn site_id(&self) -> SiteId;
}

#[derive(Default, Clone, Debug)]
pub struct Node<Atom> {
    left: Option<Box<Node<Atom>>>,
    kind: Option<Kind<Atom>>,
    right: Option<Box<Node<Atom>>>,
}
impl<A> Node<A> {
/*    pub fn insert_at(&mut self, at: &Path, v: A) {
        let mut this = self;
        for i in at.iter() {
            match i {
                &PathEntry::Bit(side) => {
                    this = if side {
                        if this.right.is_none() {
                            this.right = Some(box Default::default());
                        }
                        &mut *this.right.as_mut().unwrap()
                    } else {
                        if this.left.is_none() {
                            this.left = Some(box Default::default());
                        }
                        &mut *this.left.as_mut().unwrap()
                    };
                },
                &PathEntry::Ambiguity(ref dis) => {
                    match this.kind {
                        None => {
                            this.kind = Some(Kind::Normal(dis.clone(), v));
                        }
                    }
                },
            }
        }
    }*/
}

#[derive(Clone, Debug)]
enum Kind<Atom> {
    Tombstone,
    Normal(Disambiguator, Atom),
    Major {
        minis: BTreeMap<Disambiguator, Node<Atom>>,
    },
}
impl<Atom> Kind<Atom> {
    pub fn is_major(&self) -> bool {
        match self {
            &Kind::Major { .. } => true,
            _ => false,
        }
    }
    pub fn get_node<'a>(&'a self, dis: &Disambiguator) -> Option<&'a Node<Atom>> {
        match self {
            &Kind::Normal(..) | &Kind::Tombstone => None,
            &Kind::Major { ref minis, } => minis.get(dis),
        }
    }

}

#[derive(Eq, Ord, PartialEq, PartialOrd, Clone, Copy, Debug)]
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
}
#[derive(Debug, Clone)]
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
    pub fn new_raw(prefix: Bitv, next: Option<(Disambiguator, Option<Box<Path>>)>) -> Path {
        Path {
            prefix: prefix,
            next: next,
        }
    }
    /// Assuming this Path starts at the root, return the height of the tree.
    pub fn height(&self) -> usize {
        let mut h = 0;
        // TODO count the prefix all at once.
        for i in self.iter() {
            match i {
                PathEntry::Bit(_) => {
                    h += 1;
                },
                _ => {},
            }
        }
        return h;
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

        unreachable!();
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
        /*fn at_node_kind<F>(at: &Path, state: &mut State<A>) {
            let mut this = self;
            let mut iter = at.iter().peekable();
            for i in iter {
                match i {
                    &PathEntry::Bit(side) => {
                        this = if side {
                            if this.right.is_none() {
                                this.right = Some(box Default::default());
                            }
                            &mut *this.right.as_mut().unwrap()
                        } else {
                            if this.left.is_none() {
                                this.left = Some(box Default::default());
                            }
                            &mut *this.left.as_mut().unwrap()
                        };
                    },
                    &PathEntry::Ambiguity(ref dis) => {
                        match this.kind {
                            None => {
                                this.kind = Some(Kind::Normal(dis.clone(), v));
                            }
                        }
                    },
                }
            }
        }

        fn insert() {
            let mut this = self;
            let mut iter = at.iter().peekable();
            for i in at.iter() {
                match i {
                    &PathEntry::Bit(side) => {
                        this = if side {
                            if this.right.is_none() {
                                this.right = Some(box Default::default());
                            }
                            &mut *this.right.as_mut().unwrap()
                        } else {
                            if this.left.is_none() {
                                this.left = Some(box Default::default());
                            }
                            &mut *this.left.as_mut().unwrap()
                        };
                    },
                    &PathEntry::Ambiguity(ref dis) => {
                        if iter.peek().is_some( {

                        match this.kind {
                            None => {
                                this.kind = Some(Kind::Normal(dis.clone(), v));
                            }
                        }
                    },
                }
            }
        }*/

        match self {
            &Op::Insert {
                at: ref at,
                node: ref n,
            } => {
                assert!(state.insert(at.clone(), n.clone()).is_none());
                Ok(())
            },
            &Op::Delete(ref at) => {
                assert!(state.remove(at).is_some());
                Ok(())
            },
        }
    }
}
//pub type State<A> = Node<A>;
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

    fn next_disambiguator(&mut self) -> Disambiguator {
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

    pub fn next_available_path_right(&self, prev: Path) -> Path {
        unimplemented!()
    }

    pub fn insert_at(&mut self, at: Path,
                     value: A) -> Result<(), TreedocError<L, A>> {
        unimplemented!()
    }
    pub fn insert_right(&mut self,
                        left: Path, value: A) -> Result<Path, TreedocError<L,A>> {
        let new_path = self.next_available_path_right(left);
        let op = Op::Insert {
            at: new_path.clone(),
            node: value,
        };
        self.update(op)
            .map(|()| new_path.clone() )
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod test {
    mod path {
        use super::super::*;
        use std::collections::Bitv;
        use std::cell::Cell;

        thread_local! { static DIS_COUNTER: Cell<u64> = Cell::new(0) }
        fn next_dis() -> Disambiguator {
            const SITE: SiteId = SiteId(20);
            Disambiguator::new_raw({
                DIS_COUNTER.with(|v| {
                    v.set(v.get() + 1);
                    v.get()
                })
            },
                                   SITE.clone())
        }

        fn debug_path(p: Path) -> Path {
            println!("");
            println!("path iters:");
            for i in p.iter() {
                println!("{:?}", i);
            }
            p
        }

        fn debug_paths(l: &Path, r: &Path) {
            println!("");
            println!("path iters:");
            let mut li = l.iter();
            let mut ri = r.iter();
            let mut i: usize = 0;
            loop {
                let l = li.next();
                let r = ri.next();
                println!("left  {}: {:?}", i, l);
                println!("right {}: {:?}", i, r);
                if l.is_none() && r.is_none() { break; }
                i += 1;
            }
        }

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
    }
}
