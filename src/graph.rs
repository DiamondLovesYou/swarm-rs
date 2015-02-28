
// A collection of graph CRDTs.
use std::collections::hash_map;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::default::Default;
use std::fmt;

use uuid::Uuid;

use super::treedoc;
pub use super::treedoc::Side;
use super::{NullError, SiteId};

// First up, a DAG type. Because a DAG is acyclic, we store the
// graph as a set of treedocs, one for each path through the graph. Every vertex
// gets its own Uuid, assigned by node which initially created it. To insert a
// new vertex+associated data into the graph, the vertex must be connected in a
// path. Conversely, to delete a vertex+associate data, each path referencing
// said vertex must be edited to remove that vertex. When inserting into an
// existing Path, the corresponding op is mapped to include the inserted node's
// value. This datatype is both operation and state based: inserts and edits send the
// value of every vertex in the affected path in the operation. Inserts
// win. Edit level removes are overriden by complete path removal when both
// happen concurrently.
//
// Note this implementation doesn't completely disallow cycles; only infinite
// cycles are disallowed, though not by checks: it's impossible to represent
// such via paths through the graph.

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct NullLog(SiteId);

impl<O> super::Log<O> for NullLog {
    type Error = NullError;
    fn get_site_id(&self) -> SiteId {
        let &NullLog(ref id) = self;
        id.clone()
    }
    fn apply_downstream(&mut self, _op: O) -> Result<(), NullError> {
        Ok(())
    }
    fn add_to_log(&mut self, _op: O) -> Result<(), NullError> {
        Ok(())
    }
}

pub type PathId = Uuid;
pub type VertId = Uuid;

pub type Path = treedoc::Treedoc<NullLog, Uuid>;

#[derive(Clone, Debug)]
pub enum EditOp<A> {
    Insert {
        pos: treedoc::Path,
        vert: VertId,

        /// No duplicates allowed! The usize in the inner tuple specifies the
        /// number of references this path will add to the vertice's reference
        /// count.
        verts: Vec<(VertId, (A, usize))>,

        /// Doesn't have to be in order, but each path should be
        /// unique. Duplicate vertex ids are allowed.
        order: Vec<(treedoc::Path, VertId)>,
    },
    RemoveOne(treedoc::Path),
    RemoveMultiple(Vec<treedoc::Path>),
}
#[derive(Clone, Debug)]
pub enum Op<A> {
    Insert {
        id: PathId,
        /// It's assumed no duplicate entries exist in `verts`.
        verts: Vec<(Uuid, A)>,

        path: Path,
    },
    Edit {
        path_id: PathId,

        op: EditOp<A>,
    },
    Remove(PathId),
}

pub struct DAGraphIter<'a, 'b, L, A> where 'a: 'b, L: 'a, A: 'a {
    graph: &'a DAGraph<L, A>,
    paths_iter: hash_map::Iter<'a, Uuid, Path>,
    /// The first element is the uuid of the path we're currently iterating
    /// through.
    path_iter: Option<(Uuid, treedoc::Iter<'b, Uuid>)>,
}
impl<'a, 'b, L, A> Iterator for DAGraphIter<'a, 'b, L, A> {
    type Item = (Uuid, Uuid, &'b treedoc::Path, &'b A);
    fn next(&mut self) -> Option<(Uuid, Uuid, &'b treedoc::Path, &'b A)> {
        loop {
            let path = self.path_iter
                .as_mut()
                .and_then(|&mut (ref id, ref mut iter)| {
                    iter.next()
                        .map(|(pos, v)| (id.clone(), pos, v) )
                });
            if path.is_none() {
                // move on to the next path in the graph:
                let next_path = self.paths_iter.next();
                if next_path.is_none() { return None; }
                let path_iter = next_path
                    .map(|(pid, path)| (pid.clone(), path.iter()) );
                self.path_iter = path_iter;
                continue;
            }
            let (path_id, pos, vert_id) = path.unwrap();
            let val_opt = self.graph.verts.get(&vert_id);
            debug_assert!(val_opt.is_some(), "Internal check failed: a vertex \
                                              should never have no value!");
            let &(ref val_opt, _) = val_opt.unwrap();
            return Some((path_id,
                         vert_id.clone(),
                         pos,
                         val_opt));
        }
    }
}
pub struct DAGraphInputVertIter<'a, L, A> where L: 'a, A: 'a {
    graph: &'a DAGraph<L, A>,
    paths_iter: hash_map::Iter<'a, Uuid, Path>,
}
impl<'a, L, A> Iterator for DAGraphInputVertIter<'a, L, A> {
    type Item = (Uuid, Uuid, &'a A);
    fn next(&mut self) -> Option<(Uuid, Uuid, &'a A)> {
        let path = self.paths_iter.next();
        path.map(|(id, path)| {
            let input_id = path.values()
                .next()
                .expect("unexpected invalid, empty path");

            let &(ref vert, _) = self.graph
                .verts
                .get(input_id)
                .expect("invalid vertex id");
            (id.clone(), input_id.clone(), vert)
        })
    }
}

#[derive(Debug)]
pub enum Error<L, A>
    where L: super::Log<Op<A>>,
          <L as super::Log<Op<A>>>::Error: fmt::Debug,
{
    Log(<L as super::Log<Op<A>>>::Error),
    InvalidPath(PathId),
    InvalidVert(VertId),
    DuplicateEdge(treedoc::Path),
}
pub type GraphResult<R, L, A> = Result<R, Error<L, A>>;

#[derive(Clone)]
pub struct DAGraph<L, A> {
    log: L,
    paths: HashMap<Uuid, Path>,

    // the vertices and their values.
    verts: HashMap<Uuid, (A, usize)>,
}

pub enum InsertionValue<A> {
    Preexisting(Uuid),
    New(A),
}

impl<L, A> DAGraph<L, A>
    where L: super::Log<Op<A>>,
          A: Clone,
          <L as super::Log<Op<A>>>::Error: fmt::Debug
{
    pub fn new_with_log(log: L) -> DAGraph<L, A> {
        DAGraph {
            log: log,
            paths: Default::default(),

            verts: Default::default(),
        }
    }

    pub fn log_imm<'a>(&'a self) -> &'a L { &self.log }
    pub fn log_mut<'a>(&'a mut self) -> &'a mut L { &mut self.log }

    pub fn insert_path(&mut self, path: &[InsertionValue<A>]) ->
        GraphResult<(Uuid, Vec<(Uuid, treedoc::Path)>), L, A>
    {
        let id = Uuid::new_v4();

        let mut fp = Vec::new();
        for v in path.iter() {
            let insert = match v {
                &InsertionValue::Preexisting(ref id) => {
                    let v = self.verts.get(id);
                    if v.is_none() {
                        return Err(Error::InvalidVert(id.clone()));
                    }
                    let &(ref v, _) = v.unwrap();
                    (id.clone(), v.clone())
                }
                &InsertionValue::New(ref v) => {
                    // Don't insert yet; wait till we've successfully written
                    // to the log.
                    (Uuid::new_v4(), v.clone())
                }
            };
            fp.push(insert);
        }
        let site_id = self.log.get_site_id().clone();
        let mut p = treedoc::Treedoc::new(NullLog(site_id.clone()));
        let mut res: Vec<(Uuid, treedoc::Path)> = Vec::new();
        res.reserve(fp.len());
        let mut last_path = None;
        for &(ref id, _) in fp.iter() {
            let id = id.clone();
            match last_path {
                None => {
                    last_path = p.insert_at(None, id.clone())
                        .ok();
                },
                Some(lp) => {
                    last_path = p.insert_right(lp, id.clone())
                        .ok();
                },
            }
            debug_assert!(last_path.is_some(),
                          "internal error: insert failed, but this \
                           shouldn't happen");

            res.push((id, last_path.clone().unwrap()));
        }

        let op = Op::Insert {
            id: id.clone(),
            verts: fp.clone(),
            path: p.clone(),
        };
        let logres = self.log
            .apply_downstream(op);
        let logres = logres.map_err(move |err| Error::Log(err) );
        if logres.is_err() { return Err(logres.err().unwrap()); }

        for (id, v) in fp.into_iter() {
            match self.verts.entry(id) {
                Entry::Vacant(ve) => {
                    ve.insert((v, 1));
                },
                Entry::Occupied(oe) => {
                    let &mut (_, ref mut rc) = oe.into_mut();
                    *rc += 1;
                },
            }
        }
        assert!(self.paths.insert(id.clone(), p).is_none());
        return Ok((id, res));
    }

    pub fn insert_vert_path(&mut self, path: Uuid, parent: Option<treedoc::Path>,
                            vert: InsertionValue<A>,
                            side: Side) -> GraphResult<(Uuid, treedoc::Path), L, A> {

        let p = self.paths.get_mut(&path);
        let p = if p.is_none() {
            return Err(Error::InvalidPath(path));
        } else {
            p.unwrap()
        };

        let (vert_id, vert_val) = match vert {
            InsertionValue::Preexisting(id) => {
                let v = self.verts.get(&id);
                if v.is_none() { return Err(Error::InvalidVert(id)); }
                let &(ref val, _) = v.unwrap();
                (id, val.clone())
            },
            InsertionValue::New(val) => {
                let id = Uuid::new_v4();
                (id, val)
            },
        };

        let path_pos = p.get_next_empty_path(parent, side);

        let mut verts: HashMap<Uuid, (A, usize)> = HashMap::new();
        verts.reserve(p.len());
        let mut order = Vec::new();
        let mut path_pos_insert = Some(path_pos.clone());
        for (pos, vid) in p.iter() {
            let v = self.verts.get(vid);
            debug_assert!(v.is_some(), "invalid vert id present in path");
            let &(ref v, _) = v.unwrap();
            match verts.entry(vid.clone()) {
                Entry::Vacant(ve) => {
                    ve.insert((v.clone(), 1));
                },
                Entry::Occupied(oe) => {
                    let &mut (_, ref mut rc) = oe.into_mut();
                    *rc += 1;
                },
            }

            order.push((pos.clone(), vid.clone()));
            if path_pos_insert.is_some() && pos > &path_pos {
                order.push((path_pos_insert.take().unwrap(),
                            vert_id.clone()));
            }
        }
        match verts.entry(vert_id.clone()) {
            Entry::Vacant(ve) => {
                ve.insert((vert_val.clone(), 1));
            },
            Entry::Occupied(oe) => {
                let &mut (_, ref mut rc) = oe.into_mut();
                *rc += 1;
            },
        }

        let op = Op::Edit {
            path_id: path,
            op: EditOp::Insert {
                vert: vert_id.clone(),
                pos: path_pos.clone(),


                verts: verts.into_iter().collect(),
                order: order,
            },
        };

        let logres = self.log.apply_downstream(op);
        if logres.is_err() { return Err(Error::Log(logres.err().unwrap())); }

        p.insert_at(Some(path_pos.clone()), vert_id.clone())
            .unwrap();

        match self.verts.entry(vert_id.clone()) {
            Entry::Occupied(oe) => {
                let &mut (_, ref mut rc) = oe.into_mut();
                *rc += 1;
            }
            Entry::Vacant(ve) => {
                ve.insert((vert_val, 1));
            }
        }

        Ok((vert_id, path_pos))
    }

    /// Empty positions in the path are ignored. Deplicate positions cause an error.
    pub fn remove_path_verts(&mut self,
                             pid: PathId,
                             verts: &[treedoc::Path]) ->
        GraphResult<(), L, A>
    {

        let p = self.paths.get_mut(&pid);
        let p = if p.is_none() {
            return Err(Error::InvalidPath(pid));
        } else {
            p.unwrap()
        };

        let mut values: HashMap<Uuid, (A, usize)> =
            HashMap::new();
        values.reserve(verts.len());
        {
            let mut positions = HashSet::new();
            for pos in verts.iter() {
                if !positions.insert(pos) {
                    return Err(Error::DuplicateEdge(pos.clone()));
                }

                let id = p.get(pos);
                if id.is_none() { continue; }
                let id = id.unwrap();

                let val = self.verts.get(id)
                    .map(move |&(ref v, _)| v.clone() );

                debug_assert!(val.is_some(), "invalid vertex id!");
                let val = val.unwrap();

                match values.entry(id.clone()) {
                    Entry::Vacant(ve) => {
                        ve.insert((val, 1));
                    },
                    Entry::Occupied(oe) => {
                        let &mut (_, ref mut rc) = oe.into_mut();
                        *rc += 1;
                    },
                }
            }
        }

        let op = Op::Edit {
            path_id: pid.clone(),

            op: if verts.len() == 1 {
                EditOp::RemoveOne(verts[0].clone())
            } else {
                let vec = verts.iter()
                    .map(move |pos| pos.clone() )
                    .collect();
                EditOp::RemoveMultiple(vec)
            },
        };

        let logres = self.log.apply_downstream(op);
        if logres.is_err() { return Err(Error::Log(logres.err().unwrap())); }

        for pos in verts.iter() {
            p.remove(pos.clone()).unwrap();
        }

        for (id, (_, derefs)) in values.into_iter() {
            match self.verts.entry(id) {
                Entry::Vacant(..) => unreachable!(),
                Entry::Occupied(oe) => {
                    let &(_, refs) = oe.get();
                    assert!(derefs <= refs);
                    if refs - derefs == 0 {
                        oe.remove();
                        continue;
                    } else {
                        let &mut (_, ref mut rc) = oe.into_mut();
                        *rc -= derefs;
                    }
                },
            }
        }

        Ok(())
    }

    pub fn remove_path(&mut self, id: &Uuid) -> GraphResult<(), L, A> {
        {
            let p = self.paths.get(id);
            let p = if p.is_none() {
                return Err(Error::InvalidPath(id.clone()));
            } else {
                p.unwrap()
            };

            let op = Op::Remove(id.clone());
            let res = self.log.apply_downstream(op);
            if res.is_err() { return Err(Error::Log(res.err().unwrap())); }

            for id in p.values() {
                match self.verts.entry(id.clone()) {
                    Entry::Vacant(_) => {
                        unreachable!();
                    },
                    Entry::Occupied(oe) => {
                        let &(_, rc) = oe.get();
                        if rc == 1 {
                            oe.remove();
                        } else {
                            let &mut (_, ref mut rc) = oe.into_mut();
                            *rc -= 1;
                        }
                    },
                }
            }
        }

        self.paths.remove(&id);
        Ok(())
    }

    pub fn get_vertex(&self, path: PathId, pos: &treedoc::Path) -> Option<&A> {
        self.paths.get(&path)
            .and_then(move |p| {
                p.get(pos)
            })
            .and_then(move |vert_id| {
                self.verts.get(vert_id)
                    .map(move |&(ref val, _)| val)
            })
    }

    pub fn paths_len(&self) -> usize { self.paths.len() }
    pub fn verts_len(&self) -> usize { self.verts.len() }

    pub fn contains_path(&self, id: &PathId) -> bool {
        self.paths.contains_key(id)
    }
    pub fn contains_vert(&self, id: &VertId) -> bool {
        self.verts.contains_key(id)
    }

    /// An iterator over only the first vertex in every path. The paths will not
    /// be in any order.
    /// ```rust
    /// for (path_id, input_vert_id, vert_val_ref) in graph.input_iter() {
    /// }
    /// ```
    pub fn input_iter(&self) -> DAGraphInputVertIter<L, A> {
        DAGraphInputVertIter {
            graph: self,
            paths_iter: self.paths.iter(),
        }
    }

    /// An iterator over every vertex in every path. Paths will not be in order,
    /// but the vertices will be in order relative to the current path.
    /// ```rust
    /// for (path_id, vert_id, vert_val_ref) in graph.iter() {
    /// }
    /// ```
    pub fn iter(&self) -> DAGraphIter<L, A> {
        DAGraphIter {
            graph: self,
            paths_iter: self.paths.iter(),
            path_iter: None,
        }
    }

    pub fn deliver(&mut self, op: Op<A>) {
        match op {
            Op::Insert { id, verts, path, } => {
                assert!(!self.paths.contains_key(&id),
                        "path already present; was this operation delivered \
                         twice?");

                let mut vert_map: Option<HashMap<VertId, A>> = None;
                fn get_verts<'a, A>(verts: &Vec<(VertId, A)>,
                                    vert_map: &'a mut Option<HashMap<VertId, A>>) ->
                    &'a HashMap<VertId, A> where A: Clone
                {
                    // Lazy initialize the map:
                    match vert_map {
                        &mut Some(ref v) => v,
                        &mut None => {
                            let verts = verts.iter()
                                .map(|&(ref id, ref v)| (id.clone(), v.clone()) )
                                .collect();
                            *vert_map = Some(verts);
                            match vert_map {
                                &mut Some(ref v) => v,
                                _ => unreachable!(),
                            }
                        },
                    }
                };

                self.paths.insert(id.clone(), path.clone());

                for (_, vert) in path.iter() {
                    match self.verts.entry(vert.clone()) {
                        Entry::Vacant(ve) => {
                            let val = get_verts(&verts, &mut vert_map)
                                .get(vert)
                                .expect("the operation is missing a vert!");
                            ve.insert((val.clone(), 1));
                        },
                        Entry::Occupied(oe) => {
                            let &mut (_, ref mut rc) = oe.into_mut();
                            *rc += 1;
                        },
                    }
                }
            },
            Op::Edit {
                path_id,
                op: EditOp::Insert {
                    pos, vert: new_vert_id, verts, order,
                },
            } => {
                match self.paths.entry(path_id) {
                    Entry::Vacant(ve) => {
                        // info!("path removed concurrently; re-inserting.");
                        // In this case, we ignore `new_vert_id` && `pos`; they
                        // both should already be in `verts` && `order`.

                        // Insert any new vertices:
                        for (vid, (val, count)) in verts.into_iter() {
                            assert!(count > 0);
                            match self.verts.entry(vid) {
                                Entry::Vacant(ve) => {
                                    ve.insert((val, count));
                                },
                                Entry::Occupied(oe) => {
                                    // In this case, ignore the value in the op.
                                    let &mut (_, ref mut rc) = oe.into_mut();
                                    *rc += count;
                                },
                            }
                        }


                        let mut p = {
                            let site_id = self.log.get_site_id();
                            treedoc::Treedoc::new(NullLog(site_id))
                        };

                        for (position, vid) in order.into_iter() {
                            debug_assert!(self.verts.contains_key(&vid));
                            p.insert_at(Some(position), vid)
                                .unwrap();
                        }

                        ve.insert(p);
                    },
                    Entry::Occupied(oe) => {
                        // Ignore `order`.

                        let mut val = None;
                        for (id, (v, c)) in verts.into_iter() {
                            if id == new_vert_id {
                                assert!(c == 1);
                                val = Some(v);
                                break;
                            }
                        }

                        assert!(val.is_some(), "operation error: operation is
                                                missing requisite vertex!");

                        match self.verts.entry(new_vert_id.clone()) {
                            Entry::Vacant(ve) => {
                                ve.insert((val.unwrap(), 1));
                            }
                            Entry::Occupied(oe) => {
                                let &mut (_, ref mut rc) = oe.into_mut();
                                *rc += 1;
                            }
                        }

                        let p = oe.into_mut();
                        p.insert_at(Some(pos), new_vert_id)
                            .unwrap();
                    },
                }
            },

            Op::Edit {
                path_id,
                op: EditOp::RemoveOne(pos),
            } => {
                let p = self.paths.get_mut(&path_id);
                if p.is_none() { return; }
                let mut p = p.unwrap();

                let vert_id = {
                    let vert_id_opt = p.get(&pos);
                    assert!(vert_id_opt.is_some());
                    vert_id_opt
                        .unwrap().clone()
                };

                match self.verts.entry(vert_id.clone()) {
                    Entry::Vacant(_) => unreachable!(),
                    Entry::Occupied(mut oe) => {
                        let &(_, rc) = oe.get();
                        if rc == 1 {
                            oe.remove();
                        } else {
                            let &mut (_, ref mut rc) = oe.get_mut();
                            *rc -= 1;
                        }
                    },
                }

                p.remove(pos).unwrap();
            },

            Op::Edit {
                path_id,
                op: EditOp::RemoveMultiple(poses),
            } => {
                let p = self.paths.get_mut(&path_id);
                if p.is_none() { return; }
                let p = p.unwrap();

                for pos in poses.into_iter() {

                    let vert_id = {
                        let vert_id_opt = p.get(&pos);
                        assert!(vert_id_opt.is_some());
                        vert_id_opt
                            .unwrap().clone()
                    };

                    match self.verts.entry(vert_id.clone()) {
                        Entry::Vacant(_) => unreachable!(),
                        Entry::Occupied(mut oe) => {
                            let &(_, rc) = oe.get();
                            if rc == 1 {
                                oe.remove();
                            } else {
                                let &mut (_, ref mut rc) = oe.get_mut();
                                *rc -= 1;
                            }
                        },
                    }

                    p.remove(pos).unwrap();
                }
            },

            Op::Remove(path_id) => {
                self.paths.remove(&path_id);
            }
        }
    }
}

impl<L, A: fmt::Debug + Clone> fmt::Debug for DAGraph<L, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "DAGraph {{ verts = `{:?}`, \n paths = `{:?}` }}",
                 self.verts, self.paths)
    }
}

#[cfg(test)]
mod test {
    impl_log_for_state! { super::Op<u64> }
    mod dag {
        use treedoc;
        use uuid::Uuid;
        use test_helpers::*;
        use super::super::*;
        type TestDag = DAGraph<DumbLog<Op<u64>>, u64>;
        fn new_dag() -> TestDag {
            DAGraph::new_with_log(new_dumb_log!())
        }
        fn new_path() -> Path {
            use Log;

            let l: DumbLog<Op<u64>> = new_dumb_log!();
            let sid = l.get_site_id();

            treedoc::Treedoc::new(NullLog(sid))
        }

        #[test]
        fn inserts() {
            let mut l = new_dag();
            let v1 = vec![InsertionValue::New(6u64),
                          InsertionValue::New(7u64),];
            let (path_id, verts): (_, Vec<(_, _)>) = l.insert_path(&v1[..])
                .ok()
                .unwrap();
            assert_eq!(verts.len(), 2);

            let vert_ids: Vec<VertId> = verts.iter()
                .map(move |&(ref id, _)| id.clone() )
                .collect();
            let vert_pos: Vec<treedoc::Path> = verts.into_iter()
                .map(move |(_, p)| p )
                .collect();

            let mut iter = l.iter();
            assert_eq!(iter.next().unwrap(),
                       (path_id.clone(), vert_ids[0].clone(),
                        &vert_pos[0], &6u64));
            assert_eq!(iter.next().unwrap(),
                       (path_id.clone(), vert_ids[1].clone(),
                        &vert_pos[1], &7u64));
            assert_eq!(iter.next(), None);

            assert_eq!(l.log_imm().downstreamed(), 1);
        }
        #[test]
        fn insert_remove() {
            let mut l = new_dag();
            let v1 = vec![InsertionValue::New(6u64),
                          InsertionValue::New(7u64),];
            let (id, _) = l.insert_path(&v1[..])
                .ok()
                .unwrap();
            assert!(l.remove_path(&id).is_ok());
            assert!(!l.contains_path(&id));

            assert_eq!(l.log_imm().downstreamed(), 2);
        }

        #[test]
        fn verts_are_removed() {
            let mut l = new_dag();
            let (id, _) = l.insert_path(&[InsertionValue::New(5u64)])
                .ok()
                .unwrap();
            assert!(l.remove_path(&id).is_ok());
            assert_eq!(l.verts_len(), 0);

            assert_eq!(l.log_imm().downstreamed(), 2);
        }

        #[test]
        fn verts_are_removed_b() {
            let mut l = new_dag();

            let (_, v1) = l.insert_path(&[InsertionValue::New(5u64)])
                .ok()
                .unwrap();
            let &(ref vid1, _) = &v1[0];

            let (id2, v2) =
                l.insert_path(&[InsertionValue::New(6u64),
                                InsertionValue::Preexisting(vid1.clone())])
                .ok()
                .unwrap();
            let &(ref vid2, _) = &v2[0];

            assert_eq!(l.paths_len(), 2);
            assert_eq!(l.verts_len(), 2);
            l.remove_path(&id2)
                .ok()
                .unwrap();
            assert_eq!(l.verts_len(), 1);
            assert!(l.contains_vert(&vid1));
            assert!(!l.contains_vert(&vid2));

            assert_eq!(l.log_imm().downstreamed(), 3);
        }
        #[test]
        fn insert_path_vert() {
            let mut l = new_dag();
            let (id, v) = l.insert_path(&[InsertionValue::New(5),
                                          InsertionValue::New(6),
                                          InsertionValue::New(8),])
                .ok()
                .unwrap();
            assert_eq!(v.len(), 3);
            let &(_, ref p) = &v[1];
            l.insert_vert_path(id, Some(p.clone()),
                               InsertionValue::New(7),
                               Side::Right)
                .ok()
                .unwrap();

            println!("l = `{:?}`", l);
            let mut iter = l.iter()
                .map(move |(_, _, _, v)| v.clone() );
            assert_eq!(iter.next(), Some(5));
            assert_eq!(iter.next(), Some(6));
            assert_eq!(iter.next(), Some(7));
            assert_eq!(iter.next(), Some(8));

            assert_eq!(l.log_imm().downstreamed(), 2);
        }

        #[test]
        fn subgraph_finite_cycles() {
            let mut l = new_dag();
            let (pid0, v) = l.insert_path(&[InsertionValue::New(5),
                                            InsertionValue::New(6),])
                .ok()
                .unwrap();
            let &(ref id0, _) = &v[0];
            let &(ref id1, _) = &v[1];

            let (pid1, _) = l.insert_path(&[InsertionValue::Preexisting(id0.clone()),
                                            InsertionValue::Preexisting(id1.clone()),
                                            InsertionValue::Preexisting(id0.clone()),
                                            InsertionValue::Preexisting(id1.clone()),])
                .ok()
                .unwrap();

            l.remove_path(&pid0).ok().unwrap();
            l.remove_path(&pid1).ok().unwrap();

            assert_eq!(l.verts_len(), 0);
            assert_eq!(l.paths_len(), 0);
        }

        #[test]
        fn remove_path_verts_a() {
            let mut l = new_dag();
            let (pid, v) = l.insert_path(&[InsertionValue::New(5),
                                           InsertionValue::New(6),
                                           InsertionValue::New(9),
                                           InsertionValue::New(10),
                                           InsertionValue::New(7),])
                .ok()
                .unwrap();
            let &(_, ref pos1) = &v[2];
            let &(_, ref pos2) = &v[3];
            l.remove_path_verts(pid, &[pos1.clone(), pos2.clone()])
                .unwrap();

            assert_eq!(l.verts_len(), 3);

            let values = l.iter()
                .map(|(_, _, _, &val)| val )
                .collect();
            assert_eq!(values, vec![5, 6, 7]);
        }

        #[test]
        fn remove_path_verts_b() {
            let mut l = new_dag();
            let (pid, v) = l.insert_path(&[InsertionValue::New(5),]).unwrap();
            let &(_, ref val) = &v[0];
            assert!(match l.remove_path_verts(pid, &[val.clone(),
                                                     val.clone(),]) {
                Err(Error::DuplicateEdge(ref edge)) if edge == val => true,
                _ => false,
            });
        }

        #[test]
        fn deliver_insert() {
            let mut c = new_dag();
            let id = Uuid::new_v4();
            let vert_id = Uuid::new_v4();
            let verts = vec![(vert_id.clone(), 5u64)];
            let mut path = new_path();

            path.insert_at(None, vert_id)
                .unwrap();

            let op = Op::Insert {
                id: id,
                verts: verts,
                path: path,
            };
            c.deliver(op);
        }
        #[test] #[should_fail]
        fn deliver_insert_fail_a() {
            // ensure that invalid things are invalid.

            let mut c = new_dag();
            let id = Uuid::new_v4();
            let vert_id = Uuid::new_v4();
            let verts = vec![];
            let mut path = new_path();

            path.insert_at(None, vert_id)
                .unwrap();

            let op = Op::Insert {
                id: id,
                verts: verts,
                path: path,
            };
            c.deliver(op);
        }

        fn check_values(v: TestDag, values: Vec<u64>) {
            for ((_, _, _, &r), l) in v.iter().zip(values.into_iter()) {
                assert_eq!(r, l);
            }
        }

        #[test]
        fn deliver_edit_a() {
            let mut l = new_dag();
            let (id, v) = l.insert_path(&[InsertionValue::New(5),
                                          InsertionValue::New(6),
                                          InsertionValue::New(8),])
                .ok()
                .unwrap();

            let mut r = l.clone();

            let &(_, ref p) = &v[1];
            l.insert_vert_path(id, Some(p.clone()),
                               InsertionValue::New(7),
                               Side::Right)
                .ok()
                .unwrap();

            r.deliver(l.log_imm().log[1].clone());
            check_values(r, vec![5, 6, 7, 8]);
        }
        #[test]
        fn deliver_edit_remove_single() {
            let mut l = new_dag();
            let (id, v) = l.insert_path(&[InsertionValue::New(5),
                                          InsertionValue::New(6),
                                          InsertionValue::New(7),
                                          InsertionValue::New(8),])
                .ok()
                .unwrap();

            let mut r = l.clone();

            let &(_, ref p1) = &v[1];
            let remove = vec![p1.clone()];
            l.remove_path_verts(id, &remove[..])
                .ok()
                .unwrap();

            r.deliver(l.log_imm().log[1].clone());
            check_values(r, vec![5, 7, 8]);
        }
        #[test]
        fn deliver_edit_remove_multiple() {
            let mut l = new_dag();
            let (id, v) = l.insert_path(&[InsertionValue::New(5),
                                          InsertionValue::New(6),
                                          InsertionValue::New(7),
                                          InsertionValue::New(8),])
                .ok()
                .unwrap();

            let mut r = l.clone();

            let &(_, ref p1) = &v[1];
            let &(_, ref p2) = &v[2];
            let remove = vec![p1.clone(), p2.clone()];
            l.remove_path_verts(id, &remove[..])
                .ok()
                .unwrap();

            r.deliver(l.log_imm().log[1].clone());
            check_values(r, vec![5, 8]);
        }
        #[test]
        fn deliver_edit_insert_concurrent_remove() {
            let mut l = new_dag();
            let (id, v) = l.insert_path(&[InsertionValue::New(5),
                                          InsertionValue::New(6),
                                          InsertionValue::New(8),])
                .ok()
                .unwrap();

            let mut r = l.clone();

            let &(_, ref p) = &v[1];
            l.insert_vert_path(id.clone(), Some(p.clone()),
                               InsertionValue::New(7),
                               Side::Right)
                .ok()
                .unwrap();

            r.remove_path(&id)
                .unwrap();

            r.deliver(l.log_imm().log[1].clone());
            check_values(r, vec![5, 6, 7, 8]);
        }
        #[test]
        fn deliver_remove() {
            let mut l = new_dag();
            let (id, _) = l.insert_path(&[InsertionValue::New(5),
                                          InsertionValue::New(6),
                                          InsertionValue::New(8),])
                .ok()
                .unwrap();

            l.deliver(Op::Remove(id));
        }

        #[test]
        fn deliver_double_concurrent_remove() {
            // double concurrent removes.
            let mut l = new_dag();

            l.deliver(Op::Remove(Uuid::new_v4()));
        }

        #[test] #[should_fail]
        fn deliver_insert_twice() {
            let mut c = new_dag();
            let id = Uuid::new_v4();
            let vert_id = Uuid::new_v4();
            let verts = vec![(vert_id.clone(), 5u64)];
            let mut path = new_path();

            path.insert_at(None, vert_id)
                .unwrap();

            let op = Op::Insert {
                id: id,
                verts: verts,
                path: path,
            };
            c.deliver(op.clone());
            c.deliver(op);
        }
    }
}
