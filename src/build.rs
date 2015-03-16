extern crate capnpc;

use capnpc::compile;
use std::env::{current_dir, set_current_dir, var};
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

fn main() {
    let cwd = current_dir().unwrap();

    let out_dir = PathBuf::new(&var("OUT_DIR").unwrap()[..]);

    compile(&Path::new("src"),
            &[Path::new("src/common.capnp")])
        .unwrap();

    set_current_dir(&cwd).unwrap();
    create_dir_all(&out_dir.join("treedoc")).unwrap();

    compile(&Path::new("src"),
            &[Path::new("src/treedoc/treedoc.capnp")])
        .unwrap();

    set_current_dir(&cwd).unwrap();
    create_dir_all(&out_dir.join("graph")).unwrap();

    compile(&Path::new("src"),
            &[Path::new("src/graph/dagraph.capnp")])
        .unwrap();

    set_current_dir(&cwd).unwrap();
}
