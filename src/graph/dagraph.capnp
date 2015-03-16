@0x90719b7deb116556;

using import "../treedoc/treedoc.capnp".TreedocPath;

struct Uuid @0xb1966732b5a66dd2 {
  first  @0 :UInt64;
  second @1 :UInt64;
}

struct DAGraphEditOp @0x96c62819ae4adf6e {
  struct IdValRef {
    id  @0 :Uuid;
    val @1 :ValRef;

    struct ValRef {
      val  @0 :AnyPointer;
      refs @1 :UInt64;
    }
  }

  struct PosId {
    pos @0 :TreedocPath;
    id  @1 :Uuid;
  }

  union {
    insert :group {
      pos     @0 :TreedocPath;
      vertId @1 :Uuid;
      verts   @2 :List(IdValRef);
      order   @3 :List(PosId);
    }
    remove :group {
      pos @4 :List(TreedocPath);
    }
  }
}

struct DAGraphOp @0x9d92811484527964 {
  id @0 :Uuid;

  struct IdVal {
    vertId @0 :Uuid;
    value  @1 :AnyPointer;
  }
  struct PosId {
    pos    @0 :TreedocPath;
    vertId @1 :Uuid;
  }

  union {
    insert :group {
      verts @1 :List(IdVal);
      order @2 :List(PosId);
    }
    edit :group {
      op @3 :DAGraphEditOp;
    }
    remove :group {
      v @4 :Void;
    }
  }
}
