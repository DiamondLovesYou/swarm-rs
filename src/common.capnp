@0xc26dc2797a443b04;

struct Option @0xc8bc35acb71c3faa {
  union {
    some @0 :AnyPointer;
    none @1 :Void;
  }
}

struct SiteId {
  id @0 :UInt64;
}
