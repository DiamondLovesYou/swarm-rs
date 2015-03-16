@0x8a940bb8df1ef1c7;

using import "../common.capnp".SiteId;

struct Disambiguator @0xedc3b70a81582bf5 {
  siteId        @0 :SiteId;
  disambiguator @1 :UInt64;
}

struct TreedocPath @0xf13d96a1fde4f2e1 {
  struct Part {
    bits          @0 :List(Bool);
    disambiguator @1 :Disambiguator;
  }
  parts @0 :List(Part);
}
