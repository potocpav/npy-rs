
# npy-rs
[![crates.io version](https://img.shields.io/crates/v/npy.svg)](https://crates.io/crates/npy) [![Documentation](https://docs.rs/npy/badge.svg)](https://docs.rs/npy/) [![Build Status](https://travis-ci.org/potocpav/npy-rs.svg?branch=master)](https://travis-ci.org/potocpav/npy-rs)

Numpy format (*.npy) serialization and deserialization.

<!-- [![Build Status](xxx)](xxx) -->


[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

## Usage

To use **npy-rs**, two dependencies must be specified in `Cargo.toml`:

```toml
npy = "0.3"
npy-derive = "0.3"
```

The second dependency implements the custom `derive` macro. A typical way to import everything needed is:

```rust
#[macro_use]
extern crate npy_derive;
extern crate npy;
```

Several usage examples are available in the [examples](examples) directory; the [simple](examples/simple.rs) example shows how to load a file, [roundtrip](examples/roundtrip.rs) shows both reading and writing.

[Documentation](https://docs.rs/npy/)

## Performance

Version 0.3 brought ten-fold performance improvements. On my laptop, it now loads and writes files from a ramdisk at approx. 700 MB/s.

Only the header is parsed on the `NpyData::from_bytes` call. The data can then be accessed sequentially by iterating over `NpyData`, randomly by using the `get` function, or the whole file can be deserialized into a `Vec` at once by using the `to_vec` function. Only the third option requires the whole file to fit into the RAM at once.

To load large files, they can be memory-mapped as illustrated in the [large example](examples/large.rs).
