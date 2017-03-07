
# npy-rs &emsp; [![crates.io version](https://img.shields.io/crates/v/npy.svg)](https://crates.io/crates/npy) [![Documentation](https://docs.rs/npy/badge.svg)](https://docs.rs/npy/) ![Build Status](https://travis-ci.org/potocpav/npy-rs.svg?branch=master)

Numpy format (*.npy) serialization and deserialization.

<!-- [![Build Status](xxx)](xxx) -->


[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

To use **npy-rs**, two dependencies must be specified in `Cargo.toml`:

```toml
npy = "*"
npy-derive = "*"
```

Several examples are available in the [examples](examples) directory.

[Documentation](https://docs.rs/npy/)
