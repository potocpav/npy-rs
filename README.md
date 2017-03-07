
# npy-rs

Numpy format (*.npy) serialization and deserialization.

[![Build Status](xxx)](xxx)

[Documentation](http://xxx.com)

[NPY](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.
