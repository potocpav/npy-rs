#![warn(missing_docs)]

/*!
Serialize and deserialize the NumPy's
[*.npy binary format](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html).

# Overview

[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

Only one-dimensional arrays are supported. They may either be made of primitive numerical types
like integers or floating-point numbers or be [structured
arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html) that map to Rust structs.

To successfully import an array from NPY using the `#[derive(Serializable)]` mechanism, the target struct
must contain:

* corresponding number of fields in the same order,
* corresponding names of fields,
* compatible field types.

Currently, all the primitive numeric types and arrays of up to 16 elements are supported, though
they work only with little-endian. To deserialize other types or big-endian values, one must
manually implement [`Serializable`](trait.Serializable.html). A very common object that (right now)
requires a manual `impl` is a vector, as illustrated in
[an example](https://github.com/potocpav/npy-rs/tree/master/examples/vector.rs).

# Examples

More examples can be found in the [examples](https://github.com/potocpav/npy-rs/tree/master/examples)
directory.

Let's create a simple *.npy file in Python:

```python
import numpy as np
a = np.array([1, 3.5, -6, 2.3])
np.save('examples/plain.npy', a)
```

Now, we can load it in Rust:

```
extern crate npy;

use std::io::Read;
use npy::NpyData;

fn main() {
    let mut buf = vec![];
    std::fs::File::open("examples/plain.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    let data: NpyData<f64> = NpyData::from_bytes(&buf).unwrap();
    for arr in data {
        eprintln!("{:?}", arr);
    }
}
```

And we can see our data:

```text
1
3.5
-6
2.3
```

## Reading structs from record arrays

Let us move on to a slightly more complex task. We create a structured array in Python:

```python
import numpy as np
a = np.array([(1,2.5,4), (2,3.1,5)], dtype=[('a', 'i4'),('b', 'f4'),('c', 'i8')])
np.save('examples/simple.npy', a)
```

To load this in Rust, we need to create a corresponding struct, that derives `Serializable`. Make sure
the field names and types all match up:

```
#[macro_use]
extern crate npy_derive;
extern crate npy;

use std::io::Read;
use npy::NpyData;

#[derive(Serializable, Debug)]
struct Array {
    a: i32,
    b: f32,
    c: i64,
}

fn main() {
    let mut buf = vec![];
    std::fs::File::open("examples/simple.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    let data: NpyData<Array> = NpyData::from_bytes(&buf).unwrap();
    for arr in data {
        eprintln!("{:?}", arr);
    }
}
```

The output is:

```text
Array { a: 1, b: 2.5, c: 4 }
Array { a: 2, b: 3.1, c: 5 }
```

*/

extern crate byteorder;
#[macro_use]
extern crate nom;

mod header;
mod serializable;
mod npy_data;
mod out_file;

pub use serializable::Serializable;
pub use header::{DType, Field};
pub use npy_data::NpyData;
pub use out_file::{to_file, OutFile};

#[cfg(test)]
mod tests {
    // use super::header::*;
    // use super::header::Value::*;
    // use super::nom::*;

    // #[test]
    // #[derive(Serializable)]
    // struct S {
    //     batchId: i32,
    //     hostHash: i64,
    //     user: i64,
    //     aggregate: f64,
    //     label: i8,
    // }

    //
    // #[test]
    // fn from_file() {
    //     let file_mmap = Mmap::open_path("test/file.npy", Protection::Read).unwrap();
    //     let bytes: &[u8] = unsafe { file_mmap.as_slice() }; // No concurrent modification allowed
    //     let res: Vec<_> = S::from_bytes(bytes).unwrap().collect();
    //     eprintln!("{:?}", res);
    // }
}
