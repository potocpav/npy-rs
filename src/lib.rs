#![warn(missing_docs)]
#![feature(slice_patterns)]

/*!
Serialize and deserialize the NumPy's *.npy binary format.

# Overview

NPY is a simple binary data format. It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple type-safe way to read and write
*.npy files. Files that can't fit into the memory are handled by using iterators instead of strict
data structures.

TODO: Data format spec

# Example

```
TODO: example
```

*/

extern crate byteorder;
#[macro_use]
extern crate nom;
extern crate memmap;

mod header;
mod readable;
mod writeable;
mod npy_data;

pub use readable::Readable;
pub use writeable::Writeable;
pub use npy_data::{NpyData, NpyIterator, from_bytes, to_file};


#[cfg(test)]
mod tests {
    use super::header::*;
    use super::header::Value::*;
    use super::nom::*;
    use super::Cursor;

    #[test]
    npy_data! {
        pub struct S {
            batchId: i32,
            hostHash: i64,
            user: i64,
            aggregate: f64,
            label: i8,
        }
    }


    #[test]
    fn parse_header() {
        assert_eq!(integer(b"1234  "), IResult::Done(&b""[..], Integer(1234)));
        assert_eq!(string(br#" "Hello"   "#), IResult::Done(&b""[..], String("Hello".into())));
        assert_eq!(string(br#" 'World!'   "#), IResult::Done(&b""[..], String("World!".into())));
        assert_eq!(boolean(b"  True"), IResult::Done(&b""[..], Bool(true)));
        assert_eq!(boolean(b"False "), IResult::Done(&b""[..], Bool(false)));
        assert_eq!(list(b" ()"), IResult::Done(&b""[..], List(vec![]))); // FIXME: Make this not parse as a List
        assert_eq!(list(b" (4)"), IResult::Done(&b""[..], List(vec![Integer(4)]))); // FIXME: Make this not parse as a List
        assert_eq!(list(b" (1 , 2 ,)"), IResult::Done(&b""[..], List(vec![Integer(1), Integer(2)])));
        assert_eq!(list(b" [5 , 6 , 7]"), IResult::Done(&b""[..], List(vec![Integer(5), Integer(6), Integer(7)])));
    }
    //
    // #[test]
    // fn from_file() {
    //     let file_mmap = Mmap::open_path("test/file.npy", Protection::Read).unwrap();
    //     let bytes: &[u8] = unsafe { file_mmap.as_slice() }; // No concurrent modification allowed
    //     let res: Vec<_> = S::from_bytes(bytes).unwrap().collect();
    //     println!("{:?}", res);
    // }
}
