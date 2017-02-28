// Temporary file we will be testing the new library by

#[macro_use]
extern crate npy;
extern crate rustc_serialize;
extern crate nom;
extern crate memmap;

use memmap::{Mmap, Protection};
use npy::NpyData;

npy_data! {
    pub struct Data {
        batchId: i32,
        hostHash: i64,
        user: i64,
        aggregate: f64,
        label: i8,
    }
}

fn main() {
    let file_mmap = Mmap::open_path("test/large-file.npy", Protection::Read).unwrap();
    let bytes: &[u8] = unsafe { file_mmap.as_slice() }; // No concurrent modification allowed

    let res: Vec<_> = Data::from_bytes(bytes).unwrap().collect();
    println!("{:?}", res.len());
}
