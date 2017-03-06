extern crate npy;
extern crate memmap;

#[macro_use]
extern crate npy_derive;

use memmap::{Mmap, Protection};

#[allow(non_snake_case)]
#[derive(NpyData)]
pub struct Data {
    batchId: i32,
    hostHash: i64,
    user: i64,
    aggregate: f64,
    label: i8,
}

fn main() {
    let file_mmap = Mmap::open_path("test/file.npy", Protection::Read).unwrap();
    let bytes: &[u8] = unsafe { file_mmap.as_slice() }; // No concurrent modification allowed

    let res: Vec<Data> = npy::from_bytes(bytes).unwrap().collect();
    println!("{:?}", res.len());

    npy::to_file("test/output.npy", res.into_iter().take(10)).unwrap();
}
