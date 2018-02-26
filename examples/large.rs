
extern crate memmap;
#[macro_use]
extern crate npy_derive;
extern crate npy;

use memmap::{Mmap, Protection};


#[derive(Serializable, Debug, Default)]
struct Array {
    a: i32,
    b: f32,
    c: i64,
}

// Use a memory map and iterators, so that this example works for files larger than can fit into RAM

fn main() {
    let file_mmap = Mmap::open_path("examples/simple.npy", Protection::Read).unwrap();
    let bytes: &[u8] = unsafe { file_mmap.as_slice() };
    let data = npy::NpyData::from_bytes(bytes).unwrap();

    let sum = data.into_iter().fold(Array::default(), |accum, arr: Array| {
        eprintln!("read: {:?}", arr);
        Array { a: accum.a + arr.a, b: accum.b + arr.b, c: accum.c + arr.c }
    });
    eprintln!("sum: {:?}", sum);
}
