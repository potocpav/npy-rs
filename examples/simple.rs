#[macro_use]
extern crate npy_derive;
extern crate npy;

use std::io::Read;

#[derive(NpyData, Debug)]
struct Array {
    a: i32,
    b: f32,
    c: i64,
}

fn main() {
    let mut buf = vec![];
    std::fs::File::open("examples/simple.npy").unwrap()
        .read_to_end(&mut buf).unwrap();
        
    for arr in npy::from_bytes::<Array>(&buf).unwrap() {
        println!("{:?}", arr);
    }
}
