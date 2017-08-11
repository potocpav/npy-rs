#![feature(test)]

#[macro_use]
extern crate npy_derive;
extern crate npy;
extern crate test;

use npy::NpyRecord;
use test::Bencher;
use test::black_box as bb;

#[derive(NpyRecord, Debug, PartialEq)]
struct Array {
    a: i32,
    b: f32,
}

const NITER: usize = 100_000;

fn test_data() -> Vec<u8> {
    let mut raw = Vec::new();
    for i in 0..NITER {
        let arr = Array { a: i as i32, b: i as f32 };
        arr.write(&mut raw).unwrap();
    }
    raw
}

#[bench]
fn read_little_endian(b: &mut Bencher) {
    let raw = test_data();
    b.iter(|| {
        for i in 0..NITER {
            bb(Array::read(&raw[i*8..]));
        }
    });
}

#[bench]
fn write_little_endian(b: &mut Bencher) {
    b.iter(|| {
        bb(test_data())
    });
}
