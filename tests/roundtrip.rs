#[macro_use]
extern crate npy_derive;
extern crate npy;

use std::io::Read;

#[derive(NpyData, Debug, PartialEq)]
struct Array {
    v_i8: i8,
    v_i16: i16,
    v_i32: i32,
    v_i64: i64,
    v_u8: u8,
    v_u16: u16,
    v_u32: u32,
    v_u64: u64,
    v_f32: f32,
    v_f64: f64,
}

#[test]
fn roundtrip() {
    let n = 100;

    let mut arrays = vec![];
    for i in 0..n {
        let a = Array {
            v_i8: i as i8,
            v_i16: i as i16,
            v_i32: i as i32,
            v_i64: i as i64,
            v_u8: i as u8,
            v_u16: i as u16,
            v_u32: i as u32,
            v_u64: i as u64,
            v_f32: i as f32,
            v_f64: i as f64,
        };
        arrays.push(a);
    }

    npy::to_file("tests/roundtrip.npy", arrays.iter()).unwrap();

    let mut buf = vec![];
    std::fs::File::open("tests/roundtrip.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    let arrays2 = npy::from_bytes::<Array>(&buf).unwrap().collect::<Vec<_>>();
    assert_eq!(arrays, arrays2);
}
