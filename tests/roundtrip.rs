#[macro_use]
extern crate npy_derive;
extern crate npy;
extern crate byteorder;

use byteorder::ByteOrder;
use std::io::{Read, Write};
use byteorder::{WriteBytesExt, LittleEndian};
use npy::{DType, Serializable};

#[derive(Serializable, Debug, PartialEq, Clone)]
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
    v_arr_u32: [u32;7],
    v_mat_u64: [[u64; 3]; 5],
    vec: Vector5,
}

#[derive(Debug, PartialEq, Clone)]
struct Vector5(Vec<i32>);

impl Serializable for Vector5 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<i4".to_string(), shape: vec![5] }
    }

    #[inline]
    fn n_bytes() -> usize { 5 * 4 }

    #[inline]
    fn read(buf: &[u8]) -> Self {
        let mut ret = Vector5(vec![]);
        let mut off = 0;
        for _ in 0..5 {
            ret.0.push(LittleEndian::read_i32(&buf[off..]));
            off += i32::n_bytes();
        }
        ret
    }

    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for i in 0..5 {
            writer.write_i32::<LittleEndian>(self.0[i])?
        }
        Ok(())
    }
}

#[test]
fn roundtrip() {
    let n = 100i64;

    let mut arrays = vec![];
    for i in 0..n {
        let j = i as u32 * 5 + 2;
        let k = i as u64 * 2 + 5;
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
            v_arr_u32: [j,1+j,2+j,3+j,4+j,5+j,6+j],
            v_mat_u64: [[k,1+k,2+k],[3+k,4+k,5+k],[6+k,7+k,8+k],[9+k,10+k,11+k],[12+k,13+k,14+k]],
            vec: Vector5(vec![1,2,3,4,5]),
        };
        arrays.push(a);
    }

    npy::to_file("tests/roundtrip.npy", arrays.clone()).unwrap();

    let mut buf = vec![];
    std::fs::File::open("tests/roundtrip.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    let arrays2 = npy::NpyData::from_bytes(&buf).unwrap().to_vec();
    assert_eq!(arrays, arrays2);
}

#[test]
fn roundtrip_with_simple_dtype() {
    let array_written = vec![2., 3., 4., 5.];

    npy::to_file("tests/roundtrip_simple.npy", array_written.clone()).unwrap();

    let mut buffer = vec![];
    std::fs::File::open("tests/roundtrip_simple.npy").unwrap()
        .read_to_end(&mut buffer).unwrap();

    let array_read = npy::NpyData::from_bytes(&buffer).unwrap().to_vec();
    assert_eq!(array_written, array_read);
}

#[derive(Serializable, Debug, PartialEq, Clone)]
struct S {
    s: [[[i8; 2]; 3]; 4],
}
