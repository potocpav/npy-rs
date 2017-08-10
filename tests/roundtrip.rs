#[macro_use]
extern crate npy_derive;
extern crate npy;
extern crate byteorder;

use std::io::{Cursor,Read,Write};
use byteorder::{WriteBytesExt, ReadBytesExt, LittleEndian};
use npy::{DType,Serializable};

#[derive(NpyRecord, Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
struct Vector5(Vec<i32>);

impl Serializable for Vector5 {
    fn dtype() -> DType {
        DType { ty: "<i4", shape: vec![5] }
    }

    fn read(c: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        let mut ret = Vector5(vec![]);
        for _ in 0..5 {
            ret.0.push(c.read_i32::<LittleEndian>()?);
        }
        Ok(ret)
    }

    fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for i in 0..5 {
            writer.write_i32::<LittleEndian>(self.0[i])?
        }
        Ok(())
    }
}

#[test]
fn roundtrip() {
    let n = 100;

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
            v_arr_u32: [0+j,1+j,2+j,3+j,4+j,5+j,6+j],
            v_mat_u64: [[0+k,1+k,2+k],[3+k,4+k,5+k],[6+k,7+k,8+k],[9+k,10+k,11+k],[12+k,13+k,14+k]],
            vec: Vector5(vec![1,2,3,4,5]),
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
