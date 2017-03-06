
use std::io::{Cursor,Result};
use byteorder::{ReadBytesExt, LittleEndian};

/// A type that can be deserialized as a part of a struct implementing NpyData.
pub trait Readable: Sized {
    /// Deserialize a single data field
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self>;
}

impl Readable for i8 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i8()
    }
}
impl Readable for i16 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i16::<LittleEndian>()
    }
}
impl Readable for i32 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i32::<LittleEndian>()
    }
}
impl Readable for i64 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i64::<LittleEndian>()
    }
}
impl Readable for u8 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u8()
    }
}
impl Readable for u16 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u16::<LittleEndian>()
    }
}
impl Readable for u32 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u32::<LittleEndian>()
    }
}
impl Readable for u64 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u64::<LittleEndian>()
    }
}
impl Readable for f32 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_f32::<LittleEndian>()
    }
}
impl Readable for f64 {
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_f64::<LittleEndian>()
    }
}
