
use std::io::{Cursor,Write,Result};
use byteorder::{WriteBytesExt, ReadBytesExt, LittleEndian};
use header::DType;

/// This trait contains information on how to serialize and deserialize a type.
///
/// It must be implemented for every member of a struct that we use as a serialization target,
/// typically by using `#[derive(NpyData)]`.
pub trait Serializable : Sized {
    /// Convert a type to a structure representing a Numpy type
    fn dtype() -> DType;

    /// Deserialize a single data field, advancing the cursor in the process.
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self>;

    /// Serialize a single data field into a writer.
    fn write<W: Write>(&self, writer: &mut W) -> Result<()>;
}

// impl<'a, T: Serializable + Copy + 'a> Serializable for &'a T {
//     fn dtype() -> DType {
//         Self::dtype()
//     }
//     fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
//         T::read(c)
//     }
//     fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
//         (*self).write(writer)
//     }
// }

impl Serializable for i8 {
    fn dtype() -> DType {
        DType { ty: "<i1", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i8()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i8(*self)
    }
}

impl Serializable for i16 {
    fn dtype() -> DType {
        DType { ty: "<i2", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i16::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i16::<LittleEndian>(*self)
    }
}

impl Serializable for i32 {
    fn dtype() -> DType {
        DType { ty: "<i4", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i32::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i32::<LittleEndian>(*self)
    }
}

impl Serializable for i64 {
    fn dtype() -> DType {
        DType { ty: "<i8", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_i64::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i64::<LittleEndian>(*self)
    }
}

impl Serializable for u8 {
    fn dtype() -> DType {
        DType { ty: "<u1", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u8()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u8(*self)
    }
}

impl Serializable for u16 {
    fn dtype() -> DType {
        DType { ty: "<u2", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u16::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u16::<LittleEndian>(*self)
    }
}

impl Serializable for u32 {
    fn dtype() -> DType {
        DType { ty: "<u4", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u32::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u32::<LittleEndian>(*self)
    }
}

impl Serializable for u64 {
    fn dtype() -> DType {
        DType { ty: "<u8", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_u64::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u64::<LittleEndian>(*self)
    }
}

impl Serializable for f32 {
    fn dtype() -> DType {
        DType { ty: "<f4", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_f32::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_f32::<LittleEndian>(*self)
    }
}

impl Serializable for f64 {
    fn dtype() -> DType {
        DType { ty: "<f8", shape: vec![] }
    }
    fn read(c: &mut Cursor<&[u8]>) -> Result<Self> {
        c.read_f64::<LittleEndian>()
    }
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_f64::<LittleEndian>(*self)
    }
}
