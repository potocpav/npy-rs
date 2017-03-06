
use std::io::{Write,Result};
use byteorder::{WriteBytesExt, LittleEndian};



pub trait Writeable {
    fn write<W: Write>(self, writer: &mut W) -> Result<()>;
}

impl Writeable for i8 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_i8(self)
    }
}
impl Writeable for i16 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_i16::<LittleEndian>(self)
    }
}
impl Writeable for i32 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_i32::<LittleEndian>(self)
    }
}
impl Writeable for i64 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_i64::<LittleEndian>(self)
    }
}
impl Writeable for u8 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_u8(self)
    }
}
impl Writeable for u16 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_u16::<LittleEndian>(self)
    }
}
impl Writeable for u32 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_u32::<LittleEndian>(self)
    }
}
impl Writeable for u64 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_u64::<LittleEndian>(self)
    }
}
impl Writeable for f32 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_f32::<LittleEndian>(self)
    }
}
impl Writeable for f64 {
    fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        writer.write_f64::<LittleEndian>(self)
    }
}
