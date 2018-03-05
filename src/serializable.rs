
use std::io::{Write,Result};
use byteorder::{WriteBytesExt, LittleEndian};
use header::DType;
use byteorder::ByteOrder;

/// This trait contains information on how to serialize and deserialize a type.
///
/// An example illustrating a `Serializable` implementation for a fixed-size vector is in
/// [the roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
/// It is strongly advised to annotate the `Serializable` functions as `#[inline]` for good
/// performance.
pub trait Serializable : Sized {
    /// Convert a type to a structure representing a Numpy type
    fn dtype() -> DType;

    /// Get the number of bytes of the binary repr
    fn n_bytes() -> usize;

    /// Deserialize a single data field, advancing the cursor in the process.
    fn read(c: &[u8]) -> Self;

    /// Serialize a single data field into a writer.
    fn write<W: Write>(&self, writer: &mut W) -> Result<()>;
}

impl Serializable for i8 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<i1".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 1 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        unsafe { ::std::mem::transmute(buf[0]) } // TODO: a better way
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i8(*self)
    }
}

impl Serializable for i16 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<i2".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 2 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_i16(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i16::<LittleEndian>(*self)
    }
}

impl Serializable for i32 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<i4".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 4 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_i32(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i32::<LittleEndian>(*self)
    }
}

impl Serializable for i64 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<i8".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 8 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_i64(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_i64::<LittleEndian>(*self)
    }
}

impl Serializable for u8 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<u1".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 1 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        buf[0]
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u8(*self)
    }
}

impl Serializable for u16 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<u2".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 2 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_u16(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u16::<LittleEndian>(*self)
    }
}

impl Serializable for u32 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<u4".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 4 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_u32(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u32::<LittleEndian>(*self)
    }
}

impl Serializable for u64 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<u8".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 8 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_u64(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_u64::<LittleEndian>(*self)
    }
}

impl Serializable for f32 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<f4".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 4 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_f32(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_f32::<LittleEndian>(*self)
    }
}

impl Serializable for f64 {
    #[inline]
    fn dtype() -> DType {
        DType::Plain { ty: "<f8".to_string(), shape: vec![] }
    }
    #[inline]
    fn n_bytes() -> usize { 8 }
    #[inline]
    fn read(buf: &[u8]) -> Self {
        LittleEndian::read_f64(buf)
    }
    #[inline]
    fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_f64::<LittleEndian>(*self)
    }
}

macro_rules! gen_array_serializable {
    ($($n:tt),+) => { $(
        impl<T: Serializable + Default + Copy> Serializable for [T; $n] {
            #[inline]
            fn dtype() -> DType {
                use DType::*;
                match T::dtype() {
                    Plain { ref ty, ref shape } => DType::Plain {
                        ty: ty.clone(),
                        shape: shape.clone().into_iter().chain(Some($n)).collect()
                    },
                    Record(_) => unimplemented!("arrays of nested records")
                }
            }
            #[inline]
            fn n_bytes() -> usize { T::n_bytes() * $n }
            #[inline]
            fn read(buf: &[u8]) -> Self {
                let mut a = [T::default(); $n];
                let mut off = 0;
                for x in &mut a {
                    *x = T::read(&buf[off..]);
                    off += T::n_bytes();
                }
                a
            }
            #[inline]
            fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
                for item in self {
                    item.write(writer)?;
                }
                Ok(())
            }
        }
    )+ }
}

gen_array_serializable!(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
