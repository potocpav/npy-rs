use header::DType;
use type_str::{TypeStr, Endianness, TypeKind};
use byteorder::{ByteOrder, NativeEndian, WriteBytesExt};
use self::{TypeKind::*};
use std::io;
use std::fmt;

/// Trait that permits reading a type from an `.npy` file.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait Deserialize: Sized {
    /// Think of this as like a `Fn(&[u8]) -> (Self, &[u8])`.
    ///
    /// There is no closure-like sugar for these; you must manually define a type that
    /// implements [`TypeRead`].
    type Reader: TypeRead<Value=Self>;

    /// Get a function that deserializes a single data field at a time
    ///
    /// The function receives a byte buffer containing at least
    /// `dtype.num_bytes()` bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the `DType` is not compatible with `Self`.
    fn reader(dtype: &DType) -> Result<Self::Reader, DTypeError>;
}

/// Trait that permits writing a type to an `.npy` file.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait Serialize {
    /// Think of this as some sort of `for<W: io::Write> Fn(W, &Self) -> io::Result<()>`.
    ///
    /// There is no closure-like sugar for these; you must manually define a type that
    /// implements [`TypeWrite`].
    type Writer: TypeWrite<Value=Self>;

    /// Get a function that serializes a single data field at a time.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the `DType` is not compatible with `Self`.
    fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError>;
}

/// Subtrait of [`Serialize`] for types which have a reasonable default [`DType`].
///
/// This opens up some simpler APIs for serialization. (e.g. [`::to_file`])
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait AutoSerialize: Serialize {
    /// A suggested format for serialization.
    ///
    /// The builtin implementations for primitive types generally prefer `|` endianness if possible,
    /// else the machine endian format.
    fn default_dtype() -> DType;
}

/// Like a `Fn(&[u8]) -> (T, &[u8])`.
///
/// It is a separate trait from `Fn` for consistency with [`TypeWrite`], and so that
/// default methods can potentially be added in the future that may be overriden
/// for efficiency.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait TypeRead {
    /// Type returned by the function.
    type Value;

    /// The function.
    ///
    /// Receives *at least* enough bytes to read `Self::Value`, and returns the remainder.
    fn read_one<'a>(&self, bytes: &'a [u8]) -> (Self::Value, &'a [u8]);
}

/// Like some sort of `for<W: io::Write> Fn(W, &T) -> io::Result<()>`.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait TypeWrite {
    /// Type accepted by the function.
    type Value: ?Sized;

    /// The function.
    fn write_one<W: io::Write>(&self, writer: W, value: &Self::Value) -> io::Result<()>
    where Self: Sized;
}

/// Indicates that a particular rust type does not support serialization or deserialization
/// as a given [`DType`].
#[derive(Debug, Clone)]
pub struct DTypeError(ErrorKind);

#[derive(Debug, Clone)]
enum ErrorKind {
    Custom(String),
    ExpectedScalar {
        dtype: String,
        rust_type: &'static str,
    },
    BadScalar {
        type_str: TypeStr,
        rust_type: &'static str,
        verb: &'static str,
    },
}

impl std::error::Error for DTypeError {}

impl DTypeError {
    /// Construct with a custom error message.
    pub fn custom<S: AsRef<str>>(msg: S) -> Self {
        DTypeError(ErrorKind::Custom(msg.as_ref().to_string()))
    }

    // verb should be "read" or "write"
    fn bad_scalar(verb: &'static str, type_str: &TypeStr, rust_type: &'static str) -> Self {
        let type_str = type_str.clone();
        DTypeError(ErrorKind::BadScalar { type_str, rust_type, verb })
    }

    fn expected_scalar(dtype: &DType, rust_type: &'static str) -> Self {
        let dtype = dtype.descr();
        DTypeError(ErrorKind::ExpectedScalar { dtype, rust_type })
    }
}

impl fmt::Display for DTypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            ErrorKind::Custom(msg) => {
                write!(f, "{}", msg)
            },
            ErrorKind::ExpectedScalar { dtype, rust_type } => {
                write!(f, "type {} requires a scalar (string) dtype, not {}", rust_type, dtype)
            },
            ErrorKind::BadScalar { type_str, rust_type, verb } => {
                write!(f, "cannot {} type {} with type-string '{}'", verb, rust_type, type_str)
            },
        }
    }
}

// Takes info about each data size, from largest to smallest.
macro_rules! impl_integer_serializable {
    ( @iterate
        meta: $meta:tt
        remaining: []
    ) => {};

    ( @iterate
        meta: $meta:tt
        remaining: [$first:tt $($smaller:tt)*]
    ) => {
        impl_integer_serializable! {
          @generate
            meta: $meta
            current: $first
        }

        impl_integer_serializable! {
          @iterate
            meta: $meta
            remaining: [ $($smaller)* ]
        }
    };

    (
        @generate
        meta: [ (main_ty: $Int:ident) (date_ty: $DateTime:ident) ]
        current: [ $size:literal $int:ident
                   (size1: $size1_cfg:meta) $read_int:ident $write_int:ident
                 ]
    ) => {
        mod $int {
            use super::*;

            pub struct AnyEndianReader { pub(super) swap_byteorder: bool }
            pub struct AnyEndianWriter { pub(super) swap_byteorder: bool }

            pub(super) fn expect_scalar_dtype(dtype: &DType) -> Result<&TypeStr, DTypeError> {
                dtype.as_scalar().ok_or_else(|| {
                    DTypeError::expected_scalar(dtype, stringify!($int))
                })
            }

            #[inline]
            fn maybe_swap(swap: bool, x: $int) -> $int {
                match swap {
                    true => x.to_be().to_le(),
                    false => x,
                }
            }

            impl TypeRead for AnyEndianReader {
                type Value = $int;

                #[inline(always)]
                fn read_one<'a>(&self, bytes: &'a [u8]) -> (Self::Value, &'a [u8]) {
                    let value = maybe_swap(self.swap_byteorder, NativeEndian::$read_int(bytes));
                    (value, &bytes[$size..])
                }
            }

            impl TypeWrite for AnyEndianWriter {
                type Value = $int;

                #[inline(always)]
                fn write_one<W: io::Write>(&self, mut writer: W, &value: &Self::Value) -> io::Result<()> {
                    writer.$write_int::<NativeEndian>(maybe_swap(self.swap_byteorder, value))
                }
            }
        }

        impl Deserialize for $int {
            type Reader = $int::AnyEndianReader;

            fn reader(dtype: &DType) -> Result<Self::Reader, DTypeError> {
                match $int::expect_scalar_dtype(dtype)? {
                    // Read an integer of the same size and signedness.
                    //
                    // DateTime is an unsigned integer and TimeDelta is a signed integer,
                    // so we support those too.
                    TypeStr { size: $size, endianness, type_kind: $Int, .. } |
                    TypeStr { size: $size, endianness, type_kind: $DateTime, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok($int::AnyEndianReader { swap_byteorder })
                    },
                    type_str => Err(DTypeError::bad_scalar("read", type_str, stringify!($int))),
                }
            }
        }

        impl Serialize for $int {
            type Writer = $int::AnyEndianWriter;

            fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError> {
                match $int::expect_scalar_dtype(dtype)? {
                    // Write a signed integer of the correct size
                    TypeStr { size: $size, endianness, type_kind: $Int, .. } |
                    TypeStr { size: $size, endianness, type_kind: $DateTime, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok($int::AnyEndianWriter { swap_byteorder })
                    },
                    type_str => Err(DTypeError::bad_scalar("write", type_str, stringify!($int))),
                }
            }
        }

        impl AutoSerialize for $int {
            fn default_dtype() -> DType {
                DType::new_scalar(TypeStr::with_auto_endianness($Int, $size, None))
            }
        }
    };
}

// Needed by the macro: Methods missing from byteorder
trait ReadSingleByteExt {
    #[inline(always)] fn read_u8_(bytes: &[u8]) -> u8 { bytes[0] }
    #[inline(always)] fn read_i8_(bytes: &[u8]) -> i8 { i8::from_ne_bytes([bytes[0]]) }
}

impl<E: ByteOrder> ReadSingleByteExt for E {}

/// Needed by the macro: Methods modified to take a generic type param
trait WriteSingleByteExt: WriteBytesExt {
    #[inline(always)] fn write_u8_<B>(&mut self, value: u8) -> io::Result<()> { self.write_u8(value) }
    #[inline(always)] fn write_i8_<B>(&mut self, value: i8) -> io::Result<()> { self.write_i8(value) }
}

impl<W: WriteBytesExt + ?Sized> WriteSingleByteExt for W {}

// `all()` means "true", `any()` means "false". (these get put inside `cfg`)
impl_integer_serializable! {
    @iterate
    meta: [ (main_ty: Int) (date_ty: TimeDelta) ]
    remaining: [
        // numpy doesn't support i128
        [ 8  i64 (size1: any()) read_i64  write_i64 ]
        [ 4  i32 (size1: any()) read_i32  write_i32 ]
        [ 2  i16 (size1: any()) read_i16  write_i16 ]
        [ 1   i8 (size1: all()) read_i8_  write_i8_ ]
    ]
}

impl_integer_serializable! {
    @iterate
    meta: [ (main_ty: Uint) (date_ty: DateTime) ]
    remaining: [
        // numpy doesn't support i128
        [ 8  u64 (size1: any()) read_u64  write_u64 ]
        [ 4  u32 (size1: any()) read_u32  write_u32 ]
        [ 2  u16 (size1: any()) read_u16  write_u16 ]
        [ 1   u8 (size1: all()) read_u8_  write_u8_ ]
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader_output<T: Deserialize>(dtype: &DType, bytes: &[u8]) -> T {
        T::reader(dtype).unwrap_or_else(|e| panic!("{}", e)).read_one(bytes).0
    }

    fn reader_expect_err<T: Deserialize>(dtype: &DType) {
        T::reader(dtype).err().expect("reader_expect_err failed!");
    }

    fn writer_output<T: Serialize + ?Sized>(dtype: &DType, value: &T) -> Vec<u8> {
        let mut vec = vec![];
        T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
            .write_one(&mut vec, value).unwrap();
        vec
    }

    fn writer_expect_err<T: Serialize + ?Sized>(dtype: &DType) {
        T::writer(dtype).err().expect("writer_expect_err failed!");
    }

    fn writer_expect_write_err<T: Serialize + ?Sized>(dtype: &DType, value: &T) {
        let mut vec = vec![];
        T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
            .write_one(&mut vec, value)
            .err().expect("writer_expect_write_err failed!");
    }

    const BE_ONE_64: &[u8] = &[0, 0, 0, 0, 0, 0, 0, 1];
    const LE_ONE_64: &[u8] = &[1, 0, 0, 0, 0, 0, 0, 0];
    const BE_ONE_32: &[u8] = &[0, 0, 0, 1];
    const LE_ONE_32: &[u8] = &[1, 0, 0, 0];

    #[test]
    fn identity() {
        let be = DType::parse("'>i4'").unwrap();
        let le = DType::parse("'<i4'").unwrap();

        assert_eq!(reader_output::<i32>(&be, BE_ONE_32), 1);
        assert_eq!(reader_output::<i32>(&le, LE_ONE_32), 1);
        assert_eq!(writer_output::<i32>(&be, &1), BE_ONE_32);
        assert_eq!(writer_output::<i32>(&le, &1), LE_ONE_32);

        let be = DType::parse("'>u4'").unwrap();
        let le = DType::parse("'<u4'").unwrap();

        assert_eq!(reader_output::<u32>(&be, BE_ONE_32), 1);
        assert_eq!(reader_output::<u32>(&le, LE_ONE_32), 1);
        assert_eq!(writer_output::<u32>(&be, &1), BE_ONE_32);
        assert_eq!(writer_output::<u32>(&le, &1), LE_ONE_32);

        for &dtype in &["'>i1'", "'<i1'", "'|i1'"] {
            let dtype = DType::parse(dtype).unwrap();
            assert_eq!(reader_output::<i8>(&dtype, &[1]), 1);
            assert_eq!(writer_output::<i8>(&dtype, &1), &[1][..]);
        }

        for &dtype in &["'>u1'", "'<u1'", "'|u1'"] {
            let dtype = DType::parse(dtype).unwrap();
            assert_eq!(reader_output::<u8>(&dtype, &[1]), 1);
            assert_eq!(writer_output::<u8>(&dtype, &1), &[1][..]);
        }
    }

    #[test]
    fn datetime_as_int() {
        let be = DType::parse("'>m8[ns]'").unwrap();
        let le = DType::parse("'<m8[ns]'").unwrap();

        assert_eq!(reader_output::<i64>(&be, BE_ONE_64), 1);
        assert_eq!(reader_output::<i64>(&le, LE_ONE_64), 1);
        assert_eq!(writer_output::<i64>(&be, &1), BE_ONE_64);
        assert_eq!(writer_output::<i64>(&le, &1), LE_ONE_64);

        let be = DType::parse("'>M8[ns]'").unwrap();
        let le = DType::parse("'<M8[ns]'").unwrap();

        assert_eq!(reader_output::<u64>(&be, BE_ONE_64), 1);
        assert_eq!(reader_output::<u64>(&le, LE_ONE_64), 1);
        assert_eq!(writer_output::<u64>(&be, &1), BE_ONE_64);
        assert_eq!(writer_output::<u64>(&le, &1), LE_ONE_64);
    }

    #[test]
    fn wrong_size_int() {
        let t_i32 = DType::parse("'<i4'").unwrap();
        let t_u32 = DType::parse("'<u4'").unwrap();

        reader_expect_err::<i64>(&t_i32);
        reader_expect_err::<i16>(&t_i32);
        reader_expect_err::<u64>(&t_u32);
        reader_expect_err::<u16>(&t_u32);
        writer_expect_err::<i64>(&t_i32);
        writer_expect_err::<i16>(&t_i32);
        writer_expect_err::<u64>(&t_u32);
        writer_expect_err::<u16>(&t_u32);
    }

    #[test]
    fn default_simple_type_strs() {
        assert_eq!(i8::default_dtype().descr(), "'|i1'");
        assert_eq!(u8::default_dtype().descr(), "'|u1'");

        if 1 == i32::from_be(1) {
            assert_eq!(i16::default_dtype().descr(), "'>i2'");
            assert_eq!(i32::default_dtype().descr(), "'>i4'");
            assert_eq!(i64::default_dtype().descr(), "'>i8'");
            assert_eq!(u32::default_dtype().descr(), "'>u4'");
        } else {
            assert_eq!(i16::default_dtype().descr(), "'<i2'");
            assert_eq!(i32::default_dtype().descr(), "'<i4'");
            assert_eq!(i64::default_dtype().descr(), "'<i8'");
            assert_eq!(u32::default_dtype().descr(), "'<u4'");
        }
    }
}
