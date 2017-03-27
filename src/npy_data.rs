
use nom::*;
use std::io::{Result,ErrorKind,Error,Write,Cursor};
use std::marker::PhantomData;

use header::{DTypeToValue, Value, DType, parse_header};

/// A result of NPY file deserialization.
///
/// It is an iterator to offer a lazy interface in case the data don't fit into memory.
pub struct NpyIterator<'a, T> {
    cursor: Cursor<&'a [u8]>,
    remaining: usize,
    _t: PhantomData<T>
}

impl<'a, T> Clone for NpyIterator<'a, T> {
    fn clone(&self) -> Self {
        NpyIterator {
            cursor: self.cursor.clone(),
            remaining: self.remaining,
            _t: PhantomData,
        }
    }
}

impl<'a, T> NpyIterator<'a, T> {
    fn new(cursor: Cursor<&'a [u8]>, n_rows: usize) -> Self {
        NpyIterator {
            cursor: cursor,
            remaining: n_rows,
            _t: PhantomData
        }
    }
}

impl<'a, T> Iterator for NpyIterator<'a, T> where T: NpyData {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            if let Ok(_) = T::read_row(&mut self.cursor) {
                panic!("File was longer than the shape implied.");
            }
            None
        } else {
            self.remaining -= 1;
            Some(T::read_row(&mut self.cursor).expect("File was too short (or the stated shape was too small)."))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for NpyIterator<'a, T> where T: NpyData {}

/// This trait is often automatically implemented by a `#[derive(NpyData)]`
pub trait NpyData : Sized {
    /// Get a vector of pairs (field_name, DType) representing the struct type.
    fn get_dtype() -> Vec<(&'static str, DType)>;

    /// Deserialize binary data to a single instance of Self
    fn read_row(c: &mut Cursor<&[u8]>) -> ::std::io::Result<Self>;

    /// Write Self in a binary form to a writer.
    fn write_row<W: Write>(&self, writer: &mut W) -> ::std::io::Result<()>;
}

fn cursor_from_bytes<T: NpyData>(bytes: &[u8]) -> Result<(Cursor<&[u8]>, i64)> {
    let (data, header) = match parse_header(bytes) {
        IResult::Done(data, header) => {
            Ok((data, header))
        },
        IResult::Incomplete(needed) => {
            Err(Error::new(ErrorKind::InvalidData, format!("{:?}", needed)))
        },
        IResult::Error(err) => {
            Err(Error::new(ErrorKind::InvalidData, format!("{:?}", err)))
        }
    }?;


    let n_rows: i64 =
        if let Value::Map(ref map) = header {
            if let Some(&Value::List(ref l)) = map.get("shape") {
                if l.len() == 1 {
                    if let Some(&Value::Integer(ref n)) = l.get(0) {
                        Some(*n)
                    } else { None }
                } else { None }
            } else { None }
        } else { None }
        .ok_or(Error::new(ErrorKind::InvalidData,
                "\'shape\' field is not present or doesn't consist of a tuple of length 1."))?;

    let descr: &[Value] =
        if let Value::Map(ref map) = header {
            if let Some(&Value::List(ref l)) = map.get("descr") {
                Some(l)
            } else { None }
        } else { None }
        .ok_or(Error::new(ErrorKind::InvalidData,
                "\'descr\' field is not present or doesn't contain a list."))?;

    let expected_type_ast = T::get_dtype().into_iter().map(|(s,dt)| dt.to_value(&s)).collect::<Vec<_>>();
    // TODO: It would be better to compare DType, not Value AST.
    if expected_type_ast != descr {
        return Err(Error::new(ErrorKind::InvalidData,
            format!("Types don't match! type1: {:?}, type2: {:?}", expected_type_ast, descr)
        ));
    }

    Ok((Cursor::new(data), n_rows))
}


/// Deserialize a NPY file represented as bytes
pub fn from_bytes<'a, T: NpyData>(bytes: &'a [u8]) -> ::std::io::Result<NpyIterator<'a, T>> {
    let (cur, n_rows) = cursor_from_bytes::<T>(bytes)?;
    Ok(NpyIterator::new(cur, n_rows as usize))
}
