
use nom::*;
use std::io::{Result,ErrorKind,Error,Write};
use std::marker::PhantomData;

use header::{DTypeToValue, Value, DType, parse_header};


/// This trait is often automatically implemented by a `#[derive(NpyRecord)]`
pub trait NpyRecord : Sized {
    /// Get a vector of pairs (field_name, DType) representing the struct type.
    fn get_dtype() -> Vec<(&'static str, DType)>;

    /// Get the number of bytes of this record in the serialized representation
    fn n_bytes() -> usize;

    /// Deserialize binary data to a single instance of Self
    fn read_row(buf: &[u8]) -> Self;

    /// Write Self in a binary form to a writer.
    fn write_row<W: Write>(&self, writer: &mut W) -> ::std::io::Result<()>;
}

pub struct NpyData<'a, T> {
    data: &'a [u8],
    n_records: usize,
    _t: PhantomData<T>,
}

impl<'a, T: NpyRecord> NpyData<'a, T> {
    /// Deserialize a NPY file represented as bytes
    pub fn from_bytes(bytes: &'a [u8]) -> ::std::io::Result<NpyData<'a, T>> {
        let (data_slice, n_rows) = Self::get_data_slice(bytes)?;
        Ok(NpyData { data: data_slice, n_records: n_rows as usize, _t: PhantomData })
    }

    pub fn get(&self, i: usize) -> Option<T> {
        if i < self.n_records {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.n_records
    }

    pub fn get_unchecked(&self, i: usize) -> T {
        T::read_row(&self.data[i * T::n_bytes()..])
    }

    pub fn to_vec(&self) -> Vec<T> {
        let mut v = Vec::with_capacity(self.n_records);
        for i in 0..self.n_records {
            v.push(self.get_unchecked(i));
        }
        v
    }
    //
    // pub fn iter<'b>(&'b self) -> IntoIter<'b, T> {
    //     IntoIter::new(self)
    // }

    fn get_data_slice(bytes: &[u8]) -> Result<(&[u8], i64)> {
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

        Ok((data, n_rows))
    }
}

/// A result of NPY file deserialization.
///
/// It is an iterator to offer a lazy interface in case the data don't fit into memory.
pub struct IntoIter<'a, T: 'a> {
    data: NpyData<'a, T>,
    i: usize,
}

impl<'a, T> IntoIter<'a, T> {
    fn new(data: NpyData<'a, T>) -> Self {
        IntoIter { data, i: 0 }
    }
}

impl<'a, T: 'a + NpyRecord> IntoIterator for NpyData<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T> Iterator for IntoIter<'a, T> where T: NpyRecord {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        self.data.get(self.i - 1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len() - self.i, Some(self.data.len() - self.i))
    }
}

impl<'a, T> ExactSizeIterator for IntoIter<'a, T> where T: NpyRecord {}
