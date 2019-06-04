use nom::*;
use std::io::{Error, ErrorKind, Result};
use std::marker::PhantomData;

use header::{parse_header, DType, Value};
use serializable::Serializable;

/// The data structure representing a deserialized `npy` file.
///
/// The data is internally stored
/// as a byte array, and deserialized only on-demand to minimize unnecessary allocations.
/// The whole contents of the file can be deserialized by the [`to_vec`](#method.to_vec)
/// member function.
pub struct NpyData<'a, T> {
    data: &'a [u8],
    n_records: usize,
    _t: PhantomData<T>,
}

impl<'a, T: Serializable> NpyData<'a, T> {
    /// Deserialize a NPY file represented as bytes
    pub fn from_bytes(bytes: &'a [u8]) -> ::std::io::Result<NpyData<'a, T>> {
        let (data_slice, ns) = Self::get_data_slice(bytes)?;
        Ok(NpyData {
            data: data_slice,
            n_records: ns as usize,
            _t: PhantomData,
        })
    }

    /// Gets a single data-record with the specified index. Returns None, if the index is
    /// out of bounds
    pub fn get(&self, i: usize) -> Option<T> {
        if i < self.n_records {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    /// Returns the total number of records
    pub fn len(&self) -> usize {
        self.n_records
    }

    /// Returns whether there are zero records in this NpyData structure
    pub fn is_empty(&self) -> bool {
        self.n_records == 0
    }

    /// Gets a single data-record wit the specified index. Panics, if the index is out of bounds.
    pub fn get_unchecked(&self, i: usize) -> T {
        T::read(&self.data[i * T::n_bytes()..])
    }

    /// Construct a vector with the deserialized contents of the whole file
    pub fn to_vec(&self) -> Vec<T> {
        let mut v = Vec::with_capacity(self.n_records);
        for i in 0..self.n_records {
            v.push(self.get_unchecked(i));
        }
        v
    }

    fn get_data_slice(bytes: &[u8]) -> Result<(&[u8], i64)> {
        let (data, header) = match parse_header(bytes) {
            IResult::Done(data, header) => Ok((data, header)),
            IResult::Incomplete(needed) => {
                Err(Error::new(ErrorKind::InvalidData, format!("{:?}", needed)))
            }
            IResult::Error(err) => Err(Error::new(ErrorKind::InvalidData, format!("{:?}", err))),
        }?;

        let ns: i64 = if let Value::Map(ref map) = header {
            if let Some(&Value::List(ref l)) = map.get("shape") {
                if l.len() == 1 {
                    if let Some(&Value::Integer(ref n)) = l.get(0) {
                        Some(*n)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                "\'shape\' field is not present or doesn't consist of a tuple of length 1.",
            )
        })?;

        let descr: &Value = if let Value::Map(ref map) = header {
            map.get("descr")
        } else {
            None
        }
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                "\'descr\' field is not present or doesn't contain a list.",
            )
        })?;

        if let Ok(dtype) = DType::from_descr(descr.clone()) {
            let expected_dtype = T::dtype();
            if dtype != expected_dtype {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Types don't match! found: {:?}, expected: {:?}",
                        dtype, expected_dtype
                    ),
                ));
            }
        } else {
            return Err(Error::new(ErrorKind::InvalidData, format!("fail?!?")));
        }

        Ok((data, ns))
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

impl<'a, T: 'a + Serializable> IntoIterator for NpyData<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T> Iterator for IntoIter<'a, T>
where
    T: Serializable,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        self.data.get(self.i - 1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len() - self.i, Some(self.data.len() - self.i))
    }
}

impl<'a, T> ExactSizeIterator for IntoIter<'a, T> where T: Serializable {}
