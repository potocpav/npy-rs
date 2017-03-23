
use nom::*;
use std::io::{Result,ErrorKind,Error,Write,BufWriter,Seek,SeekFrom,Cursor};
use std::fs::File;
use std::marker::PhantomData;
use byteorder::{WriteBytesExt, LittleEndian};

use header::{DTypeToValue, Value, DType, parse_header};

/// A result of NPY file deserialization.
///
/// It is an iterator to offer a lazy interface in case the data don't fit into memory.
#[derive(Clone)]
pub struct NpyIterator<'a, T> {
    cursor: Cursor<&'a [u8]>,
    remaining: usize,
    // TODO: Handle correctly. T does not need to be Clone for the iterator to be Clone.
    _t: PhantomData<T>
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
///
/// TODO: Explanation
pub fn from_bytes<'a, T: NpyData>(bytes: &'a [u8]) -> ::std::io::Result<NpyIterator<'a, T>> {
    let (cur, n_rows) = cursor_from_bytes::<T>(bytes)?;
    Ok(NpyIterator::new(cur, n_rows as usize))
}

/// Serialize an iterator over a struct to a NPY file
///
/// TODO: Explanation
pub fn to_file<'a,S,T>(filename: &str, data: T) -> ::std::io::Result<()> where
        S: NpyData + 'a,
        T: IntoIterator<Item=&'a S> {
    let mut fw = BufWriter::new(File::create(filename)?);
    fw.write(&[0x93u8])?;
    fw.write(b"NUMPY")?;
    fw.write(&[0x01u8, 0x00])?;
    let mut header: Vec<u8> = vec![];
    header.extend(&b"{'descr': ["[..]);

    for (id, t) in S::get_dtype() {

        if t.shape.len() == 0 {
            header.extend(format!("('{}', '{}'), ", id, t.ty).as_bytes());
        } else {
            let shape_str = t.shape.into_iter().fold(String::new(), |o,n| o + &format!("{},", n));
            header.extend(format!("('{}', '{}', ({})), ", id, t.ty, shape_str).as_bytes());
        }
    }

    header.extend(&b"], 'fortran_order': False, 'shape': ("[..]);
    let shape_pos = header.len() + 10;
    let filler = &b"abcdefghijklmnopqrs"[..];
    header.extend(filler);
    header.extend(&b",), }"[..]);

    let mut padding: Vec<u8> = vec![];
    padding.extend(&::std::iter::repeat(b' ').take(15 - ((header.len() + 10) % 16)).collect::<Vec<_>>());
    padding.extend(&[b'\n']);

    let len = header.len() + padding.len();
    assert! (len <= ::std::u16::MAX as usize);
    assert!((len + 10) % 16 == 0);

    fw.write_u16::<LittleEndian>(len as u16)?;
    fw.write(&header)?;
    // Padding to 8 bytes
    fw.write(&padding)?;

    // Write data
    let mut num = 0usize;
    for row in data {
        num += 1;
        row.write_row(&mut fw)?;
    }

    // Write the size to the header
    fw.seek(SeekFrom::Start(shape_pos as u64))?;
    let length = format!("{}", num);
    fw.write(length.as_bytes())?;
    fw.write(&b",), }"[..])?;
    fw.write(&::std::iter::repeat(b' ').take(filler.len() - length.len()).collect::<Vec<_>>())?;

    Ok(())
}
