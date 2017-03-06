
use nom::*;
use std::io::{Result,ErrorKind,Error,Write,BufWriter,Seek,SeekFrom};
use std::fs::File;
use std::marker::PhantomData;
// use std::str::from_utf8;
use byteorder::{WriteBytesExt, LittleEndian};

use super::Cursor;
use header::{Value, parse_header};


pub struct NpyIterator<'a, T> {
    cursor: Cursor<&'a [u8]>,
    remaining: usize,
    _t: PhantomData<T>
}

impl<'a, T> NpyIterator<'a, T> {
    pub fn new(cursor: Cursor<&'a [u8]>, n_rows: usize) -> Self {
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
            if let Some(_) = T::yield_row(&mut self.cursor) {
                panic!("File was longer than the shape implied.");
            }
            None
        } else {
            self.remaining -= 1;
            Some(T::yield_row(&mut self.cursor).expect("File was too short (or the stated shape was too small)."))
        }
    }
}

pub trait NpyData : Sized {

    fn cursor_from_bytes(bytes: &[u8]) -> Result<(Cursor<&[u8]>, i64)> {
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

        let fields = Self::get_fields();

        if fields.len() != descr.len() {
            return Err(Error::new(ErrorKind::InvalidData,
                "The npy header lists a different number of columns than the NpyData trait."));
        }

        // let mut big_endian = Vec::with_capacity(fields.len()); // FIXME: support big endian
        // Get the endianness and check that the field names and types are OK
        for ((name, ty), value) in Self::get_fields().into_iter().zip(descr.into_iter()) {
            if let &Value::List(ref l) = value {
                match &l[..] {
                    &[Value::String(ref n), Value::String(ref t)] => {
                        if n != name {
                            return Err(Error::new(ErrorKind::InvalidData,
                                format!("The descriptor name {:?} doesn't match {:?}.", n, name)))
                        }

                        if Self::to_dtype(ty) != Some(t) {
                            return Err(Error::new(ErrorKind::InvalidData,
                                format!("Type {:?} doesn't match {:?} for descriptor {:?}.", ty, t, n)
                            ));
                        }
                        // if be == true {
                        //     return Err(Error::new(ErrorKind::InvalidData,
                        //         format!("Type {:?} is big endian, which is unsupported at the moment.", ty)
                        //     ));
                        // }
                        // big_endian.push(be);
                    },
                    &[Value::String(ref _n), Value::String(ref _t), Value::List(ref _s)] => {
                        unimplemented!()
                    },
                    _ => return Err(Error::new(ErrorKind::InvalidData,
                        "A type desriptor's type isn't [String, String] nor [String, String, List]."))
                }
            } else {
                return Err(Error::new(ErrorKind::InvalidData,
                    "A type desriptor is not a list."));
            };
        }

        Ok((Cursor::new(data), n_rows))
    }

    // fn check_type(rust_ty: &str, npy_ty: &str) -> Option<bool> {
    //     match (rust_ty, npy_ty) {
    //         ("i8",  ">i1") => Some(true),
    //         ("i8",  "<i1") => Some(false),
    //         ("i16", ">i2") => Some(true),
    //         ("i16", "<i2") => Some(false),
    //         ("i32", ">i4") => Some(true),
    //         ("i32", "<i4") => Some(false),
    //         ("i64", ">i8") => Some(true),
    //         ("i64", "<i8") => Some(false),
    //         ("f32", ">f4") => Some(true),
    //         ("f32", "<f4") => Some(false),
    //         ("f64", ">f8") => Some(true),
    //         ("f64", "<f8") => Some(false),
    //         _ => None
    //     }
    // }

    fn to_dtype<'a>(rust_ty: &str) -> Option<&'a str> {
        match rust_ty {
            "i8" => Some("<i1"),
            "i16" => Some("<i2"),
            "i32" => Some("<i4"),
            "i64" => Some("<i8"),
            "f32" => Some("<f4"),
            "f64" => Some("<f8"),
            _ => None
        }
    }

    fn from_bytes<'a>(bytes: &'a [u8]) -> ::std::io::Result<NpyIterator<'a, Self>> {
        let (cur, n_rows) = Self::cursor_from_bytes(bytes)?;
        Ok(NpyIterator::new(cur, n_rows as usize))
    }

    fn to_file<T>(filename: &str, data: T) -> ::std::io::Result<()> where
            T: Iterator<Item=Self> {
        let mut fw = BufWriter::new(File::create(filename)?);
        fw.write(&[0x93u8])?;
        fw.write(b"NUMPY")?;
        fw.write(&[0x01u8, 0x00])?;
        let mut header: Vec<u8> = vec![];
        header.extend(&b"{'descr': ["[..]);
        for (id, ty) in Self::get_fields() {
            if let Some(t) = Self::to_dtype(ty) {
                header.extend(format!("('{}', '{}'), ", id, t).as_bytes());
            } else {
                return Err(Error::new(ErrorKind::InvalidData,
                    format!("Serialization of type {:?} not implemented.", ty)
                ));
            }
        }
        header.extend(&b"], 'fortran_order': False, 'shape': ("[..]);
        let shape_pos = header.len() + 10;
        let filler = &b"abcdefghijklmnopqrs"[..];
        header.extend(filler);
        header.extend(&b",), }"[..]);
        assert!(header.len() <= ::std::i16::MAX as usize);

        fw.write_i16::<LittleEndian>(header.len() as i16)?;
        fw.write(&header)?;
        // Padding to 8 bytes
        fw.write(&::std::iter::repeat(b' ').take(15 - ((header.len() + 10) % 16)).collect::<Vec<_>>())?;
        fw.write(&[b'\n'])?;

        // Write data
        let mut num = 0usize;
        for row in data {
            num += 1;
            row.write_row(&mut fw)?;
        }

        // Write the size to the header
        fw.seek(SeekFrom::Start(shape_pos as u64))?;
        let length = format!("{}", num);
        fw.write(length.as_bytes());
        fw.write(&b",), }"[..]);
        fw.write(&::std::iter::repeat(b' ').take(filler.len() - length.len()).collect::<Vec<_>>())?;

        Ok(())
    }

    // Provided by the macro npy_data
    fn get_fields() -> Vec<(&'static str, &'static str)>;

    // Provided by the macro npy_data
    fn yield_row(c: &mut Cursor<&[u8]>) -> Option<Self>;

    // Provided by the macro npy_data
    fn write_row<W: Write>(&self, writer: &mut W) -> ::std::io::Result<()>;
}

#[macro_export]
macro_rules! npy_data {
    (pub struct $name:ident {
        $($field_name:ident: $field_type:ty,)*
    }) => {
        #[allow(non_snake_case)]
        #[derive(Debug, Default)]
        pub struct $name {
            $($field_name: $field_type,)*
        }

        impl NpyData for $name {
            fn get_fields() -> Vec<(&'static str, &'static str)> {
                vec![$((stringify!($field_name), stringify!($field_type))),*]
            }

            fn yield_row(c: &mut npy::Cursor<&[u8]>) -> Option<Self> {
                Some(Self { $(
                    $field_name: {
                        if let Ok(v) = npy::Readable::read(c) {
                            v
                        } else {
                            return None;
                        }
                    }
                ),* })
            }

            fn write_row<W: ::std::io::Write>(&self, writer: &mut W) -> ::std::io::Result<()> {
                $( npy::Writeable::write(self.$field_name, writer)?; )*
                Ok(())
            }

        }
    }
}
