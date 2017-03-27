
use std::io::{self,Write,BufWriter,Seek,SeekFrom};
use std::fs::File;
use std::path::Path;
use std::marker::PhantomData;

use byteorder::{WriteBytesExt, LittleEndian};

use npy_data::NpyData;

const FILLER: &'static [u8] = &[42; 19];

/// Serialize into a file one row at a time. To serialize an iterator, use the
/// [`to_file`](fn.to_file.html) function.
pub struct OutFile<Row: NpyData> {
    shape_pos: usize,
    len: usize,
    fw: BufWriter<File>,
    _t: PhantomData<Row>
}


impl<Row: NpyData> OutFile<Row> {
    /// Open a file
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut fw = BufWriter::new(File::create(path)?);
        fw.write(&[0x93u8])?;
        fw.write(b"NUMPY")?;
        fw.write(&[0x01u8, 0x00])?;
        let mut header: Vec<u8> = vec![];
        header.extend(&b"{'descr': ["[..]);

        for (id, t) in Row::get_dtype() {

            if t.shape.len() == 0 {
                header.extend(format!("('{}', '{}'), ", id, t.ty).as_bytes());
            } else {
                let shape_str = t.shape.into_iter().fold(String::new(), |o,n| o + &format!("{},", n));
                header.extend(format!("('{}', '{}', ({})), ", id, t.ty, shape_str).as_bytes());
            }
        }

        header.extend(&b"], 'fortran_order': False, 'shape': ("[..]);
        let shape_pos = header.len() + 10;
        header.extend(FILLER);
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

        Ok(OutFile {
            shape_pos: shape_pos,
            len: 0,
            fw: fw,
            _t: PhantomData,
        })
    }

    /// Append a single `NpyData` instance to the file
    pub fn push(&mut self, row: &Row) -> io::Result<()> {
        self.len += 1;
        row.write_row(&mut self.fw)
    }

    fn close_(&mut self) -> io::Result<()> {
        // Write the size to the header
        self.fw.seek(SeekFrom::Start(self.shape_pos as u64))?;
        let length = format!("{}", self.len);
        self.fw.write(length.as_bytes())?;
        self.fw.write(&b",), }"[..])?;
        self.fw.write(&::std::iter::repeat(b' ').take(FILLER.len() - length.len()).collect::<Vec<_>>())?;
        Ok(())
    }

    /// Finish writing the file by finalizing the header and closing the file.
    ///
    /// If omitted, the file will be closed on drop automatically, but it will panic on error.
    pub fn close(mut self) -> io::Result<()> {
        self.close_()
    }
}

impl<Row: NpyData> Drop for OutFile<Row> {
    fn drop(&mut self) {
        self.close_().unwrap()
    }
}


/// Serialize an iterator over a struct to a NPY file
///
/// A single-statement alternative to saving row by row using the [`OutFile`](struct.OutFile.html).
pub fn to_file<'a, S, T, P>(filename: P, data: T) -> ::std::io::Result<()> where
        P: AsRef<Path>,
        S: NpyData + 'a,
        T: IntoIterator<Item=&'a S> {

    let mut of = OutFile::open(filename)?;
    for row in data {
        of.push(row)?;
    }
    of.close()
}
