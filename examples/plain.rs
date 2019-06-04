extern crate npy;

use npy::NpyData;
use std::io::Read;

// examples/plain.npy is generated by this Python code:
//
// import numpy as np
// a = np.array([1, 3.5, -6, 2.3])
// np.save('examples/plain.npy', a)

fn main() {
    let mut buf = vec![];
    std::fs::File::open("examples/plain.npy")
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();

    let data: NpyData<f64> = NpyData::from_bytes(&buf).unwrap();
    for arr in data {
        eprintln!("{:?}", arr);
    }
}
