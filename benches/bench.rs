#![feature(test)]

#[macro_use]
extern crate npy_derive;
extern crate npy;
extern crate test;

use npy::Serializable;
use test::Bencher;
use test::black_box as bb;

const NITER: usize = 100_000;

macro_rules! gen_benches {
    ($T:ty, $new: expr) => {
        #[inline(never)]
        fn test_data() -> Vec<u8> {
            let mut raw = Vec::new();
            for i in 0usize..NITER {
                let arr = $new(i);
                arr.write(&mut raw).unwrap();
            }
            raw
        }

        #[bench]
        fn read(b: &mut Bencher) {
            let raw = test_data();
            b.iter(|| {
                for i in 0usize..NITER {
                    bb(<$T>::read(&raw[i * <$T>::n_bytes()..]));
                }
            });
        }

        #[bench]
        fn write(b: &mut Bencher) {
            b.iter(|| {
                bb(test_data())
            });
        }
    };
}

mod simple {
    use super::*;

    #[derive(Serializable, Debug, PartialEq)]
    struct Simple {
        a: i32,
        b: f32,
    }

    gen_benches!(Simple, |i| Simple { a: i as i32, b: i as f32 });
}

mod one_field {
    use super::*;

    #[derive(Serializable, Debug, PartialEq)]
    struct OneField {
        a: i32,
    }

    gen_benches!(OneField, |i| OneField { a: i as i32 });
}

mod array {
    use super::*;

    #[derive(Serializable, Debug, PartialEq)]
    struct Array {
        a: [f32; 8],
    }

    gen_benches!(Array, |i| Array { a: [i as f32; 8] });
}

mod plain_f32 {
    use super::*;

    gen_benches!(f32, |i| i as f32);
}
