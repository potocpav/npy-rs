
use nom::IResult;
use std::collections::HashMap;

/// Representation of a Numpy type
#[derive(PartialEq, Eq, Debug)]
pub struct DType {
    /// Numpy type string. First character is `'>'` for big endian, `'<'` for little endian.
    ///
    /// Examples: `>i4`, `<u8`, `>f8`. The number corresponds to the number of bytes.
    pub ty: &'static str,

    /// Shape of a type.
    ///
    /// Scalar has zero entries. Otherwise, number of entries == number of dimensions and each
    /// entry specifies size in the respective dimension.
    pub shape: Vec<u64>,
}

/// To avoid exporting the `to_value` function, it is on a separate trait.
pub trait DTypeToValue {
    fn to_value(&self, name: &str) -> Value;
}

impl DTypeToValue for DType {
    fn to_value(&self, name: &str) -> Value {
        if self.shape.is_empty() { // scalar
            Value::List(vec![
                Value::String(name.into()),
                Value::String(self.ty.into()),
            ])
        } else {
            Value::List(vec![
                Value::String(name.into()),
                Value::String(self.ty.into()),
                Value::List(self.shape.iter().map(|&n| Value::Integer(n as i64)).collect::<Vec<_>>()),
            ])
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum Value {
    String(String),
    Integer(i64),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String,Value>),
}

pub fn parse_header(bs: &[u8]) -> IResult<&[u8], Value> {
    parser::header(bs)
}

mod parser {
    use super::Value;
    use nom::*;

    named!(pub header<Value>,
        do_parse!(
            tag!(&[0x93u8]) >>
            tag!(b"NUMPY") >>
            tag!(&[0x01u8, 0x00]) >>
            hdr: length_value!(le_u16, item) >>
            (hdr)
        )
    );


    named!(pub integer<Value>,
        map!(
            map_res!(
                map_res!(
                    ws!(digit),
                    ::std::str::from_utf8
                ),
                ::std::str::FromStr::from_str
            ),
            Value::Integer
        )
    );

    named!(pub boolean<Value>,
        ws!(alt!(
            tag!("True") => { |_| Value::Bool(true) } |
            tag!("False") => { |_| Value::Bool(false) }
        ))
    );

    named!(pub string<Value>,
        map!(
            map!(
                map_res!(
                    ws!(alt!(
                        delimited!(tag!("\""),
                            is_not_s!("\""),
                            tag!("\"")) |
                        delimited!(tag!("\'"),
                            is_not_s!("\'"),
                            tag!("\'"))
                        )),
                    ::std::str::from_utf8
                ),
                |s: &str| s.to_string()
            ),
            Value::String
        )
    );

    named!(pub item<Value>, alt!(integer | boolean | string | list | map));

    named!(pub list<Value>,
        map!(
            ws!(alt!(
                delimited!(tag!("["),
                    terminated!(separated_list!(tag!(","), item), alt!(tag!(",") | tag!(""))),
                    tag!("]")) |
                delimited!(tag!("("),
                    terminated!(separated_list!(tag!(","), item), alt!(tag!(",") | tag!(""))),
                    tag!(")"))
            )),
            Value::List
        )
    );

    named!(pub map<Value>,
        map!(
            ws!(
                delimited!(tag!("{"),
                    terminated!(separated_list!(tag!(","),
                        separated_pair!(map_opt!(string, |it| match it { Value::String(s) => Some(s), _ => None }), tag!(":"), item)
                    ), alt!(tag!(",") | tag!(""))),
                    tag!("}"))
            ),
            |v: Vec<_>| Value::Map(v.into_iter().collect())
        )
    );
}

// #[test]
// fn parse_header() {
//     assert_eq!(integer(b"1234  "), IResult::Done(&b""[..], Integer(1234)));
//     assert_eq!(string(br#" "Hello"   "#), IResult::Done(&b""[..], String("Hello".into())));
//     assert_eq!(string(br#" 'World!'   "#), IResult::Done(&b""[..], String("World!".into())));
//     assert_eq!(boolean(b"  True"), IResult::Done(&b""[..], Bool(true)));
//     assert_eq!(boolean(b"False "), IResult::Done(&b""[..], Bool(false)));
//     assert_eq!(list(b" ()"), IResult::Done(&b""[..], List(vec![]))); // FIXME: Make this not parse as a List
//     assert_eq!(list(b" (4)"), IResult::Done(&b""[..], List(vec![Integer(4)]))); // FIXME: Make this not parse as a List
//     assert_eq!(list(b" (1 , 2 ,)"), IResult::Done(&b""[..], List(vec![Integer(1), Integer(2)])));
//     assert_eq!(list(b" [5 , 6 , 7]"), IResult::Done(&b""[..], List(vec![Integer(5), Integer(6), Integer(7)])));
// }
