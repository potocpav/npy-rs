
use nom::IResult;
use std::collections::HashMap;
use std::io::Result;

/// Representation of a Numpy type
#[derive(PartialEq, Eq, Debug)]
pub enum DType {
    /// A simple array with only a single field
    Plain {
        /// Numpy type string. First character is `'>'` for big endian, `'<'` for little endian.
        ///
        /// Examples: `>i4`, `<u8`, `>f8`. The number corresponds to the number of bytes.
        ty: String,

        /// Shape of a type.
        ///
        /// Scalar has zero entries. Otherwise, number of entries == number of dimensions and each
        /// entry specifies size in the respective dimension.
        shape: Vec<u64>
    },

    /// A structure record array
    Record(Vec<(String, DType)>)
}

/// To avoid exporting the `to_value` function, it is on a separate trait.
pub trait DTypeToValue {
    fn to_value(&self, name: &str) -> Value;
}

impl DTypeToValue for DType {
    fn to_value(&self, name: &str) -> Value {
        use DType::*;
        match *self {
            Plain { ref shape, ref ty } =>
                if shape.is_empty() { // scalar
                    Value::List(vec![
                        Value::String(name.into()),
                        Value::String(ty.clone()),
                    ])
                } else {
                    Value::List(vec![
                        Value::String(name.into()),
                        Value::String(ty.clone()),
                        Value::List(shape.iter().map(|&n| Value::Integer(n as i64)).collect::<Vec<_>>()),
                    ])
                },
            Record(_) => unimplemented!("DType::Record -> Value ")
        }
    }
}
impl DType {
    /// Numpy format description of record dtype.
    pub fn descr(&self) -> String {
        use DType::*;
        match *self {
            Record(ref fields) =>
                fields.iter()
                    .map(|&(ref id, ref t)|
                        match *t {
                            Plain { ref ty, ref shape } =>
                                if shape.len() == 0 {
                                    format!("('{}', '{}'), ", id, ty)
                                } else {
                                    let shape_str = shape.iter().fold(String::new(), |o,n| o + &format!("{},", n));
                                    format!("('{}', '{}', ({})), ", id, ty, shape_str)
                                },
                            Record(_) => unimplemented!("nested record dtypes")
                        }
                    )
                    .fold("[".to_string(), |o, n| o + &n) + "]",
            Plain { ref ty, .. } => format!("'{}'", ty),
        }
    }

    /// Create from description AST
    pub fn from_descr(descr: Value) -> Result<Self> {
        use DType::*;
        match descr {
            Value::String(string) => Ok(Plain { ty: string, shape: vec![] }),
            Value::List(values) => Ok(Record(from_list(values)?)),
            _ => unimplemented!()
        }
    }
}

fn from_list(values: Vec<Value>) -> Result<Vec<(String, DType)>> {
    let mut pairs = vec![];
    for value in values {
        if let Value::List(field) = value {
            pairs.push(convert_field(field)?);
        } else {
            unimplemented!()
        }
    }
    Ok(pairs)
}

fn convert_field(field: Vec<Value>) -> Result<(String, DType)> {
    use self::Value::String;
    match (&field[0], &field[1], field.len()) {
        (&String(ref id), &String(ref t), 2) =>
            Ok((id.clone(), DType::Plain { ty: t.clone(), shape: vec![] })),
        (&String(ref id), &String(ref t), 3) =>
            Ok((id.clone(), DType::Plain { ty: t.clone(), shape: convert_shape(&field[2])? })),
        _ => unimplemented!()
    }
}

fn convert_shape(field: &Value) -> Result<Vec<u64>> {
    if let Value::List(ref lengths) = *field {
        let mut numbers = vec![];
        for length in lengths {
            if let Value::Integer(number) = *length {
                if number > 0 {
                    numbers.push(number as u64);
                } else {
                    unimplemented!()
                }
            } else {
                unimplemented!()
            }
        }
        Ok(numbers)
    } else {
        unimplemented!()
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn description_of_record_array_as_python_list_of_tuples() {
        let dtype = DType::Record(vec![
            ("float".to_string(), DType::Plain { ty: ">f4".to_string(), shape: vec![] }),
            ("byte".to_string(), DType::Plain { ty: "<u1".to_string(), shape: vec![] }),
        ]);
        let expected = "[('float', '>f4'), ('byte', '<u1'), ]";
        assert_eq!(dtype.descr(), expected);
    }

    #[test]
    fn description_of_unstructured_primitive_array() {
        let dtype = DType::Plain { ty: ">f8".to_string(), shape: vec![] };
        assert_eq!(dtype.descr(), "'>f8'");
    }

    #[test]
    fn converts_simple_description_to_record_dtype() {
        let dtype = ">f8".to_string();
        assert_eq!(
            DType::from_descr(Value::String(dtype.clone())).unwrap(),
            DType::Plain { ty: dtype, shape: vec![] }
        );
    }

    #[test]
    fn converts_record_description_to_record_dtype() {
        let descr = parser::item(b"[('a', '<u2'), ('b', '<f4')]").to_result().unwrap();
        let expected_dtype = DType::Record(vec![
            ("a".to_string(), DType::Plain { ty: "<u2".to_string(), shape: vec![] }),
            ("b".to_string(), DType::Plain { ty: "<f4".to_string(), shape: vec![] }),
        ]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }

    #[test]
    fn record_description_with_onedimenional_field_shape_declaration() {
        let descr = parser::item(b"[('a', '>f8', (1,))]").to_result().unwrap();
        let expected_dtype = DType::Record(vec![
            ("a".to_string(), DType::Plain { ty: ">f8".to_string(), shape: vec![1] }),
        ]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }
}
