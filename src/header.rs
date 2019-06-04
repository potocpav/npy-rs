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
        shape: Vec<u64>,
    },

    /// A structure record array
    Record(Vec<Field>),
}

#[derive(PartialEq, Eq, Debug)]
/// A field of a record dtype
pub struct Field {
    /// The name of the field
    pub name: String,

    /// The dtype of the field
    pub dtype: DType,
}

impl DType {
    /// Numpy format description of record dtype.
    pub fn descr(&self) -> String {
        use DType::*;
        match *self {
            Record(ref fields) => {
                fields
                    .iter()
                    .map(
                        |&Field {
                             ref name,
                             ref dtype,
                         }| match *dtype {
                            Plain { ref ty, ref shape } => {
                                if shape.len() == 0 {
                                    format!("('{}', '{}'), ", name, ty)
                                } else {
                                    let shape_str = shape
                                        .iter()
                                        .fold(String::new(), |o, n| o + &format!("{},", n));
                                    format!("('{}', '{}', ({})), ", name, ty, shape_str)
                                }
                            }
                            ref record @ Record(_) => format!("('{}', {}), ", name, record.descr()),
                        },
                    )
                    .fold("[".to_string(), |o, n| o + &n)
                    + "]"
            }
            Plain { ref ty, .. } => format!("'{}'", ty),
        }
    }

    /// Create from description AST
    pub fn from_descr(descr: Value) -> Result<Self> {
        use DType::*;
        match descr {
            Value::String(string) => Ok(Plain {
                ty: string,
                shape: vec![],
            }),
            Value::List(ref list) => Ok(Record(convert_list_to_record_fields(list)?)),
            _ => invalid_data("must be string or list"),
        }
    }
}

fn convert_list_to_record_fields(values: &[Value]) -> Result<Vec<Field>> {
    first_error(values.iter().map(|value| match *value {
        Value::List(ref tuple) => convert_tuple_to_record_field(tuple),
        _ => invalid_data("list must contain list or tuple"),
    }))
}

fn convert_tuple_to_record_field(tuple: &[Value]) -> Result<Field> {
    use self::Value::{List, String};

    match tuple.len() {
        2 | 3 => match (&tuple[0], &tuple[1], tuple.get(2)) {
            (&String(ref name), &String(ref dtype), ref shape) => Ok(Field {
                name: name.clone(),
                dtype: DType::Plain {
                    ty: dtype.clone(),
                    shape: if let &Some(ref s) = shape {
                        convert_value_to_shape(s)?
                    } else {
                        vec![]
                    },
                },
            }),
            (&String(ref name), &List(ref list), None) => Ok(Field {
                name: name.clone(),
                dtype: DType::Record(convert_list_to_record_fields(list)?),
            }),
            (&String(_), &List(_), Some(_)) => {
                invalid_data("nested arrays of Record types are not supported.")
            }
            _ => invalid_data("list entry must contain a string for id and a valid dtype"),
        },
        _ => invalid_data("list entry must contain 2 or 3 items"),
    }
}

fn convert_value_to_shape(field: &Value) -> Result<Vec<u64>> {
    if let Value::List(ref lengths) = *field {
        first_error(lengths.iter().map(convert_value_to_positive_integer))
    } else {
        invalid_data("shape must be list or tuple")
    }
}

fn convert_value_to_positive_integer(number: &Value) -> Result<u64> {
    if let Value::Integer(number) = *number {
        if number > 0 {
            Ok(number as u64)
        } else {
            invalid_data("number must be positive")
        }
    } else {
        invalid_data("must be a number")
    }
}

fn first_error<I, T>(results: I) -> Result<Vec<T>>
where
    I: IntoIterator<Item = Result<T>>,
{
    let mut vector = vec![];
    for result in results {
        vector.push(result?);
    }
    Ok(vector)
}

fn invalid_data<T>(message: &str) -> Result<T> {
    use std::io::{Error, ErrorKind};
    Err(Error::new(ErrorKind::InvalidData, message.to_string()))
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Value {
    String(String),
    Integer(i64),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn description_of_record_array_as_python_list_of_tuples() {
        let dtype = DType::Record(vec![
            Field {
                name: "float".to_string(),
                dtype: DType::Plain {
                    ty: ">f4".to_string(),
                    shape: vec![],
                },
            },
            Field {
                name: "byte".to_string(),
                dtype: DType::Plain {
                    ty: "<u1".to_string(),
                    shape: vec![],
                },
            },
        ]);
        let expected = "[('float', '>f4'), ('byte', '<u1'), ]";
        assert_eq!(dtype.descr(), expected);
    }

    #[test]
    fn description_of_unstructured_primitive_array() {
        let dtype = DType::Plain {
            ty: ">f8".to_string(),
            shape: vec![],
        };
        assert_eq!(dtype.descr(), "'>f8'");
    }

    #[test]
    fn description_of_nested_record_dtype() {
        let dtype = DType::Record(vec![Field {
            name: "parent".to_string(),
            dtype: DType::Record(vec![Field {
                name: "child".to_string(),
                dtype: DType::Plain {
                    ty: "<i4".to_string(),
                    shape: vec![],
                },
            }]),
        }]);
        assert_eq!(dtype.descr(), "[('parent', [('child', '<i4'), ]), ]");
    }

    #[test]
    fn converts_simple_description_to_record_dtype() {
        let dtype = ">f8".to_string();
        assert_eq!(
            DType::from_descr(Value::String(dtype.clone())).unwrap(),
            DType::Plain {
                ty: dtype,
                shape: vec![]
            }
        );
    }

    #[test]
    fn converts_record_description_to_record_dtype() {
        let descr = parse("[('a', '<u2'), ('b', '<f4')]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "a".to_string(),
                dtype: DType::Plain {
                    ty: "<u2".to_string(),
                    shape: vec![],
                },
            },
            Field {
                name: "b".to_string(),
                dtype: DType::Plain {
                    ty: "<f4".to_string(),
                    shape: vec![],
                },
            },
        ]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }

    #[test]
    fn record_description_with_onedimensional_field_shape_declaration() {
        let descr = parse("[('a', '>f8', (1,))]");
        let expected_dtype = DType::Record(vec![Field {
            name: "a".to_string(),
            dtype: DType::Plain {
                ty: ">f8".to_string(),
                shape: vec![1],
            },
        }]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }

    #[test]
    fn record_description_with_nested_record_field() {
        let descr = parse("[('parent', [('child', '<i4')])]");
        let expected_dtype = DType::Record(vec![Field {
            name: "parent".to_string(),
            dtype: DType::Record(vec![Field {
                name: "child".to_string(),
                dtype: DType::Plain {
                    ty: "<i4".to_string(),
                    shape: vec![],
                },
            }]),
        }]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }

    #[test]
    fn errors_on_nested_record_field_array() {
        let descr = parse("[('parent', [('child', '<i4')], (2,))]");
        assert!(DType::from_descr(descr).is_err());
    }

    #[test]
    fn errors_on_value_variants_that_cannot_be_converted() {
        let no_dtype = Value::Bool(false);
        assert!(DType::from_descr(no_dtype).is_err());
    }

    #[test]
    fn errors_when_record_list_does_not_contain_lists() {
        let faulty_list = parse("['a', 123]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_few_items() {
        let faulty_list = parse("[('a')]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_many_items() {
        let faulty_list = parse("[('a', 1, 2, 3)]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_non_strings_for_id_or_dtype() {
        let faulty_list = parse("[(1, 2)]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_shape_is_not_a_list() {
        let no_shape = parse("1");
        assert!(convert_value_to_shape(&no_shape).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_not_a_number() {
        let no_number = parse("[]");
        assert!(convert_value_to_positive_integer(&no_number).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_not_positive() {
        assert!(convert_value_to_positive_integer(&parse("0")).is_err());
    }

    fn parse(source: &str) -> Value {
        parser::item(source.as_bytes())
            .to_result()
            .expect("could not parse Python expression")
    }
}
