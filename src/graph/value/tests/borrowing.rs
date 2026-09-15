use serde::{Deserialize, Serialize};

use super::super::{Value, ValueError, from_value, to_value};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Item {
    Empty,
    Scalar(u64),
    Pair(bool, i64),
    Fields { flag: bool },
}

/// Traversing shared arrays, objects and enum identifiers needs no scratch allocations.
#[test]
fn shared_container_decode_allocates_only_the_typed_result() -> Result<(), ValueError> {
    let expected = [
        Item::Empty,
        Item::Scalar(7),
        Item::Pair(true, -2),
        Item::Fields { flag: false },
    ];
    let value = to_value(&expected)?;
    let mut decoded = Ok([Item::Empty, Item::Empty, Item::Empty, Item::Empty]);
    let allocations = allocation_counter::measure(|| {
        decoded = from_value::<[Item; 4]>(value.clone());
    });
    assert_eq!(decoded?, expected);
    assert_eq!(allocations.count_total, 0);
    assert_eq!(from_value::<[Item; 4]>(value)?, expected);
    Ok(())
}

/// Borrowed traversal retains ordinary owned String and Option decoding semantics.
#[test]
fn borrowed_decode_matches_consuming_decode() -> Result<(), ValueError> {
    let expected = (
        Some("hello".to_owned()),
        None::<String>,
        vec!["world".to_owned()],
    );
    let value = to_value(&expected)?;
    let borrowed = <(Option<String>, Option<String>, Vec<String>)>::deserialize(&value)?;
    assert_eq!(borrowed, expected);
    assert_eq!(
        from_value::<(Option<String>, Option<String>, Vec<String>)>(value)?,
        expected
    );
    Ok(())
}

/// Malformed enum variants fail without modifying the shared source value.
#[test]
fn invalid_enum_remains_reusable_after_decode_error() -> Result<(), ValueError> {
    let value = Value::object([("Pair", Value::array([Value::from(true)]))])?;
    let original = value.clone();
    assert!(from_value::<Item>(value.clone()).is_err());
    assert_eq!(value, original);
    Ok(())
}
