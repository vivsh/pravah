use serde::{Deserialize, Serialize};

use super::*;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Fixture {
    name: String,
    signed: i64,
    unsigned: u64,
    enabled: bool,
    values: Vec<f64>,
    optional: Option<String>,
}

/// Verifies every supported value kind survives JSON serialization.
#[test]
fn value_round_trips_through_json() {
    let value = Value::object([
        (
            "array",
            Value::array([Value::from(1_i64), Value::number(2.5).expect("finite")]),
        ),
        ("bool", Value::from(true)),
        ("null", Value::NULL),
        ("string", Value::from("pravah")),
    ])
    .expect("unique keys");
    let encoded = serde_json::to_string(&value).expect("encode runtime value");
    let decoded: Value = serde_json::from_str(&encoded).expect("decode runtime value");
    assert_eq!(decoded, value);
}

/// Verifies typed Serde conversion does not require a JSON intermediary.
#[test]
fn typed_value_conversion_round_trips() {
    let fixture = Fixture {
        name: "flow".into(),
        signed: -4,
        unsigned: u64::MAX,
        enabled: true,
        values: vec![1.25, 2.5],
        optional: None,
    };
    let value = to_value(&fixture).expect("encode fixture");
    let decoded: Fixture = from_value(value).expect("decode fixture");
    assert_eq!(decoded, fixture);
}

/// Verifies object construction sorts keys and rejects duplicates.
#[test]
fn objects_are_deterministic_and_unique() {
    let object = Value::object([("z", Value::NULL), ("a", Value::NULL)]).expect("unique");
    let keys = object
        .object_entries()
        .expect("object")
        .map(|(key, _)| key)
        .collect::<Vec<_>>();
    assert_eq!(keys, ["a", "z"]);
    assert_eq!(
        Value::object([("a", Value::NULL), ("a", Value::NULL)]),
        Err(ValueError::DuplicateKey("a".into()))
    );
}

/// Verifies unsupported numeric values fail with structured errors.
#[test]
fn numeric_domain_is_checked() {
    assert_eq!(Value::number(f64::NAN), Err(ValueError::NonFiniteNumber));
    assert_eq!(to_value(u128::MAX), Err(ValueError::IntegerOutOfRange));
    assert_eq!(to_value(i128::MIN), Err(ValueError::IntegerOutOfRange));
}

/// Verifies cloning composite values shares their immutable backing storage.
#[test]
fn composite_clones_share_storage() {
    let string = Value::from("shared");
    let array = Value::array([string]);
    let object = Value::object([("items", array)]).expect("unique key");
    let cloned = object.clone();
    let (Repr::Object(first), Repr::Object(second)) = (&object.0, &cloned.0) else {
        panic!("fixture should be an object");
    };
    assert!(Arc::ptr_eq(first, second));
}

/// Verifies runtime values preserve numeric meaning through CBOR.
#[test]
fn value_round_trips_through_cbor() {
    let value = Value::object([
        ("signed", Value::from(-7_i64)),
        ("unsigned", Value::from(u64::MAX)),
        ("number", Value::number(1.25).expect("finite")),
    ])
    .expect("unique keys");
    let mut encoded = Vec::new();
    ciborium::into_writer(&value, &mut encoded).expect("encode CBOR");
    let decoded: Value = ciborium::from_reader(encoded.as_slice()).expect("decode CBOR");
    assert_eq!(decoded, value);
}
