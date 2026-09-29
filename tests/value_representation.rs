//! Public value behavior across static Serde fields and runtime-owned object keys.

use std::collections::BTreeMap;

use pravah::GraphError;
use pravah::graph::{Value, ValueError, from_value, to_value};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Fields {
    alpha: u64,
    beta: bool,
    gamma: i64,
    delta: u32,
}

fn fields() -> Fields {
    Fields {
        alpha: 1,
        beta: true,
        gamma: -2,
        delta: 3,
    }
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::ValueConversion {
        target: "test codec".into(),
        reason: error.to_string(),
    }
}

/// Static scalar fields allocate only the shared object container, never one allocation per key.
#[test]
fn static_fields_and_clones_avoid_key_allocations() -> Result<(), GraphError> {
    let mut encoded = Ok(Value::NULL);
    let measured = allocation_counter::measure(|| encoded = to_value(fields()));
    let value = encoded.map_err(codec)?;
    assert_eq!(measured.count_total, 2);
    let clone = allocation_counter::measure(|| {
        std::hint::black_box(value.clone());
    });
    assert_eq!(clone.count_total, 0);
    assert_eq!(from_value::<Fields>(value).map_err(codec)?, fields());
    Ok(())
}

/// Dynamic keys outlive their inputs and share ordering, equality and wire output with static keys.
#[test]
fn owned_and_static_keys_have_identical_behavior() -> Result<(), GraphError> {
    let owned = Value::object([
        (String::from("delta"), Value::from(3_u32)),
        (String::from("beta"), Value::from(true)),
        (String::from("gamma"), Value::from(-2_i64)),
        (String::from("alpha"), Value::from(1_u64)),
    ])
    .map_err(codec)?;
    let borrowed = to_value(fields()).map_err(codec)?;
    assert_eq!(owned, borrowed);
    assert_eq!(
        serde_json::to_vec(&owned).map_err(codec)?,
        serde_json::to_vec(&borrowed).map_err(codec)?
    );
    let keys = owned
        .object_entries()
        .ok_or_else(|| codec("object"))?
        .map(|(key, _)| key)
        .collect::<Vec<_>>();
    assert_eq!(keys, ["alpha", "beta", "delta", "gamma"]);
    assert!(matches!(
        Value::object([("key", Value::NULL), ("key", Value::NULL)]),
        Err(ValueError::DuplicateKey(_))
    ));
    assert!(serde_json::from_str::<Value>(r#"{"key":1,"key":2}"#).is_err());
    Ok(())
}

/// Unicode and escaped dynamic keys survive source destruction and both supported codecs.
#[test]
fn dynamic_keys_roundtrip_json_and_cbor() -> Result<(), GraphError> {
    let value = {
        let source = BTreeMap::from([(String::from("नमस्ते\n\""), fields())]);
        to_value(&source).map_err(codec)?
    };
    let json: Value =
        serde_json::from_slice(&serde_json::to_vec(&value).map_err(codec)?).map_err(codec)?;
    let mut bytes = Vec::new();
    ciborium::into_writer(&value, &mut bytes).map_err(codec)?;
    let cbor: Value = ciborium::from_reader(bytes.as_slice()).map_err(codec)?;
    assert_eq!(value, json);
    assert_eq!(value, cbor);
    let decoded: BTreeMap<String, Fields> = from_value(cbor).map_err(codec)?;
    assert_eq!(decoded.get("नमस्ते\n\""), Some(&fields()));
    Ok(())
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Variant {
    Unit,
    Newtype(u64),
    Tuple(u64, bool),
    Struct { alpha: u64 },
}

/// Enum tags roundtrip and numeric map keys retain their established string-key encoding.
#[test]
fn variants_and_numeric_map_keys_preserve_encoding() -> Result<(), GraphError> {
    for variant in [
        Variant::Unit,
        Variant::Newtype(1),
        Variant::Tuple(2, true),
        Variant::Struct { alpha: 3 },
    ] {
        let value = to_value(&variant).map_err(codec)?;
        assert_eq!(
            serde_json::to_value(&value).map_err(codec)?,
            serde_json::to_value(&variant).map_err(codec)?
        );
        assert_eq!(from_value::<Variant>(value).map_err(codec)?, variant);
    }
    let source = BTreeMap::from([(-4_i64, 1_u64), (9, 2)]);
    let encoded = to_value(&source).map_err(codec)?;
    assert_eq!(
        from_value::<BTreeMap<String, u64>>(encoded).map_err(codec)?,
        BTreeMap::from([("-4".into(), 1), ("9".into(), 2)])
    );
    Ok(())
}
