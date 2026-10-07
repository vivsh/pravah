use super::*;
use serde_json::json;

/// Nullable fields and optional fields retain different constraints, including nested closure.
#[test]
fn nullability_requiredness_and_additional_properties() {
    let schema = json!({"type":"object", "required":["email"], "additionalProperties":false,
        "properties":{"email":{"type":["string","null"]}, "on":{"type":"string","format":"date"},
        "settings":{"type":"object","additionalProperties":{"type":"integer"}}}});
    let validator = compile(&schema, "staff", "input").expect("schema");
    assert!(validator.is_valid(&json!({"email":null})));
    assert!(validator.is_valid(&json!({"email":"x","settings":{"limit":2}})));
    for value in [
        json!({}),
        json!({"email":null,"extra":1}),
        json!({"email":null,"on":null}),
        json!({"email":null,"settings":{"limit":"2"}}),
    ] {
        assert!(!validator.is_valid(&value));
    }
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(
        schema["properties"]["email"]["type"],
        json!(["string", "null"])
    );
}

/// Unsupported schemas fail admission instead of silently ignoring constraints or retrieving data.
#[test]
fn unsupported_schema_semantics_are_rejected() {
    for schema in [
        json!(null),
        json!({"$schema":"http://json-schema.org/draft-07/schema#"}),
        json!({"nullable":true}),
        json!({"definitions":{}}),
        json!({"$id":"https://example.com"}),
        json!({"$anchor":"node"}),
        json!({"$dynamicRef":"#node"}),
        json!({"customConstraint":true}),
        json!({"format":"custom"}),
        json!({"contentEncoding":"base64"}),
        json!({"properties":{"x":{"$schema":DIALECT}}}),
        json!({"$ref":"https://example.invalid/schema"}),
        json!({"$ref":"file:///private/schema"}),
        json!({"$ref":"#/$defs/missing"}),
        json!({"$ref":"#/enum/0", "enum":[{}]}),
        json!({"$defs":{"x":{"$ref":"#/$defs/x"}},"$ref":"#/$defs/x"}),
    ] {
        assert!(compile(&schema, "tool", "input").is_err(), "{schema}");
    }
}

/// Local pointers retain escaped names and assertion-bearing 2020-12 ref siblings.
#[test]
fn escaped_references_and_siblings_are_supported() {
    let schema = json!({"$schema":DIALECT, "$defs":{"a/b~c":{"type":"integer"}},
        "type":"object", "properties":{"$ref":{"$ref":"#/$defs/a~1b~0c","minimum":2}},
        "required":["$ref"]});
    let validator = compile(&schema, "lookup", "input").expect("schema");
    assert!(validator.is_valid(&json!({"$ref":2})));
    assert!(!validator.is_valid(&json!({"$ref":1})));
    assert!(!validator.is_valid(&json!({"$ref":"2"})));
}

/// Literal schema-like JSON values and schema-keyword property names are never rewritten.
#[test]
fn literal_values_are_not_schemas() {
    let literal =
        json!({"additionalProperties":false,"$ref":"https://private.invalid","nullable":true});
    let schema = json!({"type":"object", "const":literal,"default":literal,"examples":[literal],
        "properties":{"additionalProperties":{"type":"boolean"},"$ref":{"type":"string"},"nullable":true}});
    let validator = compile(&schema, "literal", "output").expect("schema");
    assert!(validator.is_valid(&literal));
    assert!(!validator.is_valid(&json!({"nullable":true})));
}

/// Standard formats are asserted rather than merely accepted as annotations.
#[test]
fn known_formats_are_asserted() {
    for (format, valid) in [
        ("date", "2026-10-07"),
        ("date-time", "2026-10-07T12:00:00Z"),
        ("uuid", "550e8400-e29b-41d4-a716-446655440000"),
    ] {
        let validator = compile(
            &json!({"type":"string","format":format}),
            "format",
            "output",
        )
        .expect("schema");
        assert!(validator.is_valid(&json!(valid)));
        assert!(!validator.is_valid(&json!("invalid")));
    }
}

/// Meta-schema validation rejects malformed assertion values after profile admission.
#[test]
fn malformed_constraints_and_composition_are_checked() {
    assert!(compile(&json!({"required":"x"}), "bad", "input").is_err());
    let schema = json!({"type":"object", "properties":{"state":{"enum":["open","closed"]}},
        "required":["state"], "allOf":[{"if":{"properties":{"state":{"const":"closed"}}},
        "then":{"required":["reason"]}}], "unevaluatedProperties":false,
        "patternProperties":{"^reason$":{"type":"string","minLength":1}}});
    let validator = compile(&schema, "state", "input").expect("schema");
    assert!(validator.is_valid(&json!({"state":"open"})));
    assert!(validator.is_valid(&json!({"state":"closed","reason":"done"})));
    assert!(!validator.is_valid(&json!({"state":"closed"})));
    assert!(!validator.is_valid(&json!({"state":"unknown"})));
    assert!(!validator.is_valid(&json!({"state":"open","extra":1})));
}
