use super::sanitize_strict;
use serde_json::json;

/// Canonical nested tool schemas stay unchanged when Rath's helper becomes private.
#[test]
fn canonical_schema_preserves_authored_shape() -> Result<(), String> {
    let schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Query", "type": "object", "additionalProperties": false,
        "$defs": {"Filter": {"type": "object", "properties": {
            "title": {"type": ["string", "null"]}
        }}},
        "properties": {"filter": {"$ref": "#/$defs/Filter"}},
        "required": ["filter"]
    });
    assert_eq!(
        sanitize_strict(schema)?,
        json!({
            "type": "object", "properties": {
                "filter": {"type": "object", "properties": {
                    "title": {"type": "string"}
                }}
            }, "required": ["filter"]
        })
    );
    Ok(())
}

/// Invalid or recursive references produce errors instead of incomplete tool schemas.
#[test]
fn unsafe_references_are_rejected() {
    for schema in [
        json!({"$ref": "#/$defs/missing"}),
        json!({"$ref": "https://example.test/schema"}),
        json!({"$ref": 3}),
        json!({"$defs": {"A": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"}),
        json!({"$defs": {"A": {"type": "string"}}, "$ref": "#/$defs/A", "maxLength": 2}),
    ] {
        assert!(sanitize_strict(schema).is_err());
    }
}

/// Property names resembling schema keywords remain ordinary property names.
#[test]
fn keyword_properties_and_escaped_refs_are_preserved() -> Result<(), String> {
    assert_eq!(
        sanitize_strict(json!({
            "type": "object", "$defs": {"A/B~C": {"type": "boolean"}},
            "properties": {"$ref": {"$ref": "#/$defs/A~1B~0C"}}
        }))?,
        json!({"type": "object", "properties": {"$ref": {"type": "boolean"}}})
    );
    Ok(())
}
