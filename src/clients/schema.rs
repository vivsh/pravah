#![allow(dead_code)]

use serde_json::{Map, Value};

/// Reduces a full JSON Schema to the stricter subset accepted by Gemini.
///
/// This inlines root definitions, drops unsupported schema annotations, and
/// normalizes nullable types into the non-null representation Gemini expects.
#[allow(dead_code)]
pub(super) fn sanitize_strict(schema: Value) -> Value {
    let defs = collect_defs(&schema);
    let without_refs = inline_refs(schema, &defs);
    clean_fields(without_refs)
}

/// Collects all entries from `definitions` or `$defs` at the root level.
fn collect_defs(schema: &Value) -> Map<String, Value> {
    let mut defs = Map::new();
    if let Some(obj) = schema.as_object() {
        for key in &["definitions", "$defs"] {
            if let Some(Value::Object(map)) = obj.get(*key) {
                defs.extend(map.clone());
            }
        }
    }
    defs
}

/// Recursively replace every `{"$ref": "#/definitions/Foo"}` (or `#/$defs/Foo`)
/// with the corresponding definition, then recurse into the substituted value.
fn inline_refs(value: Value, defs: &Map<String, Value>) -> Value {
    match value {
        Value::Object(mut map) => {
            if let Some(Value::String(ref_str)) = map.get("$ref").cloned() {
                if let Some(name) = ref_name(&ref_str) {
                    if let Some(target) = defs.get(name) {
                        return inline_refs(target.clone(), defs);
                    }
                }
            }
            let entries: Vec<(String, Value)> = map
                .iter_mut()
                .map(|(k, v)| (k.clone(), inline_refs(v.clone(), defs)))
                .collect();
            Value::Object(entries.into_iter().collect())
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(|v| inline_refs(v, defs)).collect()),
        other => other,
    }
}

/// Extract the definition name from a `$ref` string like `#/definitions/Foo`.
fn ref_name(r: &str) -> Option<&str> {
    r.split('/').last()
}

/// Recursively remove schema annotation keywords and normalise nullable types.
///
/// Keys inside a `"properties"` map are property names, not keywords, and
/// must not be filtered even if they coincide with a keyword (e.g. a struct
/// field literally named `title`). `clean_properties` handles that case.
///
/// Also injects `"properties": {}` on any `type: object` node that lacks one —
/// Gemini rejects object schemas without an explicit `properties` key.
fn clean_fields(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Map<String, Value> = map
                .into_iter()
                .filter(|(k, _)| {
                    !matches!(
                        k.as_str(),
                        "$schema"
                            | "title"
                            | "definitions"
                            | "$defs"
                            | "additionalProperties"
                            | "$ref"
                    )
                })
                .map(|(k, v)| {
                    let v = match k.as_str() {
                        "type" => normalize_type(v),
                        "properties" => clean_properties(v),
                        _ => clean_fields(v),
                    };
                    (k, v)
                })
                .collect();

            if entries.get("type").and_then(|v| v.as_str()) == Some("object")
                && !entries.contains_key("properties")
            {
                entries.insert("properties".to_owned(), Value::Object(Map::new()));
            }

            Value::Object(entries)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(clean_fields).collect()),
        other => other,
    }
}

/// Process a `"properties"` value: recurse into each property's schema
/// without filtering the property name keys themselves.
fn clean_properties(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            Value::Object(map.into_iter().map(|(k, v)| (k, clean_fields(v))).collect())
        }
        other => other,
    }
}

/// Rewrites `["T", "null"]` (or `["null", "T"]`) to `"T"`.
/// Leaves all other type values untouched.
fn normalize_type(value: Value) -> Value {
    if let Value::Array(types) = &value {
        let non_null: Vec<&Value> = types
            .iter()
            .filter(|v| v.as_str() != Some("null"))
            .collect();
        if non_null.len() == 1 {
            return non_null[0].clone();
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `sanitize_strict` removes `$schema` and `title` from the root and nested schemas.
    #[test]
    fn strips_schema_and_title() {
        let input = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "MyTool",
            "type": "object",
            "properties": {
                "name": { "title": "Name", "type": "string" }
            }
        });
        let out = sanitize_strict(input);
        assert!(out.get("$schema").is_none());
        assert!(out.get("title").is_none());
        assert!(out["properties"]["name"].get("title").is_none());
    }

    /// `sanitize_strict` inlines `$ref` entries from `definitions` and removes the definitions key.
    #[test]
    fn inlines_defs_ref() {
        let input = json!({
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": { "$ref": "#/definitions/TaskDraft" }
                }
            },
            "definitions": {
                "TaskDraft": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" }
                    }
                }
            }
        });
        let out = sanitize_strict(input);
        assert!(out.get("definitions").is_none());
        let items = &out["properties"]["tasks"]["items"];
        assert_eq!(items["type"], "object");
        assert!(items.get("$ref").is_none());
    }

    /// `sanitize_strict` rewrites nullable array types to their non-null form.
    #[test]
    fn normalises_nullable_type() {
        let input = json!({
            "type": "object",
            "properties": {
                "note": { "type": ["string", "null"] }
            }
        });
        let out = sanitize_strict(input);
        assert_eq!(out["properties"]["note"]["type"], "string");
    }

    /// A struct field literally named `title` must survive sanitisation as a property key.
    #[test]
    fn preserves_property_named_title() {
        let input = json!({
            "type": "object",
            "title": "TaskDraft",
            "required": ["id", "title"],
            "properties": {
                "id":    { "type": "string" },
                "title": { "title": "Title", "type": "string" }
            }
        });
        let out = sanitize_strict(input);
        assert!(out.get("title").is_none());
        assert!(
            out["properties"].get("title").is_some(),
            "title property was incorrectly removed"
        );
        assert!(out["properties"]["title"].get("title").is_none());
    }

    /// `sanitize_strict` removes `additionalProperties` at every nesting level.
    #[test]
    fn removes_additional_properties() {
        let input = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {}
        });
        let out = sanitize_strict(input);
        assert!(out.get("additionalProperties").is_none());
    }
}
