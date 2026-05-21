use serde_json::{Map, Value};

/// Reduces a JSON Schema to the stricter subset accepted by providers.
/// Strips `$schema`, `title`, `$defs`, `definitions`, `additionalProperties`,
/// inlines `$ref` references, and normalises nullable types.
pub(crate) fn sanitize_strict(schema: Value) -> Value {
    let defs = collect_defs(&schema);
    let without_refs = inline_refs(schema, &defs);
    clean_fields(without_refs)
}

/// Collects root-level entries from `definitions` and `$defs`.
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

/// Inlines root-local `$ref` values recursively.
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

/// Extracts the definition name from a `$ref` path.
fn ref_name(r: &str) -> Option<&str> {
    r.split('/').last()
}

/// Removes unsupported keywords and normalizes nullable types.
/// Property names inside `properties` are preserved even when they match schema keywords.
/// Gemini also requires object schemas to carry an explicit `properties` map.
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

/// Cleans nested property schemas without filtering the property names.
fn clean_properties(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            Value::Object(map.into_iter().map(|(k, v)| (k, clean_fields(v))).collect())
        }
        other => other,
    }
}

/// Rewrites `["T", "null"]` to `"T"`.
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

    /// Root and nested schema annotations are removed.
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

    /// Root-local `$ref` values are inlined.
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

    /// Nullable type arrays are normalized.
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

    /// Property keys named `title` must survive sanitization.
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

    /// `additionalProperties` is removed at every level.
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
