use serde_json::{Map, Value};

/// Preserves Pravah's canonical authored tool-schema representation.
/// Strips `$schema`, `title`, `$defs`, `definitions`, `additionalProperties`,
/// inlines `$ref` references, and normalises nullable types.
pub(super) fn sanitize_strict(schema: Value) -> Result<Value, String> {
    let defs = collect_defs(&schema);
    let without_refs = inline_refs(schema, &defs, &mut Vec::new())?;
    Ok(clean_fields(without_refs))
}

/// Collects root-level entries from `definitions` and `$defs`.
fn collect_defs(schema: &Value) -> Map<String, Value> {
    let mut defs = Map::new();
    if let Some(obj) = schema.as_object() {
        for key in &["definitions", "$defs"] {
            if let Some(Value::Object(map)) = obj.get(*key) {
                defs.extend(map.iter().map(|(name, value)| {
                    (
                        format!("#/{key}/{}", name.replace('~', "~0").replace('/', "~1")),
                        value.clone(),
                    )
                }));
            }
        }
    }
    defs
}

/// Inlines root-local `$ref` values recursively.
fn inline_refs(
    value: Value,
    defs: &Map<String, Value>,
    stack: &mut Vec<String>,
) -> Result<Value, String> {
    match value {
        Value::Object(mut map) => {
            if let Some(reference) = map.remove("$ref") {
                return resolve_reference(reference, map, defs, stack);
            }
            map.into_iter()
                .map(|(k, v)| {
                    let value = if matches!(k.as_str(), "properties" | "$defs" | "definitions") {
                        inline_named(v, defs, stack)?
                    } else {
                        inline_refs(v, defs, stack)?
                    };
                    Ok((k, value))
                })
                .collect::<Result<Map<_, _>, _>>()
                .map(Value::Object)
        }
        Value::Array(values) => values
            .into_iter()
            .map(|v| inline_refs(v, defs, stack))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        other => Ok(other),
    }
}
fn invalid_ref(message: &str) -> String {
    message.to_owned()
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
        let mut non_null = types.iter().filter(|v| v.as_str() != Some("null"));
        if let Some(value) = non_null.next()
            && non_null.next().is_none()
        {
            return value.clone();
        }
    }
    value
}

/// Schema-map keys are names, so a property literally named `$ref` is not a reference.
fn inline_named(
    value: Value,
    defs: &Map<String, Value>,
    stack: &mut Vec<String>,
) -> Result<Value, String> {
    match value {
        Value::Object(map) => map
            .into_iter()
            .map(|(name, value)| Ok((name, inline_refs(value, defs, stack)?)))
            .collect::<Result<Map<_, _>, _>>()
            .map(Value::Object),
        other => Ok(other),
    }
}

/// Resolves only supported local references and detects cycles before recursive expansion.
fn resolve_reference(
    reference: Value,
    mut map: Map<String, Value>,
    defs: &Map<String, Value>,
    stack: &mut Vec<String>,
) -> Result<Value, String> {
    let reference = reference
        .as_str()
        .ok_or_else(|| invalid_ref("non-string $ref"))?;
    if !(reference.starts_with("#/$defs/") || reference.starts_with("#/definitions/")) {
        return Err(invalid_ref("unsupported non-local $ref"));
    }
    if stack.iter().any(|entry| entry == reference) {
        return Err(invalid_ref("recursive schema references are unsupported"));
    }
    for key in ["$defs", "definitions", "$schema", "title", "description"] {
        map.remove(key);
    }
    if !map.is_empty() {
        return Err(invalid_ref("$ref siblings cannot be safely normalized"));
    }
    let target = defs
        .get(reference)
        .ok_or_else(|| invalid_ref("unresolved schema $ref"))?;
    stack.push(reference.into());
    let result = inline_refs(target.clone(), defs, stack);
    stack.pop();
    result
}

#[cfg(test)]
#[path = "tests/normalize.rs"]
mod tests;
