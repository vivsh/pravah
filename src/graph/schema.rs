use serde_json::Value as JsonValue;

use super::error::GraphError;
use super::model::TypeSpec;
use super::value::Value;

/// Performs the backend's cheap runtime shape check for a value.
///
/// This is not full JSON Schema validation; it only catches obvious shape
/// mismatches before edge or variable state is mutated.
pub fn validate_value(type_spec: &TypeSpec, value: &Value, label: &str) -> Result<(), GraphError> {
    if runtime_shape_accepts(&type_spec.schema, value) {
        Ok(())
    } else {
        Err(GraphError::Schema {
            label: label.into(),
            expected: type_spec.name.clone(),
            value: compact_value(value),
        })
    }
}

/// Checks a resume value against the suspend node's declared type name.
pub fn validate_resume_type(type_name: &str, value: &Value) -> Result<(), GraphError> {
    let ok = match type_name {
        "Value" | "serde_json::Value" | "Any" => true,
        "String" | "str" => value.is_string(),
        "Bool" | "Boolean" | "bool" => value.is_bool(),
        "Number" | "f64" | "f32" => value.is_number(),
        "Integer" | "i64" | "i32" | "i16" | "i8" | "u64" | "u32" | "u16" | "u8" | "usize"
        | "isize" => value.as_i64().is_some() || value.as_u64().is_some(),
        "Array" | "Vec" => value.is_array(),
        "Object" | "Map" => value.is_object(),
        _ => true,
    };
    if ok {
        Ok(())
    } else {
        Err(GraphError::Schema {
            label: "resume value".into(),
            expected: type_name.into(),
            value: compact_value(value),
        })
    }
}

/// Minimal VM safety check for obvious shape mismatches.
///
/// `TypeSpec.schema` is backend metadata. This function intentionally does not
/// implement JSON Schema: references, definitions, formats, and complex
/// constraints are left to the graph loading layer. The runtime only rejects
/// directly expressed primitive/array/object mismatches that would otherwise
/// corrupt edge or variable state.
fn runtime_shape_accepts(schema: &JsonValue, value: &Value) -> bool {
    if schema.get("$ref").is_some()
        || schema.get("$defs").is_some()
        || schema.get("definitions").is_some()
    {
        return true;
    }
    if let Some(any_of) = schema.get("anyOf").and_then(JsonValue::as_array) {
        return any_of
            .iter()
            .any(|candidate| runtime_shape_accepts(candidate, value));
    }
    if let Some(one_of) = schema.get("oneOf").and_then(JsonValue::as_array) {
        return one_of
            .iter()
            .any(|candidate| runtime_shape_accepts(candidate, value));
    }
    if let Some(all_of) = schema.get("allOf").and_then(JsonValue::as_array) {
        return all_of
            .iter()
            .all(|candidate| runtime_shape_accepts(candidate, value));
    }
    let Some(schema_type) = schema.get("type") else {
        return true;
    };
    match schema_type {
        JsonValue::String(name) => type_accepts(name, schema, value),
        JsonValue::Array(names) => names
            .iter()
            .filter_map(JsonValue::as_str)
            .any(|name| type_accepts(name, schema, value)),
        _ => true,
    }
}

fn type_accepts(name: &str, schema: &JsonValue, value: &Value) -> bool {
    match name {
        "null" => value.is_null(),
        "boolean" => value.is_bool(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "array" => validate_array(schema, value),
        "object" => validate_object(schema, value),
        _ => true,
    }
}

fn validate_array(schema: &JsonValue, value: &Value) -> bool {
    let Some(items) = value.as_array() else {
        return false;
    };
    let Some(item_schema) = schema.get("items") else {
        return true;
    };
    items
        .iter()
        .all(|item| runtime_shape_accepts(item_schema, item))
}

fn validate_object(schema: &JsonValue, value: &Value) -> bool {
    if !value.is_object() {
        return false;
    }
    if let Some(required) = schema.get("required").and_then(JsonValue::as_array) {
        for key in required.iter().filter_map(JsonValue::as_str) {
            if value.get(key).is_none() {
                return false;
            }
        }
    }
    if let Some(properties) = schema.get("properties").and_then(JsonValue::as_object) {
        for (key, property_schema) in properties {
            if let Some(property_value) = value.get(key)
                && !runtime_shape_accepts(property_schema, property_value)
            {
                return false;
            }
        }
    }
    true
}

fn compact_value(value: &Value) -> String {
    const LIMIT: usize = 256;
    let raw = value.to_string();
    let mut preview = raw.chars().take(LIMIT).collect::<String>();
    if raw.chars().count() > LIMIT {
        preview.push_str("...");
    }
    preview
}
