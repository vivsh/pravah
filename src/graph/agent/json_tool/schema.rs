//! Admission of a local-reference Draft 2020-12 profile without schema rewriting.

use serde_json::Value;

const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";

/// Admits every schema location before compiling assertions, formats and local references.
pub(super) fn compile(
    schema: &Value,
    tool: &str,
    label: &str,
) -> Result<jsonschema::Validator, String> {
    let mut locations = Vec::new();
    admit(schema, "", &mut locations)
        .map_err(|reason| format!("tool '{tool}' {label} schema: {reason}"))?;
    check_references(schema, "", &locations, &mut Vec::new())
        .map_err(|reason| format!("tool '{tool}' {label} schema: {reason}"))?;
    // Admission forbids every retrieval source, including dialect overrides and rebasing.
    // The existing dependency also disables HTTP/file resolution through default-features=false.
    jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .should_validate_formats(true)
        .should_ignore_unknown_formats(false)
        .build(schema)
        .map_err(|error| {
            format!(
                "tool '{tool}' {label} schema at {}: {error}",
                error.instance_path(),
            )
        })
}

/// Walks only schema locations; enum, const, examples and defaults remain literal JSON.
fn admit(schema: &Value, path: &str, locations: &mut Vec<String>) -> Result<(), String> {
    locations.push(path.into());
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .ok_or_else(|| format!("{path}: expected object or boolean schema"))?;
    for (keyword, value) in object {
        let location = pointer(path, keyword);
        if !known_keyword(keyword) {
            return Err(format!("{location}: unsupported keyword '{keyword}'"));
        }
        match keyword.as_str() {
            "$schema" if !path.is_empty() || value.as_str() != Some(DIALECT) => {
                return Err(format!(
                    "{location}: only root Draft 2020-12 dialect is supported"
                ));
            }
            "$ref" if !value.as_str().is_some_and(|r| r.starts_with("#/")) => {
                return Err(format!(
                    "{location}: only same-document JSON Pointer references are supported"
                ));
            }
            "format" if !value.as_str().is_some_and(known_format) => {
                return Err(format!("{location}: unsupported format"));
            }
            _ => {}
        }
    }
    for (child, location) in children(schema, path)? {
        admit(child, &location, locations)?;
    }
    Ok(())
}

/// Enumerates keyword-owned schemas without interpreting instance data as schema structure.
fn children<'a>(schema: &'a Value, path: &str) -> Result<Vec<(&'a Value, String)>, String> {
    let Some(object) = schema.as_object() else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    for (keyword, value) in object {
        let location = pointer(path, keyword);
        match keyword.as_str() {
            "$defs" | "properties" | "patternProperties" | "dependentSchemas" => {
                let map = value
                    .as_object()
                    .ok_or_else(|| format!("{location}: expected schema map"))?;
                result.extend(
                    map.iter()
                        .map(|(key, schema)| (schema, pointer(&location, key))),
                );
            }
            "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                let array = value
                    .as_array()
                    .ok_or_else(|| format!("{location}: expected schema array"))?;
                result.extend(
                    array
                        .iter()
                        .enumerate()
                        .map(|(i, s)| (s, pointer(&location, &i.to_string()))),
                );
            }
            "additionalProperties"
            | "unevaluatedProperties"
            | "propertyNames"
            | "items"
            | "unevaluatedItems"
            | "contains"
            | "not"
            | "if"
            | "then"
            | "else" => result.push((value, location)),
            _ => {}
        }
    }
    Ok(result)
}

/// Rejects rebased, unresolved, non-schema and cyclic references before the validator sees them.
fn check_references(
    root: &Value,
    path: &str,
    locations: &[String],
    active: &mut Vec<String>,
) -> Result<(), String> {
    if active.iter().any(|entry| entry == path) {
        return Err(format!(
            "{path}: recursive schema references are unsupported"
        ));
    }
    let schema = root
        .pointer(path)
        .ok_or_else(|| format!("{path}: unresolved schema location"))?;
    active.push(path.into());
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let target = &reference[1..]; // admission established a leading #/
        if !locations.iter().any(|location| location == target) {
            return Err(format!(
                "{path}/$ref: unresolved reference or target is not a schema"
            ));
        }
        check_references(root, target, locations, active)?;
    }
    for (_, location) in children(schema, path)? {
        check_references(root, &location, locations, active)?;
    }
    active.pop();
    Ok(())
}

fn pointer(parent: &str, key: &str) -> String {
    format!("{parent}/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn known_keyword(keyword: &str) -> bool {
    assertion_keyword(keyword) || annotation_keyword(keyword)
}

/// Recognizes core, applicator and validation keywords in the supported profile.
fn assertion_keyword(keyword: &str) -> bool {
    matches!(
        keyword,
        "$schema"
            | "$ref"
            | "$defs"
            | "$comment"
            | "type"
            | "enum"
            | "const"
            | "multipleOf"
            | "maximum"
            | "exclusiveMaximum"
            | "minimum"
            | "exclusiveMinimum"
            | "maxLength"
            | "minLength"
            | "pattern"
            | "format"
            | "maxItems"
            | "minItems"
            | "uniqueItems"
            | "maxContains"
            | "minContains"
            | "maxProperties"
            | "minProperties"
            | "required"
            | "dependentRequired"
            | "properties"
            | "patternProperties"
            | "additionalProperties"
            | "unevaluatedProperties"
            | "propertyNames"
            | "items"
            | "prefixItems"
            | "unevaluatedItems"
            | "contains"
            | "dependentSchemas"
            | "allOf"
            | "anyOf"
            | "oneOf"
            | "not"
            | "if"
            | "then"
            | "else"
    )
}

fn annotation_keyword(keyword: &str) -> bool {
    matches!(
        keyword,
        "title" | "description" | "default" | "examples" | "deprecated" | "readOnly" | "writeOnly"
    )
}

/// Admits only built-in formats whose assertions the pinned validator can enforce.
fn known_format(format: &str) -> bool {
    matches!(
        format,
        "date"
            | "date-time"
            | "duration"
            | "email"
            | "hostname"
            | "idn-email"
            | "idn-hostname"
            | "ipv4"
            | "ipv6"
            | "iri"
            | "iri-reference"
            | "json-pointer"
            | "regex"
            | "relative-json-pointer"
            | "time"
            | "uri"
            | "uri-reference"
            | "uri-template"
            | "uuid"
    )
}

#[cfg(test)]
#[path = "../tests/json_schema.rs"]
mod tests;
