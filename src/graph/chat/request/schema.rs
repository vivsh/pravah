use super::*;
use serde_json::json;

/// Describes the actual Rath user-message wire shape without weakening it to arbitrary JSON.
pub(super) fn request_schema(generator: &mut SchemaGenerator) -> Schema {
    let resource = generator.subschema_for::<McpResourceRef>();
    schemars::json_schema!({
        "type": "object", "additionalProperties": false,
        "required": ["message"],
        "properties": {
            "message": message_schema(),
            "memory": {"type": ["string", "null"]},
            "tools": {"type": ["array", "null"], "items": {"type":"string"}, "uniqueItems":true},
            "resources": {"type": ["array", "null"], "items": resource, "uniqueItems":true}
        }
    })
}

/// Supports all three attachment forms and both optional usage counters from Rath.
fn message_schema() -> serde_json::Value {
    let attachments: Vec<_> = [("inline", "data"), ("file", "path"), ("url", "url")]
        .into_iter().map(|(kind, field)| json!({
            "type":"object", "required":["type", "mime_type", field],
            "properties": {"type":{"const":kind}, "mime_type":{"type":"string"}, (field):{"type":"string"}}
        })).collect();
    let counter = json!({"type":["integer","null"], "minimum":0, "maximum":u32::MAX});
    json!({
        "type":"object", "required":["role","content"],
        "properties": {
            "role":{"type":"object", "required":["role"], "properties":{"role":{"const":"user"}}},
            "content":{"type":"string"}, "key":{"type":["string","null"]},
            "attachments":{"type":"array", "items":{"oneOf":attachments}},
            "usage":{"type":["object","null"], "properties":{"input":counter,"output":counter}}
        }
    })
}
