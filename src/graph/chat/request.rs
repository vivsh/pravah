use crate::clients::{Message, Role};
use crate::graph::agent::validate_tool_names;
use crate::graph::{GraphError, McpResourceRef, Value};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

mod schema;

/// One Chat submission with invocation-local context and tool/resource selection.
/// Construction and serialization do not validate policy; `send` rejects invalid requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub(super) message: Message,
    pub(super) memory: Option<String>,
    pub(super) tools: Option<Vec<String>>,
    pub(super) resources: Option<Vec<McpResourceRef>>,
}

impl ChatRequest {
    /// Collects a message; only user-role messages are accepted when sent.
    pub fn new(message: Message) -> Self {
        Self {
            message,
            memory: None,
            tools: None,
            resources: None,
        }
    }

    /// Supplies text memory for this invocation only, outside conversation history.
    pub fn memory(mut self, memory: impl Into<String>) -> Self {
        self.memory = Some(memory.into());
        self
    }

    /// Selects declared tools for this invocation; an empty selection disables domain tools.
    pub fn tools(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// Replaces builder-default MCP references for this invocation, including with an empty list.
    pub fn resources(mut self, resources: impl IntoIterator<Item = McpResourceRef>) -> Self {
        self.resources = Some(resources.into_iter().collect());
        self
    }

    /// Borrows the submitted message, including its optional application key.
    pub fn message(&self) -> &Message {
        &self.message
    }
    /// Borrows this invocation's memory, if supplied.
    pub fn memory_text(&self) -> Option<&str> {
        self.memory.as_deref()
    }
    /// Borrows the explicit selection; `None` selects all declared candidates.
    pub fn selected_tools(&self) -> Option<&[String]> {
        self.tools.as_deref()
    }
    /// Borrows resource overrides; `None` uses builder defaults.
    pub fn selected_resources(&self) -> Option<&[McpResourceRef]> {
        self.resources.as_deref()
    }

    /// Checks semantic constraints without resolving resources or mutating the execution.
    pub(super) fn validate(&self, payload: &Value) -> Result<(), GraphError> {
        let check = || -> Result<(), String> {
            if !matches!(self.message.role, Role::User) {
                return Err("message must have the user role".into());
            }
            if let Some(names) = &self.tools {
                let tools = payload
                    .get("tools")
                    .and_then(Value::as_array)
                    .ok_or("missing candidate tool metadata")?;
                validate_tool_names(
                    names,
                    tools
                        .iter()
                        .filter_map(|tool| tool.get("name").and_then(Value::as_str)),
                )?;
            }
            if let Some(refs) = &self.resources {
                validate_resources(refs)?;
            }
            Ok(())
        };
        check().map_err(|reason| GraphError::ChatRequestValidation { reason })
    }
}

impl From<Message> for ChatRequest {
    fn from(value: Message) -> Self {
        Self::new(value)
    }
}
impl From<String> for ChatRequest {
    fn from(value: String) -> Self {
        Self::new(Message::user(value))
    }
}
impl From<&str> for ChatRequest {
    fn from(value: &str) -> Self {
        Self::new(Message::user(value))
    }
}

impl JsonSchema for ChatRequest {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ChatRequest".into()
    }
    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        schema::request_schema(generator)
    }
}

/// Validates references and the simple named-template vocabulary supported by Pravah MCP.
pub(super) fn validate_resources(refs: &[McpResourceRef]) -> Result<(), String> {
    for (index, resource) in refs.iter().enumerate() {
        if refs[..index].contains(resource) {
            return Err("duplicate MCP resource reference".into());
        }
        if resource.server().trim().is_empty() || resource.uri().trim().is_empty() {
            return Err("MCP server and URI must not be empty".into());
        }
        let mut uri = resource.uri().to_owned();
        for name in resource.arguments().keys() {
            let pattern = format!("{{{name}}}");
            if name.is_empty() || !uri.contains(&pattern) {
                return Err("unused or empty MCP template argument".into());
            }
            uri = uri.replace(&pattern, "argument");
        }
        if uri.contains(['{', '}']) || uri.chars().any(char::is_whitespace) || !uri.contains(':') {
            return Err("invalid or unresolved MCP resource URI".into());
        }
    }
    Ok(())
}
