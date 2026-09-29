use crate::graph::agent::validate_tool_names;
use crate::graph::{GraphError, McpResourceRef, Value, ValueError, from_value};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// One Chat submission with invocation-local context and tool/resource selection.
/// Construction and serialization do not validate policy; `send` rejects invalid requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest<I> {
    pub(crate) input: I,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) memory: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) resources: Option<Vec<McpResourceRef>>,
}

impl<I> ChatRequest<I> {
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

    /// Replaces configured MCP references for this invocation, including with an empty list.
    pub fn resources(mut self, resources: impl IntoIterator<Item = McpResourceRef>) -> Self {
        self.resources = Some(resources.into_iter().collect());
        self
    }

    /// Borrows the original typed input without rendering or converting it.
    pub fn input(&self) -> &I {
        &self.input
    }
    /// Borrows this invocation's memory, if supplied.
    pub fn memory_text(&self) -> Option<&str> {
        self.memory.as_deref()
    }
    /// Borrows the explicit selection; `None` preserves the configured tool filter.
    pub fn selected_tools(&self) -> Option<&[String]> {
        self.tools.as_deref()
    }
    /// Borrows resource overrides; `None` preserves configured resources.
    pub fn selected_resources(&self) -> Option<&[McpResourceRef]> {
        self.resources.as_deref()
    }

    /// Checks semantic constraints without resolving resources or mutating the execution.
    pub(super) fn validate(&self, payload: &Value) -> Result<(), GraphError> {
        let check = || -> Result<(), String> {
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

impl<I> From<I> for ChatRequest<I> {
    fn from(input: I) -> Self {
        Self {
            input,
            key: None,
            memory: None,
            tools: None,
            resources: None,
        }
    }
}
impl From<&str> for ChatRequest<String> {
    fn from(value: &str) -> Self {
        Self::from(value.to_owned())
    }
}

impl ChatRequest<Value> {
    /// Decodes options without recursively copying the shared domain input.
    pub(crate) fn decode(value: &Value) -> Result<Self, ValueError> {
        let invalid = || ValueError::Unsupported("invalid chat request envelope".into());
        let mut fields = value.object_entries().ok_or_else(invalid)?;
        if fields.any(|(key, _)| !matches!(key, "input" | "key" | "memory" | "tools" | "resources"))
        {
            return Err(invalid());
        }
        Ok(Self {
            input: value.get("input").ok_or_else(invalid)?.clone(),
            key: from_value(value.get("key").cloned().unwrap_or(Value::NULL))?,
            memory: from_value(value.get("memory").cloned().unwrap_or(Value::NULL))?,
            tools: from_value(value.get("tools").cloned().unwrap_or(Value::NULL))?,
            resources: from_value(value.get("resources").cloned().unwrap_or(Value::NULL))?,
        })
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
