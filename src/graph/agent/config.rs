use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::clients::Message;

use super::AgentToolPayload;

/// Read-only metadata presented to a runtime tool filter.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    name: String,
    description: String,
    parameters: JsonValue,
}

impl ToolInfo {
    pub(crate) fn from_payload(payload: &AgentToolPayload) -> Self {
        Self {
            name: payload.name.clone(),
            description: payload.description.clone(),
            parameters: payload.parameters.clone(),
        }
    }

    /// Returns the model-facing tool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the model-facing tool description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the tool input JSON Schema.
    pub fn parameters(&self) -> &JsonValue {
        &self.parameters
    }
}

/// Runtime predicate selecting from an agent's statically prepared tools.
#[derive(Clone)]
pub struct ToolFilter {
    predicate: Arc<dyn Fn(&ToolInfo) -> bool + Send + Sync>,
}

impl fmt::Debug for ToolFilter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ToolFilter").finish_non_exhaustive()
    }
}

impl Default for ToolFilter {
    fn default() -> Self {
        Self::all()
    }
}

impl ToolFilter {
    /// Selects every statically prepared tool.
    pub fn all() -> Self {
        Self::new(|_| true)
    }

    /// Builds a filter evaluated once when an agent invocation starts.
    pub fn new(predicate: impl Fn(&ToolInfo) -> bool + Send + Sync + 'static) -> Self {
        Self {
            predicate: Arc::new(predicate),
        }
    }

    pub(crate) fn allows(&self, tool: &AgentToolPayload) -> bool {
        (self.predicate)(&ToolInfo::from_payload(tool))
    }
}

/// Reference to one text resource exposed by a configured MCP server.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct McpResourceRef {
    server: String,
    uri: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    arguments: BTreeMap<String, String>,
}

impl McpResourceRef {
    /// Selects a concrete resource URI from a configured server.
    pub fn new(server: impl Into<String>, uri: impl Into<String>) -> Self {
        Self {
            server: server.into(),
            uri: uri.into(),
            arguments: BTreeMap::new(),
        }
    }

    /// Selects a resource-template URI with deterministic string arguments.
    pub fn template(
        server: impl Into<String>,
        uri: impl Into<String>,
        arguments: BTreeMap<String, String>,
    ) -> Self {
        Self {
            server: server.into(),
            uri: uri.into(),
            arguments,
        }
    }

    /// Returns the configured MCP server identifier.
    pub fn server(&self) -> &str {
        &self.server
    }

    /// Returns the concrete or template resource URI.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Returns resource-template arguments in deterministic key order.
    pub fn arguments(&self) -> &BTreeMap<String, String> {
        &self.arguments
    }
}

/// Dynamic settings resolved once when an agent invocation starts.
#[derive(Clone)]
pub struct AgentConfig {
    pub(crate) model: String,
    pub(crate) instructions: String,
    pub(crate) message: Message,
    pub(crate) memory: Option<String>,
    pub(crate) provider_config: Option<JsonValue>,
    pub(crate) keep_alive: bool,
    pub(crate) turn_budget: Option<u32>,
    pub(crate) turn_budget_message: Option<String>,
    pub(crate) tool_filter: ToolFilter,
    pub(crate) resources: Vec<McpResourceRef>,
}

impl AgentConfig {
    /// Creates the required model, instructions, and initial user message.
    pub fn new(
        model: impl Into<String>,
        instructions: impl Into<String>,
        message: Message,
    ) -> Self {
        Self {
            model: model.into(),
            instructions: instructions.into(),
            message,
            memory: None,
            provider_config: None,
            keep_alive: false,
            turn_budget: None,
            turn_budget_message: None,
            tool_filter: ToolFilter::all(),
            resources: Vec::new(),
        }
    }

    /// Adds invocation-specific text memory outside conversation history.
    pub fn memory(mut self, memory: impl Into<String>) -> Self {
        self.memory = Some(memory.into());
        self
    }

    /// Sets opaque provider-specific configuration passed through to Rath.
    pub fn provider_config(mut self, config: impl Into<JsonValue>) -> Self {
        self.provider_config = Some(config.into());
        self
    }

    /// Keeps one agent session across repeated invocation of this graph node.
    pub fn keep_alive(mut self) -> Self {
        self.keep_alive = true;
        self
    }

    /// Sets the maximum number of model dispatch turns for this invocation.
    pub fn turn_budget(mut self, turns: u32) -> Self {
        self.turn_budget = Some(turns);
        self
    }

    /// Overrides the reminder used on the final configured model turn.
    pub fn turn_budget_message(mut self, message: impl Into<String>) -> Self {
        self.turn_budget_message = Some(message.into());
        self
    }

    /// Selects a runtime subset of the statically prepared toolset.
    pub fn tool_filter(mut self, filter: ToolFilter) -> Self {
        self.tool_filter = filter;
        self
    }

    /// Selects MCP text resources to resolve before the first model turn.
    pub fn resources(mut self, resources: impl IntoIterator<Item = McpResourceRef>) -> Self {
        self.resources = resources.into_iter().collect();
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResolvedAgentConfig {
    pub model: String,
    pub instructions: String,
    pub memory: Option<String>,
    pub provider_config: Option<JsonValue>,
    pub keep_alive: bool,
    pub turn_budget: Option<u32>,
    pub turn_budget_message: Option<String>,
    pub tools: Vec<String>,
    pub resources: Vec<ResolvedResource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResolvedResource {
    pub server: String,
    pub uri: String,
    pub text: String,
}
