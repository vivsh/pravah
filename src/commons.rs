use std::future::Future;
use std::pin::Pin;

use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::context::Context;
use crate::tools::base::ErasedTool;
use crate::tools::{ToolBox, ToolDefinition, ToolError};

/// Configuration returned by [`Agent::build`].
pub struct AgentConfig {
    /// System prompt sent to the LLM on every turn.
    pub preamble: String,
    /// Model URL, e.g. `gemini://gemini-2.5-flash-lite` or `ollama://localhost:11434/qwen3:8b`.
    pub model_url: String,
    /// Tools available to the agent. Do **not** add the exit sentinel — the flow engine
    /// injects it automatically via [`ToolBox::with_agent`].
    pub tool_box: ToolBox,
}

impl AgentConfig {
    /// Convenience constructor for the common no-tools case.
    pub fn new(preamble: impl Into<String>, model_url: impl Into<String>) -> Self {
        Self {
            preamble: preamble.into(),
            model_url: model_url.into(),
            tool_box: ToolBox::builder().build(),
        }
    }

    /// Attach a toolbox.
    pub fn with_tools(mut self, tool_box: ToolBox) -> Self {
        self.tool_box = tool_box;
        self
    }
}

/// Trait implemented by every agent input type.
///
/// The implementing struct is both the LLM input schema (serialized into the first user turn)
/// and the anchor for associated metadata: preamble, model URL, and the tool set.
///
/// Return an [`AgentConfig`] from [`build`](Agent::build). The flow engine appends the exit
/// sentinel automatically — do not add it to the toolbox manually.
pub trait Agent: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Typed value the LLM must produce when it calls the exit sentinel.
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Node identifier used in flow graphs. Defaults to the schemars schema name.
    fn node_id() -> String {
        Self::schema_name()
    }

    /// Returns the configuration for this agent: preamble, model URL, and tools.
    fn build() -> AgentConfig;
}

/// Auto-generated sentinel tool. The only place [`AgentExit`] / [`ToolError::Exit`] is constructed.
struct AgentExitTool {
    name: String,
    def: ToolDefinition,
}

impl ErasedTool for AgentExitTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> ToolDefinition {
        self.def.clone()
    }

    fn call_raw<'a>(
        &'a self,
        _ctx: Context,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'a>> {
        Box::pin(async move { Err(ToolError::Exit(args)) })
    }
}

impl ToolBox {
    /// Appends the typed exit sentinel for agent `A` and returns the updated box.
    ///
    /// When the toolbox is empty the sentinel is **not** injected: the flow engine will
    /// instead use structured-output mode, passing `A::Output`'s schema directly to the
    /// client so the LLM returns a typed JSON object. In that case `ClientOutput::Output`
    /// is already treated as an exit by `handle_agent`.
    ///
    /// `pub(crate)` — only the flow engine calls this.
    pub(crate) fn with_agent<A: Agent>(mut self) -> Self {
        if self.is_empty() {
            // No tools — structured-output mode; sentinel is not needed.
            return self;
        }
        let name = self.exit_name().to_owned();
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let parameters = serde_json::to_value(schema_gen.root_schema_for::<A::Output>())
            .unwrap_or_else(|_| Value::Object(Default::default()));
        let def = ToolDefinition {
            name: name.clone(),
            description: "Submit your final result. Call this only when the task is complete."
                .to_owned(),
            parameters,
        };
        self.push_erased(Box::new(AgentExitTool { name, def }));
        self
    }
}
