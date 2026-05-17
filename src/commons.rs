use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::context::Context;
use crate::flows::flows::{AgentInfo, FlowNode};
use crate::flows::{Flow, FlowGraph};
use crate::tools::base::{ErasedTool, ToolOutcome};
use crate::tools::{ToolBox, ToolDefinition, ToolError};

/// Runtime settings returned by [`Agent::build`].
pub struct AgentConfig {
    /// Prompt sent before each model turn.
    pub preamble: String,
    /// Model URL used to create the client.
    pub model_url: String,
    /// Tools exposed to the agent.
    /// Do not add the exit sentinel yourself; the engine injects it.
    pub tool_box: ToolBox,
    /// Sampling temperature. `None` uses the provider default.
    pub temperature: Option<f32>,
    /// Enables provider-specific reasoning modes when supported.
    pub thinking: bool,
    /// Reasoning budget. Ignored unless `thinking` is enabled.
    pub thinking_budget: Option<u32>,
}

impl AgentConfig {
    /// Builds an agent config with no tools.
    pub fn new(preamble: impl Into<String>, model_url: impl Into<String>) -> Self {
        Self {
            preamble: preamble.into(),
            model_url: model_url.into(),
            tool_box: ToolBox::new(),
            temperature: None,
            thinking: false,
            thinking_budget: None,
        }
    }

    /// Attaches a toolbox.
    pub fn with_tools(mut self, tool_box: ToolBox) -> Self {
        self.tool_box = tool_box;
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Turns reasoning mode on or off.
    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    /// Sets the reasoning budget.
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }
}

/// Implemented by every agent input type.
/// The type defines the first user payload, the model settings, and the tool set.
/// Return an [`AgentConfig`] from [`build`](Agent::build).
/// Do not add the exit sentinel to the toolbox; the engine injects it.
pub trait Agent: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Value the agent must submit to finish.
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Graph node id. Defaults to the schema name.
    fn node_id() -> String {
        Self::schema_name()
    }

    /// Returns the runtime settings for this agent.
    fn build() -> AgentConfig;
}

/// Exit tool injected for agents.
/// It converts a final tool call into [`ToolOutcome::Exit`] and never reaches user code.
struct AgentExitTool {
    name: String,
    output_type: String,
    def: ToolDefinition,
}

impl ErasedTool for AgentExitTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> ToolDefinition {
        self.def.clone()
    }

    fn input_type(&self) -> String {
        self.output_type.clone()
    }

    fn output_type(&self) -> String {
        self.output_type.clone()
    }

    fn call_raw<'a>(
        &'a self,
        _ctx: Context,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>> {
        Box::pin(async move { Ok(ToolOutcome::Exit(args)) })
    }
}

/// Erased adapter for an agent registered with [`ToolBox::agent`].
/// `call_raw` must never run; the engine intercepts agent tool calls earlier.
struct AgentToolDispatcher<A: Agent> {
    name: String,
    _phantom: PhantomData<fn() -> A>,
}

impl<A: Agent + 'static> ErasedTool for AgentToolDispatcher<A> {
    fn name(&self) -> &str {
        &self.name
    }

    fn needs_tool_node(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        let parameters = serde_json::to_value(schemars::schema_for!(A))
            .unwrap_or_else(|_| Value::Object(Default::default()));
        ToolDefinition {
            name: self.name.clone(),
            description: format!("Invoke the {} agent.", self.name),
            parameters,
        }
    }

    fn input_type(&self) -> String {
        A::node_id()
    }

    fn output_type(&self) -> String {
        A::Output::schema_name().into()
    }

    fn call_raw<'a>(
        &'a self,
        _ctx: Context,
        _args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            unreachable!("AgentToolDispatcher::call_raw should never be called — frame-push intercepts")
        })
    }
}

/// Erased adapter for a flow registered with [`ToolBox::flow`].
struct FlowToolDispatcher<F: Flow> {
    name: String,
    _phantom: PhantomData<fn() -> F>,
}

impl<F: Flow + 'static> ErasedTool for FlowToolDispatcher<F> {
    fn name(&self) -> &str {
        &self.name
    }

    fn needs_tool_node(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        let parameters = serde_json::to_value(schemars::schema_for!(F))
            .unwrap_or_else(|_| Value::Object(Default::default()));
        ToolDefinition {
            name: self.name.clone(),
            description: format!("Invoke the {} flow.", self.name),
            parameters,
        }
    }

    fn input_type(&self) -> String {
        F::schema_name().into()
    }

    fn output_type(&self) -> String {
        F::Output::schema_name().into()
    }

    fn call_raw<'a>(
        &'a self,
        _ctx: Context,
        _args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            unreachable!("FlowToolDispatcher::call_raw should never be called — frame-push intercepts")
        })
    }
}

impl ToolBox {
    /// Registers an agent as a callable tool.
    /// When the parent agent calls it, the engine pushes a frame and wires the result
    /// back into the parent agent's exit path.
    pub fn agent<A: Agent>(mut self) -> Self {
        self.push_erased(Box::new(AgentToolDispatcher::<A> {
            name: A::node_id(),
            _phantom: PhantomData,
        }));
        self.graph_injectors.push(Box::new(|agent_name: &str, graph: &mut FlowGraph| {
            let name_str = A::node_id();
            let output_str = A::Output::schema_name();
            let entry = graph.interner.intern(&format!("{}::{}", agent_name, name_str));
            let exit  = graph.interner.intern(&output_str);
            let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
            let output_schema = serde_json::to_value(schema_gen.root_schema_for::<A::Output>())
                .unwrap_or_else(|_| Value::Object(Default::default()));
            let config = A::build();
            let tool_box = Arc::new(config.tool_box.with_agent::<A>(graph));
            let mut tool_lookup = std::collections::HashMap::new();
            for i in 0..tool_box.len() {
                let tname = tool_box.name_at(i).to_owned();
                let t_entry = graph.interner.intern(&format!("{}::{}", graph.interner.name_of(entry), tool_box.input_type_at(i)));
                let t_exit  = graph.interner.intern(&tool_box.output_type_at(i));
                tool_lookup.insert(tname, (t_entry, t_exit));
            }
            let info = Arc::new(AgentInfo {
                id: entry,
                tool_box,
                preamble: config.preamble,
                model: config.model_url,
                exit,
                output_schema,
                tool_lookup,
                temperature: config.temperature,
                thinking: config.thinking,
                thinking_budget: config.thinking_budget,
            });
            graph.nodes.insert(entry, FlowNode::AgentTool(info));
        }));
        self
    }

    /// Registers a flow as a callable tool.
    /// When the parent agent calls it, the engine pushes a frame and routes the
    /// flow output back into the parent agent.
    pub fn flow<F: Flow>(mut self) -> Self {
        self.push_erased(Box::new(FlowToolDispatcher::<F> {
            name: F::schema_name().into(),
            _phantom: PhantomData,
        }));
        self.graph_injectors.push(Box::new(|agent_name: &str, graph: &mut FlowGraph| {
            let input_str = F::schema_name();
            let output_str = F::Output::schema_name();
            let entry = graph.interner.intern(&format!("{}::{}", agent_name, input_str));
            let exit  = graph.interner.intern(&output_str);
            let inner = match FlowGraph::from_flow::<F>() {
                Ok(mut g) => {
                    g.parent_entry = Some(entry);
                    g.parent_exit = Some(exit);
                    Arc::new(g)
                }
                Err(e) => {
                    tracing::error!(flow = %input_str, error = %e, "flow tool registration failed");
                    return;
                }
            };
            graph.nodes.insert(entry, FlowNode::FlowTool { name: entry, inner });
        }));
        self
    }

    /// Injects graph-backed tools and the exit sentinel for agent `A`.
    /// If the toolbox has no regular tools, the sentinel is skipped and the agent
    /// stays in structured-output mode.
    pub(crate) fn with_agent<A: Agent>(mut self, graph: &mut FlowGraph) -> Self {
        let agent_name = A::node_id();
        for inject in self.graph_injectors.drain(..) {
            inject(&agent_name, graph);
        }
        if self.is_empty() {
            return self;
        }
        let name = self.exit_name().to_owned();
        let output_type = A::Output::schema_name();
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let parameters = serde_json::to_value(schema_gen.root_schema_for::<A::Output>())
            .unwrap_or_else(|_| Value::Object(Default::default()));
        let def = ToolDefinition {
            name: name.clone(),
            description: "Submit your final result. Call this only when the task is complete."
                .to_owned(),
            parameters,
        };
        self.push_erased(Box::new(AgentExitTool { name, def, output_type }));
        self
    }
}
