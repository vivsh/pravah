use std::collections::BTreeSet;
use std::future::Future;
use std::sync::Arc;

use futures::future::{BoxFuture, FutureExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::clients::{
    ClientOptions, ClientOutput, Message, Role, ToolCall, ToolChoice, ToolDefinition,
    materialize_messages,
};
use crate::context::Context;
use crate::tools::ToolError;

use super::error::GraphError;
use super::model::UntypedGraph;
use super::registry::{
    ContinuationChildCall, ContinuationContext, ContinuationEvent, ContinuationHandler,
    ContinuationTransition,
};
use super::value::{Value, from_value, to_value};
use super::{CompiledFlow, HandlerRegistry};

mod budget;
#[cfg(test)]
mod budget_tests;
mod checkpoint;
mod config;
mod control;
mod definition;
mod execution;
mod intervention;
pub(crate) mod support;
mod tool_execution;

use budget::*;
use checkpoint::*;
pub(crate) use config::ResolvedResource;
pub use config::{AgentConfig, McpResourceRef, ToolFilter, ToolInfo};
use config::{ResolvedAgentConfig, agent_tool_definition};
pub use control::{
    AgentDecision, AgentDirective, AgentInterventionPoint, AgentLoop, AgentLoopMetrics,
    AgentResume, AgentSuspension, AgentToolProposal, AgentToolResult,
};
use control::{AgentDecisionKind, AgentLoopData, ControlStateUpdate};
pub use definition::Agent;
use definition::{AgentConfigurator, AgentController};
use support::*;

const PAYLOAD_VERSION: u32 = 3;
const CHECKPOINT_VERSION: u32 = 4;

/// Validates the identity duplicated in an agent's generic continuation payload.
pub(crate) fn validate_payload_handler(
    payload: &Value,
    handler_key: &str,
) -> Result<(), GraphError> {
    if payload.get("agent_id").is_none() || payload.get("output_schema").is_none() {
        return Ok(());
    }
    let payload = decode_payload(payload).map_err(|err| {
        GraphError::GraphValidation(format!("invalid agent continuation payload: {err}"))
    })?;
    if payload.configure_handler_key != handler_key {
        return Err(GraphError::GraphValidation(format!(
            "agent configure handler '{}' does not match continuation handler '{handler_key}'",
            payload.configure_handler_key
        )));
    }
    let expected_control = format!("{handler_key}::control");
    if payload
        .control_handler_key
        .as_deref()
        .is_some_and(|key| key != expected_control)
    {
        return Err(GraphError::GraphValidation(format!(
            "agent control handler does not match continuation handler '{handler_key}'"
        )));
    }
    Ok(())
}

/// Rewrites identities duplicated inside a typed agent continuation payload.
pub(crate) fn namespace_payload_handler(payload: &mut Value, handler_key: &str) {
    let Ok(mut agent): Result<AgentPayload, _> = from_value(payload.clone()) else {
        return;
    };
    if agent.agent_id.is_empty() || agent.output_schema.is_null() {
        return;
    }
    agent.agent_id = handler_key.to_owned();
    agent.configure_handler_key = handler_key.to_owned();
    if agent.control_handler_key.is_some() {
        agent.control_handler_key = Some(format!("{handler_key}::control"));
    }
    if let Ok(value) = to_value(agent) {
        *payload = value;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Serializable payload stored on an agent continuation node.
///
/// It contains graph-safe metadata only; runtime services and codecs stay in
/// the handler registry.
pub(crate) struct AgentPayload {
    /// Payload format version.
    pub version: u32,
    /// Stable internal agent identity used in history and diagnostics.
    pub agent_id: String,
    /// Stable registry key for this agent's activation function.
    pub configure_handler_key: String,
    /// Stable identity of the optional intervention controller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_handler_key: Option<String>,
    /// JSON schema metadata for agent input.
    pub input_schema: JsonValue,
    /// JSON schema metadata for structured agent output.
    pub output_schema: JsonValue,
    /// Output type name used for structured/exit-tool dispatch.
    pub output_type_name: String,
    /// Tool metadata and child graph indices exposed to the model.
    pub tools: Vec<AgentToolPayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Serializable metadata for one agent tool child graph.
pub(crate) struct AgentToolPayload {
    /// Model-facing tool name.
    pub name: String,
    /// Index of the child graph invoked for this tool.
    pub child_index: usize,
    /// Model-facing tool description.
    pub description: String,
    /// JSON schema metadata for tool parameters.
    pub parameters: JsonValue,
}

#[derive(Clone)]
/// Runtime build artifact for one typed edge-agent tool.
///
/// Tool specs pair serializable graph pieces with runtime-only codecs used by
/// the agent continuation handler.
struct AgentToolSpec {
    /// Serializable tool metadata stored in the agent payload.
    pub payload: AgentToolPayload,
    /// Child graph executed when this tool is called.
    pub graph: UntypedGraph,
    /// Registry entries required by the child graph.
    pub registry: HandlerRegistry,
    runtime: Arc<EdgeAgentToolRuntime>,
}

struct EdgeAgentToolRuntime {
    decode_args: Arc<dyn Fn(JsonValue) -> Result<Value, ToolError> + Send + Sync>,
    render_result:
        Arc<dyn Fn(Value) -> Result<EdgeRenderedToolResult, EdgeToolMessageError> + Send + Sync>,
}

struct EdgeRenderedToolResult {
    message: Message,
    value: Value,
    error: bool,
}

#[derive(Default)]
/// Typed toolset containing every child graph an agent may expose.
pub struct Toolset {
    tools: Vec<AgentToolSpec>,
}

impl Toolset {
    /// Registers a non-capturing asynchronous function as an agent tool.
    ///
    /// The input type defines the stable tool identity and schema. Definition
    /// errors accumulate and surface when the containing graph is compiled.
    pub fn tool<I, O, Fut>(mut self, func: fn(I, Context) -> Fut) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    {
        let definition = match agent_tool_definition::<I>() {
            Ok(definition) => definition,
            Err(err) => {
                self.tools.push(error_tool_spec(err));
                return self;
            }
        };
        let child = build_function_tool_flow::<I, O, Fut>(func);
        self.push_tool_with_message::<I, JsonValue>(
            definition,
            child,
            Arc::new(decode_handler_tool_result::<O>),
        );
        self
    }

    /// Registers a function-defined flow as an agent tool.
    ///
    /// The flow is prepared as a child graph and uses its input type as the
    /// stable model-facing tool identity.
    pub fn flow<I, O>(mut self, flow: fn(super::Flow<I>) -> super::Flow<O>) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let definition = match agent_tool_definition::<I>() {
            Ok(definition) => definition,
            Err(err) => {
                self.tools.push(error_tool_spec(err));
                return self;
            }
        };
        let child = super::compile(flow).map_err(|err| err.to_string());
        self.push_tool_with_message::<I, O>(
            definition,
            child,
            Arc::new(decode_runtime_tool_result::<O>),
        );
        self
    }

    fn into_tools(self) -> Vec<AgentToolSpec> {
        self.tools
    }

    fn push_tool_with_message<I, CO>(
        &mut self,
        definition: ToolDefinition,
        child: Result<CompiledFlow<I, CO>, String>,
        render_result: Arc<
            dyn Fn(Value) -> Result<EdgeRenderedToolResult, EdgeToolMessageError> + Send + Sync,
        >,
    ) where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        CO: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let child_index = self.tools.len();
        let payload = AgentToolPayload {
            name: definition.name.clone(),
            child_index,
            description: definition.description.clone(),
            parameters: definition.parameters.clone(),
        };
        let runtime = EdgeAgentToolRuntime {
            decode_args: Arc::new(|value| {
                let input: I = serde_json::from_value(value).map_err(ToolError::TypeError)?;
                to_value(input).map_err(|err| ToolError::Fatal(err.to_string()))
            }),
            render_result,
        };
        match child {
            Ok(flow) => {
                let (graph, registry) = flow.into_parts();
                self.tools.push(AgentToolSpec {
                    payload,
                    graph,
                    registry,
                    runtime: Arc::new(runtime),
                });
            }
            Err(err) => self.tools.push(error_tool_spec(err)),
        }
    }
}

#[derive(Debug)]
enum EdgeToolMessageError {
    Fatal {
        expected: String,
        reason: String,
        raw: String,
    },
}

/// Fully compiled agent continuation pieces.
///
/// The typed layer embeds the payload/children in the graph and registers the
/// handler at runtime.
pub(crate) struct AgentBuild {
    /// Serializable payload embedded in the continuation node.
    pub payload: AgentPayload,
    /// Child graphs available to the continuation handler.
    pub children: Vec<UntypedGraph>,
    /// Registries required by child graphs.
    pub registries: Vec<HandlerRegistry>,
    /// Runtime handler registered for the continuation node.
    pub handler: AgentHandler,
    /// Build-time errors accumulated while preparing tools.
    pub errors: Vec<String>,
}

/// Builds the serializable payload and runtime handler for an agent node.
pub(crate) fn build_agent<I, O>(agent: Agent<O>) -> AgentBuild
where
    I: JsonSchema,
    O: JsonSchema,
{
    let (toolset, controller, configure, mut errors) = agent.into_parts();
    let tools = toolset.into_tools();
    let input_schema = schema_for::<I>();
    let output_schema = schema_for::<O>();
    let mut payload_tools = Vec::with_capacity(tools.len());
    let mut children = Vec::with_capacity(tools.len());
    let mut registries = Vec::with_capacity(tools.len());
    let mut runtime_tools = Vec::with_capacity(tools.len());
    let mut names = BTreeSet::new();
    for tool in tools {
        if !names.insert(tool.payload.name.clone()) {
            errors.push(format!("duplicate agent tool name '{}'", tool.payload.name));
            continue;
        }
        let mut payload = tool.payload;
        payload.child_index = children.len();
        payload_tools.push(payload);
        children.push(tool.graph);
        registries.push(tool.registry);
        runtime_tools.push(tool.runtime);
    }
    let configure = configure.unwrap_or_else(|| {
        errors.push("agent configure function is required".into());
        AgentConfigurator::missing()
    });
    let payload = AgentPayload {
        version: PAYLOAD_VERSION,
        agent_id: String::new(),
        configure_handler_key: String::new(),
        control_handler_key: controller.as_ref().map(|_| String::new()),
        input_schema,
        output_schema,
        output_type_name: O::schema_name(),
        tools: payload_tools,
    };
    AgentBuild {
        payload,
        children,
        registries,
        handler: AgentHandler {
            tools: runtime_tools,
            controller,
            configure,
        },
        errors,
    }
}

#[derive(Clone)]
/// Continuation handler that drives an agent session.
///
/// It owns runtime-only tool codecs and uses `ContinuationContext` for
/// runtime-owned history and VM transitions.
pub(crate) struct AgentHandler {
    tools: Vec<Arc<EdgeAgentToolRuntime>>,
    controller: Option<AgentController>,
    configure: AgentConfigurator,
}

struct PreparedToolCalls {
    child_calls: Vec<ContinuationChildCall>,
    recoverable_messages: Vec<Message>,
    running_tools: BTreeSet<String>,
    active: Vec<EdgeActiveToolCall>,
    waiting: Vec<EdgeWaitingToolCall>,
    results: Vec<EdgeCompletedToolCall>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum EdgeToolResult {
    Success { value: JsonValue },
    Error { value: JsonValue },
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value as JsonValue, json};

    use super::*;

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct MarkerLikeOutput {
        __pravah_edge_tool_error: bool,
        value: JsonValue,
    }

    /// Verifies diagnostic previews truncate Unicode only at character boundaries.
    #[test]
    fn unicode_tool_preview_never_slices_inside_a_character() {
        let value = Value::from(format!("{}é", "a".repeat(510)));
        let preview = preview_value(&value);

        assert!(preview.ends_with("..."));
        assert!(preview.contains('é'));
    }

    /// Verifies marker-shaped successful output remains a successful tagged result.
    #[test]
    fn tagged_tool_result_cannot_collide_with_user_output() {
        let output = json!({
            "__pravah_edge_tool_error": true,
            "value": {"legitimate": true}
        });
        let envelope =
            to_value(EdgeToolResult::Success { value: output }).expect("envelope should encode");
        let result = decode_handler_tool_result::<MarkerLikeOutput>(envelope)
            .expect("marker-shaped output should decode");

        assert!(result.message.content.contains("legitimate"));
    }
}
