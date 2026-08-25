use std::collections::BTreeSet;
use std::future::Future;
use std::sync::Arc;

use futures::future::{BoxFuture, FutureExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::clients::{
    Client, ClientOptions, ClientOutput, Message, Role, ToolCall, ToolChoice, ToolDefinition,
    materialize_messages,
};
use crate::context::Context;
use crate::legacy::build_tool_definition;
use crate::tools::{Tool, ToolError, ToolOutput};

use super::error::GraphError;
use super::model::UntypedGraph;
use super::registry::{
    ContinuationChildCall, ContinuationContext, ContinuationEvent, ContinuationHandler,
    ContinuationTransition,
};
use super::value::{Value, from_value, to_value};
use super::{CompiledFlow, HandlerRegistry};

mod config;
mod definition;
pub(crate) mod support;

use config::ResolvedAgentConfig;
pub(crate) use config::ResolvedResource;
pub use config::{AgentConfig, McpResourceRef, ToolFilter, ToolInfo};
pub use definition::Agent;
use definition::AgentConfigurator;
use support::*;

const PAYLOAD_VERSION: u32 = 2;
const CHECKPOINT_VERSION: u32 = 2;

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
    Ok(())
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
    to_message: Arc<dyn Fn(Value) -> Result<Message, EdgeToolMessageError> + Send + Sync>,
}

#[derive(Default)]
/// Typed toolset containing every child graph an agent may expose.
pub struct Toolset {
    tools: Vec<AgentToolSpec>,
}

impl Toolset {
    /// Registers a `Tool` implementation as an agent tool.
    pub fn tool<T: Tool>(mut self) -> Self
    where
        T::Input: Sync,
        T::Output: Sync,
    {
        let definition = match build_tool_definition::<T::Input>() {
            Ok(definition) => definition,
            Err(err) => {
                self.tools.push(error_tool_spec(err));
                return self;
            }
        };
        let child = build_tool_handler_flow::<T::Input, T::Output, _, _>(|input, ctx| async move {
            T::call(input, ctx).await
        });
        self.push_tool_with_message::<T::Input, JsonValue>(
            definition,
            child,
            Arc::new(|value| {
                let value = decode_tool_result(value)?;
                let value = match value {
                    EdgeToolResult::Success { value } => value,
                    EdgeToolResult::Error { value } => {
                        return Ok(Message::tool_output(String::new(), value.to_string()));
                    }
                };
                let output: T::Output =
                    serde_json::from_value(value).map_err(|err| EdgeToolMessageError::Fatal {
                        expected: T::Output::schema_name(),
                        reason: err.to_string(),
                        raw: "<tool output>".into(),
                    })?;
                T::to_message(output).map_err(Into::into)
            }),
        );
        self
    }

    /// Registers an inline async tool handler.
    pub fn tool_handler<I, O, Fut, H>(mut self, func: H) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: ToolOutput + Sync,
        Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
        H: Fn(I, Context) -> Fut + Send + Sync + 'static,
    {
        let definition = match build_tool_definition::<I>() {
            Ok(definition) => definition,
            Err(err) => {
                self.tools.push(error_tool_spec(err));
                return self;
            }
        };
        let child = build_tool_handler_flow::<I, O, Fut, H>(func);
        self.push_tool_with_message::<I, JsonValue>(
            definition,
            child,
            Arc::new(decode_handler_tool_message::<O>),
        );
        self
    }

    /// Registers a function-defined flow as an agent tool.
    pub fn flow<I, O>(mut self, flow: fn(super::Flow<I>) -> super::Flow<O>) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: ToolOutput + Sync,
    {
        let definition = match build_tool_definition::<I>() {
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
            Arc::new(decode_runtime_tool_message::<O>),
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
        to_message: Arc<dyn Fn(Value) -> Result<Message, EdgeToolMessageError> + Send + Sync>,
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
            to_message,
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
    Recoverable(ToolError),
    Fatal {
        expected: String,
        reason: String,
        raw: String,
    },
}

impl From<ToolError> for EdgeToolMessageError {
    fn from(value: ToolError) -> Self {
        Self::Recoverable(value)
    }
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
    let (toolset, configure, mut errors) = agent.into_parts();
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
    configure: AgentConfigurator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeAgentCheckpoint {
    version: u32,
    phase: EdgeAgentPhase,
    session_id: String,
    resolved: ResolvedAgentConfig,
    turn_offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeAgentSavedState {
    version: u32,
    session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum EdgeAgentPhase {
    Dispatch,
    PendingTool {
        active: Vec<EdgeActiveToolCall>,
        waiting: Vec<EdgeWaitingToolCall>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeActiveToolCall {
    call_id: String,
    tool_name: String,
    child_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeWaitingToolCall {
    call_id: String,
    tool_name: String,
    child_index: usize,
    args: Value,
}

struct PreparedToolCalls {
    child_calls: Vec<ContinuationChildCall>,
    recoverable_messages: Vec<Message>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum EdgeToolResult {
    Success { value: JsonValue },
    Error { value: JsonValue },
}

impl ContinuationHandler for AgentHandler {
    fn start<'a>(
        &'a self,
        payload: &'a Value,
        state: Option<Value>,
        inputs: Vec<Value>,
        ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        async move {
            let payload = decode_payload(payload)?;
            let input = single_input(inputs, "agent")?;
            let config = self
                .configure
                .configure(input, ctx.context().clone())
                .await?;
            let (resolved, message) = resolve_agent_config(&payload, config, ctx.context()).await?;
            let session_id = if resolved.keep_alive {
                restore_agent_state(state)?.unwrap_or_else(|| Uuid::now_v7().to_string())
            } else {
                Uuid::now_v7().to_string()
            };
            ctx.push_history(&session_id, &payload.agent_id, message)
                .await?;
            let turn_offset = ctx.complete_turn_count(&session_id).await;
            let checkpoint = EdgeAgentCheckpoint {
                version: CHECKPOINT_VERSION,
                phase: EdgeAgentPhase::Dispatch,
                session_id,
                resolved,
                turn_offset,
            };
            persist_checkpoint(checkpoint)
        }
        .boxed()
    }

    fn advance<'a>(
        &'a self,
        payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        async move {
            let payload = decode_payload(payload)?;
            let mut checkpoint: EdgeAgentCheckpoint = from_value(checkpoint).map_err(|err| {
                GraphError::SnapshotValidation(format!("failed to decode agent checkpoint: {err}"))
            })?;
            if checkpoint.version != CHECKPOINT_VERSION {
                return Err(GraphError::UnsupportedVersion {
                    format: "agent checkpoint",
                    got: checkpoint.version,
                    expected: CHECKPOINT_VERSION,
                });
            }
            validate_checkpoint(&payload, &checkpoint)?;
            match event {
                ContinuationEvent::Poll => self.poll(&payload, &mut checkpoint, ctx).await,
                ContinuationEvent::ChildResult { call_id, output } => {
                    self.child_result(&payload, &mut checkpoint, call_id, output, ctx)
                        .await
                }
            }
        }
        .boxed()
    }
}

impl AgentHandler {
    async fn poll(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        match &checkpoint.phase {
            EdgeAgentPhase::Dispatch => self.dispatch(payload, checkpoint, ctx).await,
            EdgeAgentPhase::PendingTool { .. } => persist_checkpoint(checkpoint.clone()),
        }
    }

    async fn dispatch(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let request_ctx = ctx.context();
        let resolved = &checkpoint.resolved;
        let preamble = effective_preamble(payload, resolved);
        let selected: BTreeSet<&str> = resolved.tools.iter().map(String::as_str).collect();

        let defs: Vec<ToolDefinition> = payload
            .tools
            .iter()
            .filter(|tool| selected.contains(tool.name.as_str()))
            .map(|tool| ToolDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            })
            .collect();
        let tool_choice = if defs.is_empty() {
            ToolChoice::Disabled
        } else {
            ToolChoice::Auto
        };
        let options = ClientOptions {
            output_type_name: payload.output_type_name.clone(),
            turn_budget: resolved.turn_budget,
            turn_budget_message: resolved.turn_budget_message.clone(),
            provider_config: resolved.provider_config.clone(),
            ..ClientOptions::default()
        }
        .with_input_schema(payload.input_schema.clone())
        .with_tools(defs)
        .with_tool_choice(tool_choice)
        .with_name(payload.agent_id.clone())
        .with_output_schema(payload.output_schema.clone())
        .with_preamble(preamble);
        let client = request_ctx
            .client_factory()
            .create(&resolved.model, options)
            .map_err(|err| GraphError::AgentClient(format!("client creation failed: {err}")))?;
        ctx.validate_history_for_session(&checkpoint.session_id)
            .await?;
        let session_history = ctx.history_for_session(&checkpoint.session_id).await;
        let mut messages = materialize_messages(&session_history, request_ctx)
            .await
            .map_err(|err| GraphError::Invalid(format!("message materialization failed: {err}")))?;
        let completed_turns = ctx.complete_turn_count(&checkpoint.session_id).await;
        maybe_inject_edge_turn_budget_message(
            &*client,
            &payload.agent_id,
            &mut messages,
            checkpoint.turn_offset,
            completed_turns,
        );
        let response = client
            .execute(&messages)
            .await
            .map_err(|err| GraphError::AgentClient(format!("execution failed: {err}")))?;
        let usage = response.usage;
        match response.output {
            ClientOutput::Output(output) => {
                let content = serde_json::to_string(&output).map_err(|err| {
                    GraphError::Invalid(format!("failed to serialize agent output: {err}"))
                })?;
                let message = match usage {
                    Some(usage) => Message::assistant(content).with_usage(usage),
                    None => Message::assistant(content),
                };
                let state = completed_agent_state(checkpoint)?;
                let output = to_value(output).map_err(|err| GraphError::ValueConversion {
                    target: "agent output".into(),
                    reason: err.to_string(),
                })?;
                let transition = ContinuationTransition {
                    checkpoint: None,
                    state,
                    outputs: vec![output],
                    writes: Vec::new(),
                    child_calls: Vec::new(),
                };
                ctx.push_history(&checkpoint.session_id, &payload.agent_id, message)
                    .await?;
                ctx.compact_history(&checkpoint.session_id).await?;
                Ok(transition)
            }
            ClientOutput::ToolCalls { thought, calls } => {
                let prepared = self.prepare_tool_calls(payload, checkpoint, calls.clone())?;
                let atc = Message {
                    role: Role::AssistantToolCalls {
                        calls: calls.clone(),
                    },
                    content: thought.unwrap_or_default(),
                    attachments: Vec::new(),
                    usage,
                };
                let mut messages = Vec::with_capacity(1 + prepared.recoverable_messages.len());
                messages.push(atc);
                messages.extend(prepared.recoverable_messages);
                ctx.push_history_batch(&checkpoint.session_id, &payload.agent_id, messages)
                    .await?;
                ctx.compact_history(&checkpoint.session_id).await?;
                let child_calls = prepared.child_calls;
                if child_calls.is_empty() {
                    checkpoint.phase = EdgeAgentPhase::Dispatch;
                    persist_checkpoint(checkpoint.clone())
                } else {
                    let active = child_calls
                        .iter()
                        .filter_map(|call| {
                            payload
                                .tools
                                .get(call.child_index)
                                .map(|tool| EdgeActiveToolCall {
                                    call_id: call.call_id.clone(),
                                    tool_name: tool.name.clone(),
                                    child_index: call.child_index,
                                })
                        })
                        .collect();
                    checkpoint.phase = EdgeAgentPhase::PendingTool {
                        active,
                        waiting: pending_waiting(checkpoint),
                    };
                    transition_with_children(checkpoint.clone(), child_calls)
                }
            }
        }
    }

    fn prepare_tool_calls(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        calls: Vec<ToolCall>,
    ) -> Result<PreparedToolCalls, GraphError> {
        let mut seen = BTreeSet::new();
        let mut active_tools: BTreeSet<String> = BTreeSet::new();
        let mut child_calls = Vec::new();
        let mut waiting = Vec::new();
        let mut recoverable_messages = Vec::new();
        for call in calls {
            if !seen.insert(call.id.clone()) {
                return Err(GraphError::Invalid(format!(
                    "agent '{}' returned duplicate tool call id '{}' for tool '{}'",
                    payload.agent_id, call.id, call.name
                )));
            }
            let Some(tool) = payload.tools.iter().find(|tool| {
                tool.name == call.name && checkpoint.resolved.tools.contains(&tool.name)
            }) else {
                recoverable_messages.push(Message::tool_output(
                    call.id,
                    format!(r#"{{"error":"unknown tool '{}'"}}"#, call.name),
                ));
                continue;
            };
            let runtime = self.tool_runtime(tool.child_index)?;
            let input = match (runtime.decode_args)(call.args) {
                Ok(input) => input,
                Err(err) if !err.is_fatal() => {
                    recoverable_messages
                        .push(err.into_error_message(&tool.name).with_call_id(call.id));
                    continue;
                }
                Err(err) => return Err(GraphError::Invalid(err.to_string())),
            };
            if active_tools.insert(tool.name.clone()) {
                child_calls.push(ContinuationChildCall {
                    child_index: tool.child_index,
                    call_id: call.id,
                    input,
                });
            } else {
                waiting.push(EdgeWaitingToolCall {
                    call_id: call.id,
                    tool_name: tool.name.clone(),
                    child_index: tool.child_index,
                    args: input,
                });
            }
        }
        checkpoint.phase = EdgeAgentPhase::PendingTool {
            active: Vec::new(),
            waiting,
        };
        Ok(PreparedToolCalls {
            child_calls,
            recoverable_messages,
        })
    }

    async fn child_result(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        call_id: String,
        output: Value,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let EdgeAgentPhase::PendingTool { active, waiting } = &mut checkpoint.phase else {
            return Err(GraphError::Invalid(format!(
                "agent '{}' received child result while not waiting for tools",
                payload.agent_id
            )));
        };
        let active_pos = active
            .iter()
            .position(|call| call.call_id == call_id)
            .ok_or_else(|| {
                GraphError::Invalid(format!(
                    "agent '{}' received unknown tool child result '{}'",
                    payload.agent_id, call_id
                ))
            })?;
        let active_call = active.remove(active_pos);
        let tool = payload
            .tools
            .get(active_call.child_index)
            .ok_or_else(|| GraphError::Invalid("tool child index is invalid".into()))?;
        let runtime = self.tool_runtime(active_call.child_index)?;
        let mut message = match (runtime.to_message)(output) {
            Ok(message) => message,
            Err(EdgeToolMessageError::Recoverable(err)) if !err.is_fatal() => {
                err.into_error_message(&tool.name)
            }
            Err(EdgeToolMessageError::Recoverable(err)) => {
                return Err(GraphError::Invalid(err.to_string()));
            }
            Err(EdgeToolMessageError::Fatal {
                expected,
                reason,
                raw,
            }) => {
                return Err(GraphError::Invalid(format!(
                    "tool '{}' output decode failed; expected {expected}: {reason}; raw: {raw}",
                    tool.name
                )));
            }
        };
        message = message.with_call_id(call_id);
        ctx.push_history(&checkpoint.session_id, &payload.agent_id, message)
            .await?;
        ctx.compact_history(&checkpoint.session_id).await?;

        let mut child_calls = Vec::new();
        if let Some(waiting_pos) = waiting
            .iter()
            .position(|call| call.tool_name == active_call.tool_name)
        {
            let next = waiting.remove(waiting_pos);
            child_calls.push(ContinuationChildCall {
                child_index: next.child_index,
                call_id: next.call_id.clone(),
                input: next.args,
            });
            active.push(EdgeActiveToolCall {
                call_id: next.call_id,
                tool_name: next.tool_name,
                child_index: next.child_index,
            });
        }

        if active.is_empty() {
            checkpoint.phase = EdgeAgentPhase::Dispatch;
        }
        if child_calls.is_empty() {
            persist_checkpoint(checkpoint.clone())
        } else {
            transition_with_children(checkpoint.clone(), child_calls)
        }
    }

    fn tool_runtime(&self, index: usize) -> Result<&EdgeAgentToolRuntime, GraphError> {
        self.tools
            .get(index)
            .map(Arc::as_ref)
            .ok_or_else(|| GraphError::Invalid(format!("tool runtime {index} is missing")))
    }
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

    impl ToolOutput for MarkerLikeOutput {}

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
        let message = decode_handler_tool_message::<MarkerLikeOutput>(envelope)
            .expect("marker-shaped output should decode");

        assert!(message.content.contains("legitimate"));
    }
}
