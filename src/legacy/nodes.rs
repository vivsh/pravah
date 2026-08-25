use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::flow::FlowGraph;
use crate::legacy::NodeId;
use crate::legacy::errors::FlowError;
use crate::{
    clients::{Message, schema::sanitize_strict},
    context::Context,
    tools::base::pascal_to_snake,
    tools::{SuspendedValue, ToolDefinition, ToolError},
};

pub(crate) type ToolMessageFn =
    Box<dyn Fn(Value) -> Result<Message, ToolMessageError> + Send + Sync>;
pub(crate) type EitherFn = Box<dyn Fn(&Value) -> Result<(NodeId, Value), FlowError> + Send + Sync>;
pub(crate) type ForkFn = Box<dyn Fn(&Value) -> Result<Vec<StateNode>, FlowError> + Send + Sync>;
pub(crate) type JoinFn = Arc<dyn Fn(&[Value]) -> Result<StateNode, FlowError> + Send + Sync>;
pub(crate) type WorkFn =
    Box<dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync>;
pub(crate) type ToolWorkFn =
    Box<dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, ToolError>> + Send + Sync>;
pub(crate) type MapFn = Box<dyn Fn(&Value) -> Result<Value, FlowError> + Send + Sync>;
pub(crate) type SuspendDeserializeFn =
    Box<dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync>;

pub(crate) enum ToolMessageError {
    Recoverable(ToolError),
    FatalOutput {
        expected: String,
        reason: String,
        raw: String,
    },
}

impl From<ToolError> for ToolMessageError {
    fn from(error: ToolError) -> Self {
        Self::Recoverable(error)
    }
}

/// Builds a [`ToolDefinition`] for type `T` using its JSON Schema.
pub(crate) fn build_tool_definition<T: JsonSchema>() -> Result<ToolDefinition, String> {
    let raw =
        serde_json::to_value(schemars::r#gen::SchemaGenerator::default().root_schema_for::<T>())
            .map_err(|e| format!("schema serialization failed: {e}"))?;
    let description = raw
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    let parameters = sanitize_strict(raw);
    Ok(ToolDefinition {
        name: pascal_to_snake(&T::schema_name()),
        description,
        parameters,
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct StateNode {
    pub(crate) name: String,
    pub(crate) value: Value,
}

pub(crate) struct ToolInfo {
    pub(crate) definition: ToolDefinition,
    pub(crate) exit_id: NodeId,
    pub(crate) to_message: ToolMessageFn,
}

pub(crate) struct AgentInfo {
    pub(crate) id: NodeId,
    pub(crate) tools: Vec<ToolInfo>,
    pub(crate) make_message: fn(Value, &Context) -> Result<Message, FlowError>,
    pub(crate) preamble: String,
    pub(crate) input_schema: Value,
    pub(crate) model: String,
    pub(crate) exit: NodeId,
    pub(crate) output_schema: Value,
    pub(crate) tool_lookup: HashMap<String, (NodeId, NodeId)>,
    pub(crate) keep_alive: bool,
    pub(crate) turn_budget: Option<u32>,
    pub(crate) turn_budget_message: Option<String>,
    pub(crate) provider_config: Option<Value>,
    /// Schema name of the output type, used by clients that require a synthetic exit-tool.
    pub(crate) output_type_name: String,
}

pub(crate) struct EitherInfo {
    pub(crate) entry: NodeId,
    pub(crate) left_name: NodeId,
    pub(crate) right_name: NodeId,
    pub(crate) func: EitherFn,
}

pub(crate) struct ForkInfo {
    pub(crate) name: NodeId,
    pub(crate) children: Vec<NodeId>,
    pub(crate) func: ForkFn,
}

pub(crate) struct JoinInfo {
    pub(crate) parents: Vec<NodeId>,
    pub(crate) target: NodeId,
    pub(crate) func: JoinFn,
}

pub(crate) struct WorkInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    pub(crate) func: WorkFn,
}

/// A tool-aware work node whose implementation returns [`ToolError`] directly.
/// Non-fatal errors are forwarded to the model as structured error messages;
/// only [`ToolError::Fatal`] aborts the flow.
pub(crate) struct ToolWorkInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    pub(crate) agent_id: NodeId,
    pub(crate) tool_name: String,
    pub(crate) func: ToolWorkFn,
}

pub(crate) struct MapInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    pub(crate) func: MapFn,
}

pub(crate) struct SuspendInfo {
    pub(crate) entry: NodeId,
    pub(crate) exit: NodeId,
    pub(crate) output_type: String,
    pub(crate) deserialize: SuspendDeserializeFn,
}

/// Builds a typed [`StateNode`] from a value.
pub(crate) fn node<A: JsonSchema + Serialize>(input: A) -> Result<StateNode, FlowError> {
    let node_id = A::schema_name();
    let value = serde_json::to_value(&input).map_err(FlowError::Serialize)?;
    Ok(StateNode {
        name: node_id.to_string(),
        value,
    })
}

/// Fan-out node: runs the same child flow once for each element of a `Vec<F>` input,
/// collecting the results into a `Vec<F::Output>`.
pub(crate) struct EachInfo {
    /// Input slot NodeId (schema name of `Vec<F>` in the parent graph).
    /// Also acts as the feedback slot — the child's output is written here
    /// so that `step_inner` re-dispatches this node for the next item.
    pub(crate) id: NodeId,
    /// Output slot NodeId (schema name of `Vec<F::Output>` in the parent graph).
    pub(crate) exit: NodeId,
    /// Child graph that processes one `F` item at a time.
    pub(crate) inner: Arc<FlowGraph>,
    /// Index into the runtime's callable table, assigned by `collect_callables`.
    pub(crate) callable_index: usize,
}

pub(crate) enum FlowNode {
    Agent(Arc<AgentInfo>),
    Either(EitherInfo),
    Fork(ForkInfo),
    Join(JoinInfo),
    Work(WorkInfo),
    ToolWork(ToolWorkInfo),
    Map(MapInfo),
    Suspend(SuspendInfo),
    Flow(Arc<FlowGraph>),
    Each(Arc<EachInfo>),
}

impl AgentInfo {
    /// Returns the full system-prompt text sent to the LLM on the first turn:
    /// static preamble + optional memory context + input-schema hint.
    ///
    /// `env` is the result of [`MemoryFactory::retrieve`] — `None` when no factory
    /// is configured or when the factory returns nothing for this agent.
    pub(crate) fn effective_preamble(&self, env: Option<String>) -> String {
        let hint = format!(
            "The user message is JSON. Interpret it using this JSON Schema: {}",
            self.input_schema
        );
        match (self.preamble.is_empty(), env) {
            (true, None) => hint,
            (false, None) => format!("{}\n\n{}", self.preamble, hint),
            (true, Some(e)) => format!("{e}\n\n{hint}"),
            (false, Some(e)) => format!("{}\n\n{e}\n\n{hint}", self.preamble),
        }
    }
}
