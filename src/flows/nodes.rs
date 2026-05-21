use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::flow::FlowGraph;
use crate::flows::NodeId;
use crate::flows::errors::FlowError;
use crate::{
    clients::{Message, schema::sanitize_strict},
    context::Context,
    tools::base::pascal_to_snake,
    tools::{SuspendedValue, ToolDefinition, ToolError},
};

/// Builds a [`ToolDefinition`] for type `T` using its JSON Schema.
pub(crate) fn build_tool_definition<T: JsonSchema>() -> Result<ToolDefinition, String> {
    let raw = serde_json::to_value(schemars::r#gen::SchemaGenerator::default().root_schema_for::<T>())
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
    pub(crate) to_message: Box<dyn Fn(Value) -> Result<Message, ToolError> + Send + Sync>,
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
}

pub(crate) struct EitherInfo {
    pub(crate) entry: NodeId,
    pub(crate) left_name: NodeId,
    pub(crate) right_name: NodeId,
    pub(crate) func: Box<dyn Fn(&Value) -> Result<(NodeId, Value), FlowError> + Send + Sync>,
}

pub(crate) struct ForkInfo {
    pub(crate) name: NodeId,
    pub(crate) children: Vec<NodeId>,
    pub(crate) func: Box<dyn Fn(&Value) -> Result<Vec<StateNode>, FlowError> + Send + Sync>,
}

pub(crate) struct JoinInfo {
    pub(crate) parents: Vec<NodeId>,
    pub(crate) target: NodeId,
    pub(crate) func: Arc<dyn Fn(&[Value]) -> Result<StateNode, FlowError> + Send + Sync>,
}

pub(crate) struct WorkInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    pub(crate) func:
        Box<dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync>,
}

pub(crate) struct MapInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    pub(crate) func: Box<dyn Fn(&Value) -> Result<Value, FlowError> + Send + Sync>,
}

pub(crate) struct SuspendInfo {
    pub(crate) entry: NodeId,
    pub(crate) exit: NodeId,
    pub(crate) output_type: String,
    pub(crate) deserialize: Box<dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync>,
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

pub(crate) enum FlowNode {
    Agent(Arc<AgentInfo>),
    Either(EitherInfo),
    Fork(ForkInfo),
    Join(JoinInfo),
    Work(WorkInfo),
    Map(MapInfo),
    Suspend(SuspendInfo),
    Flow(Arc<FlowGraph>),
}