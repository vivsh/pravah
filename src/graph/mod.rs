//! Graph-backed Pravah backend.
//!
//! This module exposes the serializable untyped graph, typed builder, VM
//! runtime, and continuation registry used by production graph workflows.

/// Agent facade built on continuation nodes.
mod agent;
/// Imperative untyped graph builder.
pub mod builder;
/// Chat helper built from mark/goto, keep-alive agent, and suspend.
pub mod chat;
/// Diagram rendering for serializable graph-backed flows.
pub mod diagram;
/// Error types for graph and VM failures.
pub mod error;
/// Dense graph identifier types.
pub mod ids;
/// Trusted transport-neutral JSON invocation facade.
pub mod json;
/// Optional Streamable HTTP MCP resource integration.
#[cfg(feature = "mcp")]
pub mod mcp;
/// Serializable graph data model.
pub mod model;
/// Runtime handler registry and continuation protocol.
pub mod registry;
/// Multi-frame edge VM runtime.
pub mod runtime;
/// Minimal runtime value-shape checks.
pub mod schema;
/// JSON helpers for untyped graphs.
pub mod serde;
/// Serializable VM state shapes.
mod state;
#[cfg(test)]
mod tests;
/// String-free typed builder and fluent API.
pub mod typed;
/// Graph and registry validation.
pub mod validation;
/// Compact, format-neutral values carried by the VM.
pub mod value;

pub use crate::diagram::{DiagramEdge, DiagramNode, DiagramNodeKind};
pub use agent::{
    Agent, AgentConfig, AgentDecision, AgentDirective, AgentInterventionPoint, AgentLoop,
    AgentLoopMetrics, AgentResume, AgentSuspension, AgentToolProposal, AgentToolResult,
    McpResourceRef, ToolFilter, ToolInfo, Toolset,
};
pub use builder::UntypedGraphBuilder;
pub use chat::{Chat, ChatTurn};
pub use diagram::GraphDiagram;
pub use error::GraphError;
pub use ids::{EdgeId, HandlerKey, MarkId, NodeId, VarId};
pub use json::{JSON_WIRE_VERSION, JsonInvoker, JsonRequest, JsonResponse};
#[cfg(feature = "mcp")]
pub use mcp::{McpError, McpResourceInfo, McpServer};
pub use model::{
    BuiltinNode, Edge, NodeKind, TypeSpec, UNTYPED_GRAPH_SCHEMA_VERSION, UntypedGraph, VarInit,
    VarKey, VarScope, Variable,
};
pub use registry::{
    ContinuationChildCall, ContinuationContext, ContinuationEvent, ContinuationHandler,
    ContinuationSuspension, ContinuationTransition, EdgeWrite, HandlerRegistry, RuntimeServices,
    ValueHandler, WorkHandler,
};
pub use runtime::{GraphFingerprint, PreparedGraph, Runtime, SNAPSHOT_VERSION, Snapshot};
pub use state::{State, Step, Suspension};
pub use typed::{
    CompiledFlow, EitherFlow, Flow, TypedEdge, TypedGraphBuilder, TypedMark, TypedVar, compile,
};
pub use value::{Value, ValueError, from_value, to_value};
