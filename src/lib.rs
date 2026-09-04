pub mod clients;
mod commons;
pub mod context;
pub mod deps;
pub mod diagram;
pub mod graph;
pub mod legacy;
#[cfg(feature = "testing")]
pub mod testing;
pub mod tools;
pub mod utils;

pub use context::{Context, FlowConf};
pub use graph::{
    Agent, AgentConfig, AgentDecision, AgentDirective, AgentInterventionPoint, AgentLoop,
    AgentLoopMetrics, AgentResume, AgentSuspension, AgentToolProposal, AgentToolResult, Chat,
    ChatTurn, CompiledFlow, EitherFlow, Flow, GraphError, McpResourceRef, Runtime, Snapshot, Step,
    Suspension, ToolFilter, ToolInfo, Toolset, TypedMark, TypedVar, compile,
};
#[cfg(feature = "mcp")]
pub use graph::{McpError, McpResourceInfo, McpServer};
