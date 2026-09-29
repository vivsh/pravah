pub mod clients;
mod commons;
pub mod context;
pub mod deps;
pub mod diagram;
pub mod graph;
pub mod history;
pub mod legacy;
#[cfg(feature = "testing")]
pub mod testing;
pub mod tools;
pub mod utils;

pub use context::{Context, FlowConf};
pub use graph::{
    Agent, AgentClientOperation, AgentConfig, AgentDecision, AgentDirective,
    AgentInterventionPoint, AgentLoop, AgentLoopMetrics, AgentResume, AgentSuspension,
    AgentToolProposal, AgentToolResult, Chat, ChatBuilder, ChatRequest, ChatStep, ChatTurn,
    CompiledFlow, EitherFlow, Fetch, FetchBody, FetchError, FetchExecutor, FetchRequest,
    FetchResponse, Flow, GraphError, McpResourceRef, Runtime, Snapshot, Step, Suspension,
    ToolFilter, ToolInfo, Toolset, TypedMark, TypedVar, compile,
};
#[cfg(feature = "mcp")]
pub use graph::{McpError, McpResourceInfo, McpServer};
pub use history::{
    CompactionRequest, CompactionResult, Compactor, HistoryEntry, HistoryStore, MessageHistory,
    NoopHistoryStore,
};
