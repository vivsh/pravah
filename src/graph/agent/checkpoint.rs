use serde::{Deserialize, Serialize};

use crate::clients::TokenUsage;
use crate::graph::value::Value;

use super::budget::AgentBudgetState;
use super::config::ResolvedAgentConfig;
use super::control::{AgentLoopMetrics, AgentToolProposal, AgentToolResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EdgeAgentCheckpoint {
    pub(super) version: u32,
    pub(super) phase: EdgeAgentPhase,
    pub(super) session_id: String,
    pub(super) input: Value,
    pub(super) resolved: ResolvedAgentConfig,
    pub(super) selected_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) budget: Option<AgentBudgetState>,
    pub(super) guidance: Option<String>,
    pub(super) metrics: AgentLoopMetrics,
    pub(super) control_state: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EdgeAgentSavedState {
    pub(super) version: u32,
    pub(super) session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(super) enum EdgeAgentPhase {
    BeforeModel,
    Dispatch {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        conclusion: Option<ConclusionCause>,
    },
    BeforeTools {
        thought: Option<String>,
        calls: Vec<EdgeProposedToolCall>,
        usage: Option<TokenUsage>,
    },
    AcceptedTools {
        thought: Option<String>,
        calls: Vec<EdgeProposedToolCall>,
        usage: Option<TokenUsage>,
    },
    PendingTool {
        active: Vec<EdgeActiveToolCall>,
        waiting: Vec<EdgeWaitingToolCall>,
        results: Vec<EdgeCompletedToolCall>,
    },
    AfterTools {
        results: Vec<AgentToolResult>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ConclusionCause {
    Explicit,
    TurnBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EdgeProposedToolCall {
    pub(super) proposal: AgentToolProposal,
    pub(super) thought_signatures: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EdgeActiveToolCall {
    pub(super) position: usize,
    pub(super) call_id: String,
    pub(super) tool_name: String,
    pub(super) child_index: usize,
    pub(super) args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EdgeWaitingToolCall {
    pub(super) position: usize,
    pub(super) call_id: String,
    pub(super) tool_name: String,
    pub(super) child_index: usize,
    pub(super) args: Value,
    pub(super) input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EdgeCompletedToolCall {
    pub(super) position: usize,
    pub(super) result: AgentToolResult,
}
