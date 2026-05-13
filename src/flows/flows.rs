use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

fn new_session_id() -> String {
    Uuid::now_v7().to_string()
}

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::flows::diagram::{build_diagram, DiagramNodeKind, FlowGraphDiagram, NodeDesc};
use crate::flows::errors::{AgentError, BuildError};
use crate::flows::phase::{AgentContinuation, Phase};
use crate::flows::state::FlowState;
use crate::{
    clients::{
        ClientFactory, ClientOptions, DefaultClientFactory, Message, Role, ToolCall, ToolChoice,
    },
    clients::ClientOutput,
    commons::Agent,
    context::Context,
    tools::{ToolBox, ToolError},
};
use super::history::FlowHistory;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct StateNode {
    name: String,
    value: serde_json::Value,
}

#[derive(Debug, Error)]
pub enum FlowError {
    #[error("Node not found: {0}")]
    NotFound(String),

    #[error("Snapshot load error: {0}")]
    SnapLoadError(String),

    #[error("Snapshot store error: {0}")]
    SnapStoreError(String),

    #[error("Build error: {0}")]
    BuildError(String),

    #[error("Serialize error: {0}")]
    SerializeError(String),

    #[error("Deserialize error: {0}")]
    DeserializeError(String),

    #[error("Failed to resume due to mismatch between suspend and resume payloads for tool '{0}'")]
    ResumeMismatchError(String),

    #[error("Flow is suspended at '{0}' — call resume() with a resumption payload, not next()")]
    ResumeRequired(String),

    #[error("Flow is not suspended — unexpected resumption payload supplied for '{0}'")]
    UnexpectedResumption(String),

    #[error("Agent error: {0}")]
    AgentError(String),

    #[error("Flow deadlock: states [{0}] are waiting but no join is ready")]
    Deadlock(String),

    #[error("Flow graph is invalid:\n{}", .0.join("\n"))]
    Invalid(Vec<String>),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Recorded for each outstanding `FlowNode::Tool` in the state map.
#[derive(Serialize, Deserialize)]
struct PendingCall {
    id: String,
    args: Value,
}

struct AgentInfo {
    name: String,
    tool_box: Arc<ToolBox>,
    preamble: String,
    model: String,
    exit_name: String,
    output_schema: Value,
}

/// Metadata for a single tool exposed by an agent, stored as a graph node so the
/// flow engine can dispatch it independently rather than inlining all tool calls.
struct ToolInfo {
    /// State-map key used for this tool: `"AgentName::tool_name"`.
    name: String,
    /// Agent state key to return to after the tool completes (agent's input key).
    agent_name: String,
    /// Zero-based index into `tool_box.tools`.
    tool_index: usize,
    /// Shared toolbox owned by the parent agent.
    tool_box: Arc<ToolBox>,
}

struct EitherInfo {
    name: String,
    left_name: String,
    right_name: String,
    func: Box<dyn Fn(&Value, Context) -> Result<StateNode, FlowError> + Send + Sync>,
}

struct ForkInfo {
    name: String,
    children: Vec<String>,
    func: Box<dyn Fn(&Value, Context) -> Result<Vec<StateNode>, FlowError> + Send + Sync>,
}

struct JoinInfo {
    parents: Vec<String>,
    target: String,
    func: Arc<dyn Fn(&[Value], Context) -> Result<StateNode, FlowError> + Send + Sync>,
}

struct WorkInfo {
    name: String,
    exit_name: String,
    func:
        Box<dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync>,
}

/// Constructs a typed [`StateNode`] from an [`Agent`] input value.
pub(crate) fn node<A: JsonSchema + Serialize>(input: A) -> Result<StateNode, FlowError> {
    let node_id = A::schema_name();
    let value = serde_json::to_value(&input)
        .map_err(|e| FlowError::SerializeError(format!("node '{}': {e}", node_id)))?;
    Ok(StateNode {
        name: node_id.to_string(),
        value,
    })
}

enum FlowNode {
    Agent(AgentInfo),
    Either(EitherInfo),
    Fork(ForkInfo),
    Join(JoinInfo),
    Work(WorkInfo),
    /// An embedded sub-flow. Boxed to break the recursive type-size cycle.
    Flow(Box<FlowGraph>),
    /// A single tool dispatched by a parent agent. Not statically reachable from the
    /// entry node; it enters the state map when the agent issues a tool call.
    Tool(ToolInfo),
}

pub(crate) enum FlowOut {
    Continue,
    Done(Value),
    Suspend { value: Value, tool_id: String },
}

/// Typed step result returned by [`FlowRuntime`].
#[derive(Debug)]
pub enum RunOut<O> {
    Continue,
    Done(O),
    Suspend { value: Value, tool_id: String },
}

pub trait Flow: 'static + JsonSchema + Serialize + DeserializeOwned + Send + Sync {
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;

    fn build() -> Result<FlowGraph, FlowError>;

    fn node_id() -> String {
        Self::schema_name()
    }
}

pub struct FlowGraph {
    nodes: HashMap<String, FlowNode>,
    entry: String,
    /// Input schema name when this graph is embedded as a sub-flow; `""` for the root graph.
    pub(crate) name: String,
    /// Output schema name written to the parent frame when this sub-flow completes; `""` for root.
    pub(crate) exit_name: String,
}

impl FlowGraph {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry: String::new(),
            name: String::new(),
            exit_name: String::new(),
        }
    }

    pub fn builder() -> FlowBuilder {
        FlowBuilder::new()
    }

    /// Produce a [`FlowGraphDiagram`] snapshot of this graph's topology.
    ///
    /// Called internally by [`FlowGraphDiagram::for_flow`]. Use that method
    /// from outside the crate.
    pub(crate) fn diagram(&self) -> FlowGraphDiagram {
        let descs: Vec<NodeDesc> = self
            .nodes
            .iter()
            .filter_map(|(key, node)| {
                let (kind, succs): (DiagramNodeKind, Vec<(String, &'static str)>) = match node {
                    FlowNode::Agent(info) => (
                        DiagramNodeKind::Agent,
                        vec![(info.exit_name.clone(), "agent")],
                    ),
                    FlowNode::Work(info) => (
                        DiagramNodeKind::Work,
                        vec![(info.exit_name.clone(), "work")],
                    ),
                    FlowNode::Fork(info) => (
                        DiagramNodeKind::Fork,
                        info.children
                            .iter()
                            .map(|c| (c.clone(), "fork"))
                            .collect(),
                    ),
                    FlowNode::Join(info) => (
                        DiagramNodeKind::Join,
                        vec![(info.target.clone(), "join")],
                    ),
                    FlowNode::Either(info) => (
                        DiagramNodeKind::Either,
                        vec![
                            (info.left_name.clone(), "either"),
                            (info.right_name.clone(), "either"),
                        ],
                    ),
                    FlowNode::Flow(inner) => (
                        DiagramNodeKind::Flow,
                        vec![(inner.exit_name.clone(), "flow")],
                    ),
                    // Tool nodes are implementation details not shown in diagrams.
                    FlowNode::Tool(_) => return None,
                };
                Some(NodeDesc {
                    id: key.clone(),
                    kind,
                    succs,
                })
            })
            .collect();
        build_diagram(self.entry.clone(), descs)
    }

    pub(crate) fn is_terminal(&self, state_name: &str) -> bool {
        !self.nodes.contains_key(state_name)
    }

    /// A join node is ready to execute when all its parent nodes have produced values.
    /// Can only be tested at runtime, not build time, because parent nodes may be dynamically generated agents.
    /// join target can be a terminal and may not be present in the node map at build time.
    fn can_join(&self, node_id: &str, state: &FlowState) -> bool {
        if let Some(FlowNode::Join(join_info)) = self.nodes.get(node_id) {
            join_info.parents.iter().all(|p| state.contains_state(p))
        } else {
            false
        }
    }

    async fn handle_work(
        node: &WorkInfo,
        ctx: Context,
        states: &mut FlowState,
    ) -> Result<(), FlowError> {
        let state = states.get_state(&node.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "work node '{}' has not produced a value",
                node.name
            ))
        })?;
        let output = (node.func)(&state, ctx).await?;
        states.set_state(node.exit_name.as_str(), output, Some(&node.name));
        Ok(())
    }

    fn handle_fork(node: &ForkInfo, ctx: Context, states: &mut FlowState) -> Result<(), FlowError> {
        let state = states.get_state(&node.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "fork parent '{}' has not produced a value",
                node.name
            ))
        })?;

        let children = (node.func)(state, ctx)?;
        if children.len() != node.children.len() {
            return Err(BuildError::NodeConflict(format!(
                "fork node '{}' produced {} child states but has {} child nodes",
                node.name,
                children.len(),
                node.children.len()
            )).into());
        }
        for child in children {
            states.set_state(&child.name, child.value, None);
        }
        states.remove_state(node.name.as_str());

        Ok(())
    }

    fn handle_join(node: &JoinInfo, ctx: Context, states: &mut FlowState) -> Result<(), FlowError> {
        let mut inputs = Vec::with_capacity(node.parents.len());
        for p in &node.parents {
            let value = states.get_state(p).ok_or_else(|| {
                FlowError::NotFound(format!("join parent '{}' has not produced a value", p))
            })?;
            inputs.push(value.clone());
        }
        let output = (node.func)(&inputs, ctx)?;
        states.set_state(&node.target, output.value, None);

        for p in &node.parents {
            states.remove_state(p.as_str());
        }

        Ok(())
    }

    /// Calls the LLM and returns the new `Phase` for the agent.
    ///
    /// - `Phase::Entry`: pushes `initial_message` as the first user turn, then calls the LLM.
    /// - `Phase::Continue(Waiting{0, None})`: all tools done, no submit — re-calls LLM (multi-turn).
    /// - Any other phase: callers must not invoke this function.
    ///
    /// Returns `Phase::Continue(ToolsDispatched{calls})` or `Phase::Exit(value)`.
    async fn handle_agent(
        agent: &AgentInfo,
        phase: &Phase,
        initial_message: Option<&str>,
        factory: &dyn ClientFactory,
        history: &mut FlowHistory,
        session_id: &str,
    ) -> Result<Phase, AgentError> {
        // Phase::Entry: push the initial user message before calling the LLM.
        if matches!(phase, Phase::Entry) {
            let msg = initial_message.ok_or_else(|| {
                AgentError::Llm("initial_message missing for Phase::Entry".to_string())
            })?;
            history.push(Message::user(msg.to_string()).with_context(&agent.name, session_id));
        }
        let options = if agent.tool_box.is_empty() {
            ClientOptions::default()
                .with_preamble(&agent.preamble)
                .with_output_schema(agent.output_schema.clone())
                .with_tool_choice(ToolChoice::Disabled)
        } else {
            ClientOptions::default()
                .with_preamble(&agent.preamble)
                .with_tools(agent.tool_box.definitions())
        };
        let client = factory
            .create(&agent.model, options)
            .map_err(|e| AgentError::Llm(e.to_string()))?;
        let resp = client
            .execute(&history.for_session(session_id))
            .await
            .map_err(|e| AgentError::Llm(e.to_string()))?;
        match resp.output {
            ClientOutput::ToolCalls { thought, calls } => {
                history.push(Message {
                    role: Role::AssistantToolCalls { calls: calls.clone() },
                    content: thought.unwrap_or_default(),
                    usage: resp.usage,
                    agent_id: agent.name.clone(),
                    session_id: session_id.to_owned(),
                });
                let cont = serde_json::to_value(AgentContinuation::ToolsDispatched { calls })
                    .map_err(|e| AgentError::Serialize(e.to_string()))?;
                Ok(Phase::Continue(cont))
            }
            ClientOutput::Output(value) => Ok(Phase::Exit(value)),
        }
    }

    fn handle_either(
        either: &EitherInfo,
        ctx: Context,
        states: &mut FlowState,
    ) -> Result<(), FlowError> {
        let state = states.get_state(&either.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "either parent '{}' has not produced a value",
                either.name
            ))
        })?;
        let output = (either.func)(&state, ctx)?;
        states.set_state(&output.name, output.value, Some(&either.name));
        Ok(())
    }

    /// Injects the entry point and runs full graph validation (entry + reachability).
    /// Called by [`FlowRuntime`] after [`Flow::build`] returns the graph.
    pub(crate) fn with_entry(mut self, entry: String) -> Result<Self, FlowError> {
        self.entry = entry;
        validate(&self.nodes, &self.entry)?;
        Ok(self)
    }

    pub(crate) async fn next(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        session_id: &str,
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        self.step(factory, ctx, history, session_id, None, states).await
    }

    pub(crate) async fn resume(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        session_id: &str,
        resumption: (String, Value),
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        self.step(factory, ctx, history, session_id, Some(resumption), states)
            .await
    }

    /// Checks whether all current states are terminal (Done) or stuck (Deadlock).
    /// When inside a sub-flow frame, a Done pops the frame and writes the result to the parent.
    fn resolve_done_or_deadlock(&self, states: &mut FlowState) -> Result<FlowOut, FlowError> {
        if states.keys().all(|k| self.is_terminal(k)) {
            let value = states
                .keys()
                .next()
                .and_then(|k| states.get_state(k))
                .cloned()
                .unwrap_or(Value::Null);
            if states.frame_depth() > 1 {
                let depth = states.frame_depth();
                let exit_name = states
                    .frame_exit_name_at(depth - 1)
                    .ok_or_else(|| FlowError::Internal("pop: sub-frame has no exit_name".into()))?
                    .to_string();
                states.pop_frame();
                states.set_state(&exit_name, value, None);
                return Ok(FlowOut::Continue);
            }
            return Ok(FlowOut::Done(value));
        }
        let stuck: Vec<&str> = states
            .keys()
            .filter(|k| !self.is_terminal(k))
            .map(String::as_str)
            .collect();
        Err(FlowError::Deadlock(stuck.join(", ")))
    }

    /// Dispatches one step on `self` (the currently active graph for the top frame).
    async fn step_inner(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        session_id: &str,
        resumption: Option<(String, Value)>,
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        let total_states = states.len();
        for state_index in 0..total_states {
            let current_node_id = states
                .get_index(state_index)
                .ok_or_else(|| {
                    FlowError::NotFound("current node has not produced a value".to_string())
                })?
                .0
                .clone();
            let current_node = match self.nodes.get(&current_node_id) {
                Some(n) => n,
                None => continue,
            };
            match current_node {
                FlowNode::Agent(agent) => {
                    // Flush a completed exit phase inline — no extra step boundary.
                    if let Some(Phase::Exit(value)) = states.phase().cloned() {
                        states.set_phase(Phase::Entry);
                        states.set_state(&agent.exit_name, value, Some(&current_node_id));
                        return Ok(FlowOut::Continue);
                    }

                    // Inspect the current continuation (if any).
                    if let Some(Phase::Continue(raw)) = states.phase().cloned() {
                        let cont: AgentContinuation =
                            serde_json::from_value(raw).map_err(|e| {
                                FlowError::from(AgentError::Deserialize(format!("agent continuation: {e}")))
                            })?;
                        match cont {
                            AgentContinuation::ToolsDispatched { .. } => {
                                // Should never be persisted — always converted inline. Treat as
                                // programmer error.
                                return Err(FlowError::from(AgentError::Llm(
                                    "ToolsDispatched persisted in frame — this is a bug".to_string(),
                                )));
                            }
                            AgentContinuation::Waiting { count, submitted } => {
                                if count > 0 {
                                    // Still awaiting tool completions.
                                    continue;
                                }
                                if let Some(v) = submitted {
                                    // submit tool fired — flush to exit state inline.
                                    states.set_phase(Phase::Entry);
                                    states.set_state(
                                        &agent.exit_name,
                                        v,
                                        Some(&current_node_id),
                                    );
                                    return Ok(FlowOut::Continue);
                                }
                                // All tools done, no submit — fall through to re-call LLM.
                            }
                        }
                    }

                    // Phase::Entry or Waiting{0, None}: call the LLM.
                    let initial_msg = if matches!(states.phase(), Some(Phase::Entry)) {
                        Some(
                            states
                                .get_state(&current_node_id)
                                .ok_or_else(|| {
                                    FlowError::NotFound(
                                        "initial state node has not produced a value".to_string(),
                                    )
                                })?
                                .to_string(),
                        )
                    } else {
                        None
                    };
                    let current_phase = states
                        .phase()
                        .ok_or_else(|| FlowError::Internal("handle_agent: empty frame stack".into()))?;
                    let next_phase = Self::handle_agent(
                        agent,
                        current_phase,
                        initial_msg.as_deref(),
                        factory,
                        history,
                        session_id,
                    )
                    .await?;
                    // ToolsDispatched: convert inline — never persist to the frame.
                    if let Phase::Continue(ref raw) = next_phase {
                        let cont: AgentContinuation =
                            serde_json::from_value(raw.clone()).map_err(|e| {
                                FlowError::from(AgentError::Deserialize(format!("agent continuation: {e}")))
                            })?;
                        if let AgentContinuation::ToolsDispatched { calls } = cont {
                            // Validate all tool names before pushing any states.
                            for call in &calls {
                                let key = format!("{}::{}", agent.name, call.name);
                                if !self.nodes.contains_key(&key) {
                                    return Err(FlowError::from(AgentError::ToolUnknown(format!(
                                        "unknown tool '{}' called by agent '{}'",
                                        call.name, agent.name
                                    ))));
                                }
                            }
                            for call in &calls {
                                let key = format!("{}::{}", agent.name, call.name);
                                let pv = serde_json::to_value(PendingCall {
                                    id: call.id.clone(),
                                    args: call.args.clone(),
                                })
                                .map_err(|e| {
                                    FlowError::from(AgentError::Serialize(format!("pending call: {e}")))
                                })?;
                                states.set_state(&key, pv, None);
                            }
                            let waiting =
                                serde_json::to_value(AgentContinuation::Waiting {
                                    count: calls.len(),
                                    submitted: None,
                                })
                                .map_err(|e| FlowError::from(AgentError::Serialize(e.to_string())))?;
                            states.set_phase(Phase::Continue(waiting));
                            return Ok(FlowOut::Continue);
                        }
                    }
                    // Phase::Exit: flush inline.
                    if let Phase::Exit(value) = next_phase {
                        states.set_phase(Phase::Entry);
                        states.set_state(&agent.exit_name, value, Some(&current_node_id));
                        return Ok(FlowOut::Continue);
                    }
                    // Should not reach here — handle_agent always returns Continue or Exit.
                    return Err(FlowError::from(AgentError::Llm("unexpected phase from handle_agent".to_string())));
                }
                FlowNode::Tool(info) => {
                    let raw_pending = states
                        .get_state(&info.name)
                        .ok_or_else(|| {
                            FlowError::NotFound(format!(
                                "tool state '{}' missing",
                                info.name
                            ))
                        })?
                        .clone();
                    let pending: PendingCall =
                        serde_json::from_value(raw_pending).map_err(|e| {
                            FlowError::from(AgentError::Deserialize(format!(
                                "pending call '{}': {e}",
                                info.name
                            )))
                        })?;
                    let call_id = pending.id;
                    let args = pending.args;

                    // Helper: read the current Waiting continuation from the agent's phase.
                    let read_waiting = |states: &FlowState| -> Result<(usize, Option<Value>), FlowError> {
                        let Some(Phase::Continue(raw)) = states.phase().cloned() else {
                            return Err(FlowError::from(AgentError::Llm(
                                "tool dispatched outside Phase::Continue".to_string(),
                            )));
                        };
                        let cont: AgentContinuation =
                            serde_json::from_value(raw).map_err(|e| {
                                FlowError::from(AgentError::Deserialize(format!(
                                    "agent continuation: {e}"
                                )))
                            })?;
                        let AgentContinuation::Waiting { count, submitted } = cont else {
                            return Err(FlowError::from(AgentError::Llm(
                                "tool dispatched in ToolsDispatched phase".to_string(),
                            )));
                        };
                        Ok((count, submitted))
                    };

                    // Inject an externally-supplied resume payload instead of executing.
                    if let Some((ref resume_id, ref resume_val)) = resumption {
                        if resume_id == &info.name {
                            history.push(
                                Message::tool_output(call_id, resume_val.to_string())
                                    .with_context(&info.agent_name, session_id),
                            );
                            states.clear_suspension();
                            let (count, submitted) = read_waiting(states)?;
                            let new_cont = serde_json::to_value(AgentContinuation::Waiting {
                                count: count.saturating_sub(1),
                                submitted,
                            })
                            .map_err(|e| FlowError::from(AgentError::Serialize(e.to_string())))?;
                            states.set_phase(Phase::Continue(new_cont));
                            states.remove_state(&info.name);
                            return Ok(FlowOut::Continue);
                        }
                    }

                    let result = info
                        .tool_box
                        .call_at_index(info.tool_index, &call_id, args, ctx.clone())
                        .await;
                    match result {
                        Ok(output) => {
                            history.push(
                                Message::tool_output(
                                    output.call.id,
                                    output.value.to_string(),
                                )
                                .with_context(&info.agent_name, session_id),
                            );
                            let (count, submitted) = read_waiting(states)?;
                            let new_cont = serde_json::to_value(AgentContinuation::Waiting {
                                count: count.saturating_sub(1),
                                submitted,
                            })
                            .map_err(|e| FlowError::from(AgentError::Serialize(e.to_string())))?;
                            states.set_phase(Phase::Continue(new_cont));
                            states.remove_state(&info.name);
                            return Ok(FlowOut::Continue);
                        }
                        Err(ToolError::Exit(value)) => {
                            history.push(
                                Message::tool_output(call_id, value.to_string())
                                    .with_context(&info.agent_name, session_id),
                            );
                            let (count, _) = read_waiting(states)?;
                            let new_cont = serde_json::to_value(AgentContinuation::Waiting {
                                count: count.saturating_sub(1),
                                submitted: Some(value),
                            })
                            .map_err(|e| FlowError::from(AgentError::Serialize(e.to_string())))?;
                            states.set_phase(Phase::Continue(new_cont));
                            states.remove_state(&info.name);
                            return Ok(FlowOut::Continue);
                        }
                        Err(ToolError::Suspend(value)) => {
                            // Phase stays Continue(Waiting) — tool slot stays in states.
                            states.suspend(&info.name);
                            return Ok(FlowOut::Suspend {
                                value,
                                tool_id: info.name.clone(),
                            });
                        }
                        Err(error) => {
                            return Err(FlowError::from(AgentError::ToolFailed(format!("Tool error: {error}"))));
                        }
                    }
                }
                FlowNode::Either(either) => {
                    Self::handle_either(either, ctx, states)?;
                    return Ok(FlowOut::Continue);
                }
                FlowNode::Fork(info) => {
                    Self::handle_fork(info, ctx, states)?;
                    return Ok(FlowOut::Continue);
                }
                FlowNode::Join(info) => {
                    if !self.can_join(&current_node_id, states) {
                        continue;
                    }
                    Self::handle_join(info, ctx, states)?;
                    return Ok(FlowOut::Continue);
                }
                FlowNode::Work(info) => {
                    Self::handle_work(info, ctx, states).await?;
                    return Ok(FlowOut::Continue);
                }
                FlowNode::Flow(inner) => {
                    let input_val = states.get_state(&inner.name).cloned().ok_or_else(|| {
                        FlowError::NotFound(format!(
                            "sub-flow '{}' has no input value",
                            inner.name
                        ))
                    })?;
                    states.remove_state(&inner.name);
                    states.push_frame(&inner.name, &inner.exit_name);
                    states.set_state(&inner.name, input_val, None);
                    return Ok(FlowOut::Continue);
                }
            }
        }
        self.resolve_done_or_deadlock(states)
    }

    /// Entry point for a step. Resolves the active inner graph from the frame stack,
    /// then delegates to [`step_inner`].
    async fn step(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        session_id: &str,
        resumption: Option<(String, Value)>,
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        match (states.suspension(), &resumption) {
            (Some(tool_id), None) => return Err(FlowError::ResumeRequired(tool_id.clone())),
            (None, Some((tool_id, _))) => {
                return Err(FlowError::UnexpectedResumption(tool_id.clone()));
            }
            (Some(expected), Some((actual, _))) if actual != expected => {
                return Err(FlowError::ResumeMismatchError(expected.clone()));
            }
            _ => {}
        }
        let active = walk_to_inner_graph(self, states);
        active
            .step_inner(factory, ctx, history, session_id, resumption, states)
            .await
    }
}

/// Walks the frame stack to find the currently active inner graph.
fn walk_to_inner_graph<'a>(root: &'a FlowGraph, states: &FlowState) -> &'a FlowGraph {
    let mut g = root;
    for depth in 1..states.frame_depth() {
        let Some(entry) = states.frame_entry_at(depth) else { break };
        if let Some(FlowNode::Flow(inner)) = g.nodes.get(entry) {
            g = inner;
        }
    }
    g
}

/// Validates per-node structural rules only — no entry or reachability checks.
/// Called by [`FlowBuilder::build`] before the entry is known.
fn validate_nodes(nodes: &HashMap<String, FlowNode>) -> Result<(), BuildError> {
    let mut problems: Vec<String> = Vec::new();
    let mut seen_join_groups: HashSet<String> = HashSet::new();
    for (key, node) in nodes {
        match node {
            FlowNode::Agent(info) => {
                if info.exit_name == info.name {
                    problems.push(format!(
                        "agent '{}': exit_name equals input name — node would overwrite its own input",
                        key
                    ));
                }
                if info.model.is_empty() {
                    problems.push(format!("agent '{}': model is empty", key));
                }
            }
            FlowNode::Work(info) => {
                if info.exit_name == info.name {
                    problems.push(format!(
                        "work '{}': exit_name equals input name — node would overwrite its own input",
                        key
                    ));
                }
            }
            FlowNode::Fork(info) => {
                if info.children.len() < 2 {
                    problems.push(format!(
                        "fork '{}': must have at least 2 children, found {}",
                        key,
                        info.children.len()
                    ));
                }
                let mut seen_children: HashSet<&str> = HashSet::new();
                for child in &info.children {
                    if !seen_children.insert(child.as_str()) {
                        problems.push(format!("fork '{}': duplicate child '{}'", key, child));
                    }
                    if !nodes.contains_key(child) {
                        problems.push(format!(
                            "fork '{}': child '{}' is not a registered node",
                            key, child
                        ));
                    }
                }
            }
            FlowNode::Join(info) => {
                let mut sorted_parents = info.parents.clone();
                sorted_parents.sort();
                let group_key = format!("{}→{}", sorted_parents.join("+"), info.target);
                if !seen_join_groups.insert(group_key) {
                    continue;
                }
                if info.parents.len() != 2 {
                    problems.push(format!(
                        "join (target '{}'): must have exactly 2 parents, found {}",
                        info.target,
                        info.parents.len()
                    ));
                }
                let mut seen_parents: HashSet<&str> = HashSet::new();
                for p in &info.parents {
                    if !seen_parents.insert(p.as_str()) {
                        problems.push(format!(
                            "join (target '{}'): duplicate parent '{}'",
                            info.target, p
                        ));
                    }
                    if p == &info.target {
                        problems.push(format!(
                            "join (target '{}'): target matches parent '{}'",
                            info.target, p
                        ));
                    }
                    if !nodes.contains_key(p.as_str()) {
                        problems.push(format!(
                            "join (target '{}'): parent '{}' is not a registered node",
                            info.target, p
                        ));
                    }
                }
            }
            FlowNode::Either(info) => {
                if info.left_name == info.right_name {
                    problems.push(format!(
                        "either '{}': both branches resolve to the same schema name '{}'",
                        key, info.left_name
                    ));
                }
            }
            FlowNode::Flow(inner) => {
                if inner.name == inner.exit_name {
                    problems.push(format!(
                        "flow '{}': exit_name equals input name — sub-flow output would overwrite its own input",
                        key
                    ));
                }
                if inner.name.is_empty() {
                    problems.push(format!("flow '{}': name is empty", key));
                }
                if inner.exit_name.is_empty() {
                    problems.push(format!("flow '{}': exit_name is empty", key));
                }
            }
            FlowNode::Tool(_) => {}
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(BuildError::Invalid(problems))
    }
}

/// Validates entry registration and graph reachability. Called by [`FlowGraph::with_entry`].
fn validate(nodes: &HashMap<String, FlowNode>, entry: &str) -> Result<(), BuildError> {
    let mut problems: Vec<String> = Vec::new();

    // --- Entry check ---
    if entry.is_empty() {
        problems.push("flow has no entry node".to_string());
    } else if !nodes.contains_key(entry) {
        problems.push(format!("entry '{}' is not a registered node", entry));
    }

    // --- Reachability (only when entry is valid) ---
    if !entry.is_empty() && nodes.contains_key(entry) {
        // Map each node to the set of keys it produces output into.
        // Tool nodes are excluded: they are dynamically reachable only when
        // an agent dispatches them, not via static graph edges.
        let successors: HashMap<&str, Vec<&str>> = nodes
            .iter()
            .filter(|(_, node)| !matches!(node, FlowNode::Tool(_)))
            .map(|(key, node)| {
                let succs: Vec<&str> = match node {
                    FlowNode::Agent(info) => vec![info.exit_name.as_str()],
                    FlowNode::Work(info) => vec![info.exit_name.as_str()],
                    FlowNode::Fork(info) => info.children.iter().map(String::as_str).collect(),
                    FlowNode::Join(info) => vec![info.target.as_str()],
                    FlowNode::Either(info) => {
                        vec![info.left_name.as_str(), info.right_name.as_str()]
                    }
                    FlowNode::Flow(inner) => vec![inner.exit_name.as_str()],
                    FlowNode::Tool(_) => vec![],
                };
                (key.as_str(), succs)
            })
            .collect();

        // Forward BFS: which registered nodes are reachable from entry?
        let mut reachable: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        reachable.insert(entry);
        queue.push_back(entry);
        while let Some(cur) = queue.pop_front() {
            if let Some(succs) = successors.get(cur) {
                for &s in succs {
                    if nodes.contains_key(s) && reachable.insert(s) {
                        queue.push_back(s);
                    }
                }
            }
        }
        for key in nodes.keys() {
            if matches!(nodes[key], FlowNode::Tool(_)) {
                continue; // tool nodes are dynamically reachable — skip static check
            }
            if !reachable.contains(key.as_str()) {
                problems.push(format!(
                    "node '{}': unreachable from entry '{}'",
                    key, entry
                ));
            }
        }

        // Build reverse adjacency for backward BFS.
        let mut predecessors: HashMap<&str, Vec<&str>> = HashMap::new();
        for (&key, succs) in &successors {
            for &s in succs {
                predecessors.entry(s).or_default().push(key);
            }
        }

        // Terminals: successor keys not present in the node map.
        let terminals: HashSet<&str> = successors
            .values()
            .flat_map(|v| v.iter().copied())
            .filter(|&s| !nodes.contains_key(s))
            .collect();

        // Backward BFS: which registered nodes can reach a terminal?
        let mut can_reach_terminal: HashSet<&str> = HashSet::new();
        let mut queue2: VecDeque<&str> = VecDeque::new();
        for &t in &terminals {
            if let Some(preds) = predecessors.get(t) {
                for &p in preds {
                    if can_reach_terminal.insert(p) {
                        queue2.push_back(p);
                    }
                }
            }
        }
        while let Some(cur) = queue2.pop_front() {
            if let Some(preds) = predecessors.get(cur) {
                for &p in preds {
                    if can_reach_terminal.insert(p) {
                        queue2.push_back(p);
                    }
                }
            }
        }
        for key in nodes.keys() {
            if matches!(nodes[key], FlowNode::Tool(_)) {
                continue; // tool nodes exit via submitted_value — no static terminal path
            }
            if !can_reach_terminal.contains(key.as_str()) {
                problems.push(format!(
                    "node '{}': has no path to any terminal — dead end",
                    key
                ));
            }
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(BuildError::Invalid(problems))
    }
}

fn collect_terminal_state_ids(nodes: &HashMap<String, FlowNode>) -> HashSet<String> {
    nodes
        .values()
        .flat_map(|node| match node {
            FlowNode::Agent(info) => vec![info.exit_name.clone()],
            FlowNode::Work(info) => vec![info.exit_name.clone()],
            FlowNode::Fork(info) => info.children.clone(),
            FlowNode::Join(info) => vec![info.target.clone()],
            FlowNode::Either(info) => vec![info.left_name.clone(), info.right_name.clone()],
            FlowNode::Flow(inner) => vec![inner.exit_name.clone()],
            FlowNode::Tool(_) => vec![], // no static terminal states
        })
        .filter(|state_id| !nodes.contains_key(state_id))
        .collect()
}

fn validate_runtime_output_contract(
    nodes: &HashMap<String, FlowNode>,
    expected_output: &str,
) -> Result<(), BuildError> {
    let terminals = collect_terminal_state_ids(nodes);
    let mut sorted_terminals: Vec<String> = terminals.iter().cloned().collect();
    sorted_terminals.sort();

    let mut problems = Vec::new();
    if sorted_terminals.len() != 1 {
        let found = if sorted_terminals.is_empty() {
            "<none>".to_string()
        } else {
            sorted_terminals.join(", ")
        };
        problems.push(format!(
            "flow must have exactly one terminal state id matching output '{}', found: {}",
            expected_output, found
        ));
    } else if sorted_terminals[0] != expected_output {
        problems.push(format!(
            "flow output '{}' does not match terminal state id '{}'",
            expected_output, sorted_terminals[0]
        ));
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(BuildError::Invalid(problems))
    }
}

pub struct FlowBuilder {
    flow: FlowGraph,
    errors: Vec<String>,
}

impl FlowBuilder {
    fn new() -> Self {
        Self {
            flow: FlowGraph::new(),
            errors: Vec::new(),
        }
    }

    /// Registers an agent node. The node name is derived from `A::node_id()`.
    pub fn agent<A: Agent>(mut self) -> Self {
        let name = A::node_id();
        if self.flow.nodes.contains_key(&name) {
            self.errors
                .push(format!("agent '{}': duplicate node key", name));
            return self;
        }
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let output_schema = match serde_json::to_value(schema_gen.root_schema_for::<A::Output>()) {
            Ok(v) => v,
            Err(e) => {
                self.errors
                    .push(format!("agent '{}' output schema: {e}", name));
                return self;
            }
        };
        let config = A::build();
        let tool_box = Arc::new(config.tool_box.with_agent::<A>());
        let agent_info = AgentInfo {
            name: name.clone(),
            tool_box: Arc::clone(&tool_box),
            preamble: config.preamble,
            model: config.model_url,
            exit_name: A::Output::schema_name(),
            output_schema,
        };
        self.flow.nodes.insert(name.clone(), FlowNode::Agent(agent_info));
        // Register a FlowNode::Tool for each tool in the toolbox (including the submit sentinel).
        for i in 0..tool_box.len() {
            let tool_name = format!("{}::{}", name, tool_box.name_at(i));
            self.flow.nodes.insert(
                tool_name.clone(),
                FlowNode::Tool(ToolInfo {
                    name: tool_name,
                    agent_name: name.clone(),
                    tool_index: i,
                    tool_box: Arc::clone(&tool_box),
                }),
            );
        }
        self
    }

    /// Registers a typed transition from `From`. The closure receives `From::Output`
    /// deserialized from stored JSON and a `&FlowContext`, and returns a [`StateNode`] — use [`node()`].
    pub fn either<From, A, B, H>(mut self, func: H) -> Self
    where
        From: Serialize + DeserializeOwned + JsonSchema,
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From, Context) -> Result<Either<A, B>, FlowError> + Send + Sync + 'static,
    {
        let from_id = From::schema_name();
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("either '{}': duplicate node key", from_id));
            return self;
        }
        let from_id_clone = from_id.clone();
        let shim: Box<dyn Fn(&Value, Context) -> Result<StateNode, FlowError> + Send + Sync> =
            Box::new(move |value: &Value, ctx: Context| {
                let typed: From = serde_json::from_value(value.clone()).map_err(|e| {
                    FlowError::DeserializeError(format!(
                        "transition from '{}': {e}",
                        from_id.clone()
                    ))
                })?;
                match func(typed, ctx)? {
                    Either::Left(a) => {
                        let node = node(a)?;
                        Ok(StateNode {
                            name: A::schema_name(),
                            value: node.value,
                        })
                    }
                    Either::Right(b) => {
                        let node = node(b)?;
                        Ok(StateNode {
                            name: B::schema_name(),
                            value: node.value,
                        })
                    }
                }
            });
        self.flow.nodes.insert(
            from_id_clone.clone(),
            FlowNode::Either(EitherInfo {
                name: from_id_clone.clone(),
                left_name: A::schema_name(),
                right_name: B::schema_name(),
                func: shim,
            }),
        );

        self
    }

    /// Registers a fork node at `From`. The closure receives the parent value and returns two
    /// child values that are placed into states for independent processing.
    pub fn fork<From, A, B, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From, Context) -> Result<(A, B), FlowError> + Send + Sync + 'static,
    {
        let from_id = From::schema_name();
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("fork '{}': duplicate node key", from_id));
            return self;
        }
        let from_id_clone = from_id.clone();
        let shim: Box<dyn Fn(&Value, Context) -> Result<Vec<StateNode>, FlowError> + Send + Sync> =
            Box::new(move |value: &Value, ctx: Context| {
                let typed: From = serde_json::from_value(value.clone()).map_err(|e| {
                    FlowError::DeserializeError(format!("fork from '{}': {e}", from_id))
                })?;
                let (a, b) = func(typed, ctx)?;
                Ok(vec![node(a)?, node(b)?])
            });
        self.flow.nodes.insert(
            from_id_clone.clone(),
            FlowNode::Fork(ForkInfo {
                name: from_id_clone,
                children: vec![A::schema_name(), B::schema_name()],
                func: shim,
            }),
        );
        self
    }

    /// Registers a join node that waits for both `A` and `B` states to be present,
    /// combines them into `Out`, and clears the parent states.
    pub fn join<A, B, Out, H>(mut self, func: H) -> Self
    where
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(A, B, Context) -> Result<Out, FlowError> + Send + Sync + 'static,
    {
        let a_id = A::schema_name();
        let b_id = B::schema_name();
        for id in [&a_id, &b_id] {
            if self.flow.nodes.contains_key(id) {
                self.errors
                    .push(format!("join: duplicate node key '{}'", id));
                return self;
            }
        }
        let target_id = Out::schema_name();
        let a_id_inner = a_id.clone();
        let b_id_inner = b_id.clone();
        let shim: Arc<dyn Fn(&[Value], Context) -> Result<StateNode, FlowError> + Send + Sync> =
            Arc::new(move |inputs: &[Value], ctx: Context| {
                let a: A = serde_json::from_value(inputs[0].clone()).map_err(|e| {
                    FlowError::DeserializeError(format!("join input '{}': {e}", a_id_inner))
                })?;
                let b: B = serde_json::from_value(inputs[1].clone()).map_err(|e| {
                    FlowError::DeserializeError(format!("join input '{}': {e}", b_id_inner))
                })?;
                node(func(a, b, ctx)?)
            });
        self.flow.nodes.insert(
            a_id.clone(),
            FlowNode::Join(JoinInfo {
                parents: vec![a_id.clone(), b_id.clone()],
                target: target_id.clone(),
                func: Arc::clone(&shim),
            }),
        );
        self.flow.nodes.insert(
            b_id.clone(),
            FlowNode::Join(JoinInfo {
                parents: vec![a_id, b_id],
                target: target_id,
                func: shim,
            }),
        );
        self
    }

    /// Embeds sub-flow `F` as a [`FlowNode::Flow`] node.
    ///
    /// The node key is `F::schema_name()` (the input type). When the flow engine
    /// encounters this node it pushes a new execution frame, seeds it with the input
    /// value, and resumes the inner graph until it produces `F::Output`. The output
    /// is then written to the parent frame under `F::Output::schema_name()`.
    pub fn flow<F: Flow>(mut self) -> Self {
        let name = F::schema_name();
        let exit_name = F::Output::schema_name();
        if self.flow.nodes.contains_key(&name) {
            self.errors
                .push(format!("flow '{}': duplicate node key", name));
            return self;
        }
        let mut inner = match F::build() {
            Ok(g) => g,
            Err(e) => {
                self.errors.push(format!("flow '{}': {e}", name));
                return self;
            }
        };
        inner.name = name.clone();
        inner.exit_name = exit_name;
        self.flow.nodes.insert(name, FlowNode::Flow(Box::new(inner)));
        self
    }

    /// Registers a work node at `From`. The async closure transforms the input value into
    /// `Out` without LLM involvement.
    pub fn work<From, Out, Fut, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        Fut: std::future::Future<Output = Result<Out, FlowError>> + Send + 'static,
        H: Fn(From, Context) -> Fut + Send + Sync + 'static,
    {
        let from_id = From::schema_name();
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("work '{}': duplicate node key", from_id));
            return self;
        }
        let from_id_clone = from_id.clone();
        let exit_id = Out::schema_name();
        let shim: Box<
            dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync,
        > = Box::new(move |value: &Value, ctx: Context| {
            let typed: From = match serde_json::from_value(value.clone()) {
                Ok(v) => v,
                Err(e) => {
                    let err = FlowError::DeserializeError(format!("work from '{}': {e}", from_id));
                    return Box::pin(async move { Err(err) });
                }
            };
            let fut = func(typed, ctx);
            Box::pin(async move {
                let out = fut.await?;
                serde_json::to_value(&out)
                    .map_err(|e| FlowError::SerializeError(format!("work output: {e}")))
            })
        });
        self.flow.nodes.insert(
            from_id_clone.clone(),
            FlowNode::Work(WorkInfo {
                name: from_id_clone,
                exit_name: exit_id,
                func: shim,
            }),
        );
        self
    }

    /// Validates the graph structure and returns the [`FlowGraph`].
    ///
    /// The entry node is not set here — it is injected later by [`FlowRuntime`] via
    /// the crate-private `FlowGraph::with_entry`. Structural rules (duplicate nodes,
    /// bad model URL, same-type work, fork/join shape) are checked immediately.
    pub fn build(self) -> Result<FlowGraph, FlowError> {
        if !self.errors.is_empty() {
            return Err(BuildError::Invalid(self.errors).into());
        }
        validate_nodes(&self.flow.nodes)?;
        Ok(self.flow)
    }
}

pub struct FlowRuntime<I: Flow> {
    state: FlowState,
    graph: FlowGraph,
    history: FlowHistory,
    session_id: String,
    factory: Arc<dyn ClientFactory>,
    _marker: std::marker::PhantomData<I>,
}

impl<I: Flow> FlowRuntime<I> {
    fn build_graph() -> Result<FlowGraph, FlowError> {
        let graph = I::build()?.with_entry(I::node_id())?;
        validate_runtime_output_contract(&graph.nodes, &I::Output::schema_name())?;
        Ok(graph)
    }

    pub fn new(flow: I) -> Result<Self, FlowError> {
        let graph = Self::build_graph()?;
        let value = serde_json::to_value(&flow).map_err(|e| {
            FlowError::SerializeError(format!("start node '{}': {e}", I::node_id()))
        })?;
        let mut state = FlowState::new();
        state.push_frame("", "");
        state.set_state(&I::node_id(), value, None);
        let mut history = FlowHistory::new(None);
        history.push(Message::user(format!("Starting flow: {}", I::node_id())));
        Ok(Self {
            state,
            graph,
            history,
            session_id: new_session_id(),
            factory: Arc::new(DefaultClientFactory),
            _marker: std::marker::PhantomData,
        })
    }

    /// Overrides the default [`ClientFactory`]. Useful for testing or selecting a provider.
    pub fn with_factory(mut self, factory: impl ClientFactory + 'static) -> Self {
        self.factory = Arc::new(factory);
        self
    }

    /// Replaces the conversation history used by agent nodes.
    ///
    /// Call after [`FlowRuntime::from_snapshot`] to restore the LLM context from a
    /// previously persisted [`FlowHistory`].
    pub fn with_history(mut self, history: FlowHistory) -> Self {
        self.history = history;
        self
    }

    pub async fn next(&mut self, ctx: Context) -> Result<RunOut<I::Output>, FlowError> {
        let factory = Arc::clone(&self.factory);
        let out = self
            .graph
            .next(factory.as_ref(), ctx, &mut self.history, &self.session_id, &mut self.state)
            .await?;
        Self::map_out(out)
    }

    pub async fn resume(
        &mut self,
        ctx: Context,
        resumption: (String, Value),
    ) -> Result<RunOut<I::Output>, FlowError> {
        let factory = Arc::clone(&self.factory);
        let out = self
            .graph
            .resume(
                factory.as_ref(),
                ctx,
                &mut self.history,
                &self.session_id,
                resumption,
                &mut self.state,
            )
            .await?;
        Self::map_out(out)
    }

    fn map_out(out: FlowOut) -> Result<RunOut<I::Output>, FlowError> {
        match out {
            FlowOut::Continue => Ok(RunOut::Continue),
            FlowOut::Done(value) => {
                let output = serde_json::from_value(value)
                    .map_err(|e| FlowError::DeserializeError(format!("flow output: {e}")))?;
                Ok(RunOut::Done(output))
            }
            FlowOut::Suspend { value, tool_id } => Ok(RunOut::Suspend { value, tool_id }),
        }
    }

    /// Captures the current execution state as an opaque [`FlowSnapshot`].
    ///
    /// The snapshot can be serialized and persisted. It does not include conversation
    /// history — manage that separately and re-attach via [`FlowRuntime::with_history`].
    pub fn snapshot(&self) -> FlowSnapshot {
        FlowSnapshot {
            state: self.state.clone(),
        }
    }

    /// Reconstructs a runtime from a previously captured [`FlowSnapshot`].
    ///
    /// The flow graph is rebuilt from `I::build()` — closures are never persisted.
    /// A fresh history is created; chain `.with_history(saved_history)` to restore
    /// the full LLM conversation context.
    pub fn from_snapshot(snapshot: FlowSnapshot) -> Result<Self, FlowError> {
        let graph = Self::build_graph()?;
        let mut history = FlowHistory::new(None);
        history.push(Message::user(format!("Starting flow: {}", I::node_id())));
        Ok(Self {
            state: snapshot.state,
            history,
            graph,
            session_id: new_session_id(),
            factory: Arc::new(DefaultClientFactory),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<I: Flow> std::fmt::Debug for FlowRuntime<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowRuntime").finish_non_exhaustive()
    }
}

/// Opaque snapshot of a flow's execution state at a point in time.
///
/// Obtained via [`FlowRuntime::snapshot`]; restored via [`FlowRuntime::from_snapshot`].
/// Safe to serialize with any `serde` format (JSON, MessagePack, etc.).
///
/// Does **not** include conversation history — manage [`FlowHistory`] separately and
/// re-attach with [`FlowRuntime::with_history`] after reconstructing the runtime.
#[derive(Serialize, Deserialize)]
pub struct FlowSnapshot {
    state: FlowState,
}
