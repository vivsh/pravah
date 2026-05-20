use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::history::FlowHistory;
use super::nary::{MergeInputs, SplitOutputs};
use crate::flows::{AgentConfig, NodeId};
use crate::flows::errors::{AgentError, BuildError, FlowError};
use crate::flows::interner::Interner;
use crate::flows::state::{AgentContinuation, AgentState, Callable, FlowState, WaitingCall};
use crate::flows::validation::{validate, validate_nodes};
use crate::{
    clients::{
        ClientFactory, ClientOptions, ClientOutput, Message, Role, ToolChoice,
        materialize_messages,
    },
    commons::{Agent, make_agent_message},
    context::Context,
    tools::{SuspendedValue, ToolDefinition, ToolError, ToolOutput},
    tools::base::pascal_to_snake,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct StateNode {
    name: String,
    value: serde_json::Value,
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
    /// Maps tool call names to their state entry and exit slots.
    pub(crate) tool_lookup: HashMap<String, (NodeId, NodeId)>,
    pub(crate) keep_alive: bool,
    /// When set together with `max_tool_calls`, an error result with this
    /// message is returned to the LLM if a tool exceeds its call budget.
    pub(crate) loop_break_message: Option<String>,
    /// Per-tool call limit used together with `loop_break_message`.
    pub(crate) max_tool_calls: Option<usize>,
}

pub(crate) struct EitherInfo {
    pub(crate) entry: NodeId,
    pub(crate) left_name: NodeId,
    pub(crate) right_name: NodeId,
    func: Box<dyn Fn(&Value) -> Result<(NodeId, Value), FlowError> + Send + Sync>,
}

pub(crate) struct ForkInfo {
    pub(crate) name: NodeId,
    pub(crate) children: Vec<NodeId>,
    func: Box<dyn Fn(&Value) -> Result<Vec<StateNode>, FlowError> + Send + Sync>,
}

pub(crate) struct JoinInfo {
    pub(crate) parents: Vec<NodeId>,
    pub(crate) target: NodeId,
    func: Arc<dyn Fn(&[Value]) -> Result<StateNode, FlowError> + Send + Sync>,
}

pub(crate) struct WorkInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    func:
        Box<dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync>,
}

/// Pure synchronous transform node.
/// Use `work` if the step needs I/O, context, or an error path.
pub(crate) struct MapInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    func: Box<dyn Fn(&Value) -> Result<Value, FlowError> + Send + Sync>,
}

/// Flow-level suspend node.
/// When `entry` is present the runtime returns a [`SuspendedValue`] and waits for `resume()`.
pub(crate) struct SuspendInfo {
    pub(crate) entry: NodeId,
    pub(crate) exit: NodeId,
    pub(crate) output_type: String,
    /// Converts the stored input into the erased suspended value returned by the runtime.
    deserialize: Box<dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync>,
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
    /// Pure synchronous transform.
    Map(MapInfo),
    /// Flow-level suspend point.
    Suspend(SuspendInfo),
    /// Embedded child flow.
    Flow(Arc<FlowGraph>),
}


/// Step result returned by the runtime.
pub enum FlowStep<T = serde_json::Value> {
    Continue,
    Done(T),
    /// Flow paused and is waiting for an external value.
    Suspend(SuspendedValue),
}

impl<T: std::fmt::Debug> std::fmt::Debug for FlowStep<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlowStep::Continue => write!(f, "Continue"),
            FlowStep::Done(v) => f.debug_tuple("Done").field(v).finish(),
            FlowStep::Suspend(s) => f.debug_tuple("Suspend").field(s).finish(),
        }
    }
}

pub trait Flow: 'static + JsonSchema + Serialize + DeserializeOwned + Send + Sync {
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;

    fn build(builder: FlowBuilder) -> FlowBuilder;

    fn node_id() -> String {
        Self::schema_name()
    }
}

/// Executable flow graph.
/// History side effects belong only in `handle_child_agent` and `dispatch_agent`.
/// Every other handler should only move state.
/// Suspension is global: resume writes the waiting node's output slot and the rest of the graph continues.
pub struct FlowGraph {
    pub(crate) nodes: HashMap<NodeId, FlowNode>,

    pub(crate) entry: NodeId,
    pub(crate) exit: NodeId,

    pub(crate) parent_exit: Option<NodeId>,

    /// Parent slot that feeds this graph when it runs as a child flow.
    pub(crate) parent_entry: Option<NodeId>,

    /// Graph-local mapping from names to `NodeId`s.
    pub(crate) interner: Interner,
    pub(crate) callable_index: usize,
}

impl FlowGraph {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry: NodeId(0),
            parent_entry: None,
            parent_exit: None,
            exit: NodeId(0),
            interner: Interner::new(),
            callable_index: 0,
        }
    }

    pub fn builder() -> FlowBuilder {
        FlowBuilder::new()
    }

    pub fn from_flow<F: Flow>() -> Result<Self, FlowError> {
        let entry = F::node_id();
        let exit = F::Output::schema_name();
        F::build(FlowBuilder::new()).build()?.with_entry(entry, exit)
    }

    /// Returns true when every parent for this join is present in state.
    fn can_join(&self, node_id: NodeId, state: &FlowState) -> bool {
        if let Some(FlowNode::Join(join_info)) = self.nodes.get(&node_id) {
            join_info.parents.iter().all(|&p| state.contains_state(p))
        } else {
            false
        }
    }

    async fn handle_work(
        node: &WorkInfo,
        ctx: Context,
        states: &mut FlowState,
    ) -> Result<(), FlowError> {
        let state = states.get_state(node.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "work node '{}' has not produced a value",
                node.name.0
            ))
        })?;
        let output = (node.func)(&state, ctx).await?;
        if !states.set_state(node.exit_name, output, Some(node.name)) {
            return Err(FlowError::Internal {
                handler: "handle_work",
                detail: "frame stack empty on set_state".into(),
            });
        }
        Ok(())
    }

    fn handle_fork(node: &ForkInfo, states: &mut FlowState) -> Result<(), FlowError> {
        let state = states.get_state(node.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "fork parent '{}' has not produced a value",
                node.name.0
            ))
        })?;

        let children = (node.func)(state)?;
        if children.len() != node.children.len() {
            return Err(BuildError::ChildCountMismatch(format!(
                "fork node '{}' produced {} child states but has {} child nodes",
                node.name.0,
                children.len(),
                node.children.len()
            ))
            .into());
        }
        for (child_node, &child_id) in children.iter().zip(&node.children) {
            if !states.set_state(child_id, child_node.value.clone(), None) {
                return Err(FlowError::Internal {
                    handler: "handle_fork",
                    detail: "frame stack empty on set_state".into(),
                });
            }
        }
        if !states.remove_state(node.name) {
            return Err(FlowError::Internal {
                handler: "handle_fork",
                detail: "frame stack empty on remove".into(),
            });
        }

        Ok(())
    }

    fn handle_join(node: &JoinInfo, states: &mut FlowState) -> Result<(), FlowError> {
        let mut inputs = Vec::with_capacity(node.parents.len());
        for &p in &node.parents {
            let value = states.get_state(p).ok_or_else(|| {
                FlowError::NotFound(format!("join parent '{}' has not produced a value", p.0))
            })?;
            inputs.push(value.clone());
        }
        let output = (node.func)(&inputs)?;
        if !states.set_state(node.target, output.value, None) {
            return Err(FlowError::Internal {
                handler: "handle_join",
                detail: "frame stack empty on set_state".into(),
            });
        }

        for &p in &node.parents {
            if !states.remove_state(p) {
                return Err(FlowError::Internal {
                    handler: "handle_join",
                    detail: "frame stack empty on remove".into(),
                });
            }
        }

        Ok(())
    }

    fn handle_either(
        either: &EitherInfo,
        states: &mut FlowState,
    ) -> Result<(), FlowError> {
        let state = states.get_state(either.entry).ok_or_else(|| {
            FlowError::NotFound(format!(
                "either parent '{}' has not produced a value",
                either.entry.0
            ))
        })?;
        let (out_id, out_val) = (either.func)(&state)?;
        if !states.set_state(out_id, out_val, Some(either.entry)) {
            return Err(FlowError::Internal {
                handler: "handle_either",
                detail: "frame stack empty on set_state".into(),
            });
        }
        Ok(())
    }

    fn handle_map(node: &MapInfo, states: &mut FlowState) -> Result<(), FlowError> {
        let state = states.get_state(node.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "map node '{}' has not produced a value",
                node.name.0
            ))
        })?;
        let output = (node.func)(&state)?;
        if !states.set_state(node.exit_name, output, Some(node.name)) {
            return Err(FlowError::Internal {
                handler: "handle_map",
                detail: "frame stack empty on set_state".into(),
            });
        }
        Ok(())
    }

    fn handle_suspend(info: &SuspendInfo, states: &mut FlowState) -> Result<FlowStep, FlowError> {
        let value = states
            .get_state(info.entry)
            .ok_or_else(|| {
                FlowError::NotFound(format!(
                    "suspend node '{}': input state missing",
                    info.entry.0
                ))
            })?
            .clone();
        let sv = (info.deserialize)(value).map_err(FlowError::Deserialize)?;
        states.suspend(info.entry, info.exit, info.output_type.clone());
        Ok(FlowStep::Suspend(sv))
    }

    /// Binds the entry and exit ids and runs full graph validation.
    fn with_entry(mut self, entry: String, exit: String) -> Result<Self, FlowError> {
        let entry_id = self.interner.intern(&entry);
        let exit_id = self.interner.intern(&exit);
        self.entry = entry_id;
        self.exit = exit_id;
        validate(&self.nodes, entry_id, &self)?;
        Ok(self)
    }

    pub(crate) async fn next(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        self.step(factory, ctx, history, None, states)
            .await
    }

    pub(crate) async fn resume(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        resumption: Value,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        self.step(factory, ctx, history, Some(resumption), states)
            .await
    }

    async fn handle_agent(
        node: &AgentInfo,
        flow: &FlowGraph,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        // Clone the current agent state so we can release the borrow on `states`
        // before mutating it later.
        let current = states.get_agent_state(node.id).cloned();

        match current {
            // ── First visit: record the user turn, then dispatch ──────────────
            None => {
                let session_id =
                    states.get_or_init_session_id(node.id, node.keep_alive);
                let input = states
                    .get_state(node.id)
                    .cloned()
                    .ok_or_else(|| {
                        FlowError::NotFound(format!(
                            "agent '{}': input state missing",
                            flow.interner.name_of(node.id)
                        ))
                    })?;
                let agent_name = flow.interner.name_of(node.id);
                let message = (node.make_message)(input, &ctx)?;
                history.push(&session_id, agent_name, message);
                states.init_agent_state(node.id, session_id.clone());
                Self::dispatch_agent(node, flow, factory, ctx, history, states, &session_id)
                    .await
            }

            // ── Structured-output exit (agent produced output via LLM) ────────
            Some(AgentState {
                continuation: AgentContinuation::Exit(value),
                ..
            }) => {
                states.remove_agent_state(node.id);
                if !states.set_state(node.exit, value, Some(node.id)) {
                    return Err(FlowError::Internal {
                        handler: "handle_agent",
                        detail: "Exit: frame stack empty on set_state".into(),
                    });
                }
                Ok(FlowStep::Continue)
            }

            // ── Ready to call the LLM ─────────────────────────────────────────
            Some(AgentState {
                ref session_id,
                continuation: AgentContinuation::Dispatch,
                ..
            }) => {
                let session_id = session_id.clone();
                Self::dispatch_agent(node, flow, factory, ctx, history, states, &session_id)
                    .await
            }

            // ── Waiting for work nodes to finish ─────────────────────────────
            Some(AgentState {
                ref session_id,
                continuation: AgentContinuation::PendingTool {
                    ref active,
                    ref waiting,
                },
                ..
            }) => {
                let session_id = session_id.clone();
                let agent_name = flow.interner.name_of(node.id);

                // Clone so we can release the borrow while mutating states.
                let mut active = active.clone();
                let mut waiting = waiting.clone();

                // Collect finished tool exits.
                let mut completions: Vec<(NodeId, String)> = Vec::new(); // (exit_id, call_id)
                for (&exit_id, (call_id, _)) in &active {
                    if states.contains_state(exit_id) {
                        completions.push((exit_id, call_id.clone()));
                    }
                }

                let completed_count = completions.len();
                for (exit_id, call_id) in completions {
                    let value = states.take_state(exit_id).unwrap_or_default();
                    let tool_info = node.tools.iter().find(|t| t.exit_id == exit_id)
                        .ok_or_else(|| FlowError::Internal {
                            handler: "handle_agent",
                            detail: format!("no ToolInfo for exit_id {:?}", exit_id),
                        })?;
                    let mut msg = (tool_info.to_message)(value).map_err(FlowError::Tool)?;
                    msg.role = Role::Tool { call_id };
                    history.push(&session_id, agent_name, msg);

                    active.remove(&exit_id);
                    if let Some(queue) = waiting.get_mut(&exit_id) {
                        if let Some(next) = queue.pop_front() {
                            if !states.set_state(next.entry_id, next.args, None) {
                                return Err(FlowError::Internal {
                                    handler: "handle_agent",
                                    detail: "PendingTool promote: frame stack empty".into(),
                                });
                            }
                            active.insert(exit_id, (next.call_id, next.call_name));
                        }
                    }
                    waiting.retain(|_, q| !q.is_empty());
                }

                tracing::debug!(
                    agent = %agent_name,
                    completed = completed_count,
                    active = active.len(),
                    "tool results collected"
                );

                if active.is_empty() {
                    // All tools done → re-dispatch.
                    if let Some(s) = states.get_agent_state_mut(node.id) {
                        s.continuation = AgentContinuation::Dispatch;
                    }
                    states.reinsert_state(node.id);
                    Self::dispatch_agent(
                        node, flow, factory, ctx, history, states, &session_id,
                    )
                    .await
                } else {
                    // Still waiting for some tools.
                    if let Some(s) = states.get_agent_state_mut(node.id) {
                        s.continuation = AgentContinuation::PendingTool { active, waiting };
                    }
                    states.reinsert_state(node.id);
                    Ok(FlowStep::Continue)
                }
            }
        }
    }

    /// Runs one model turn.
    /// Structured output writes `node.exit` and marks the agent complete.
    /// Tool calls are recorded in history and stored as a pending continuation.
    async fn dispatch_agent(
        node: &AgentInfo,
        flow: &FlowGraph,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        states: &mut FlowState,
        session_id: &str,
    ) -> Result<FlowStep, FlowError> {
        let agent_name = flow.interner.name_of(node.id).to_string();

        let defs: Vec<ToolDefinition> = node.tools.iter().map(|t| t.definition.clone()).collect();
        let tool_choice = if defs.is_empty() { ToolChoice::Disabled } else { ToolChoice::Required };
        tracing::info!(
            agent = %agent_name,
            model = %node.model,
            tools = defs.len(),
            session_id = %session_id,
            "LLM dispatch"
        );

        let has_prior_history = !history.session_entries(session_id).is_empty();
        let options = ClientOptions::default()
            .with_input_schema(node.input_schema.clone())
            .with_tools(defs)
            .with_tool_choice(tool_choice)
            .with_output_schema(node.output_schema.clone())
            .with_name(agent_name.clone());
        let options = if has_prior_history { options } else { options.with_preamble(node.preamble.clone()) };

        let client = factory
            .create(&node.model, options)
            .map_err(|e| {
                tracing::error!(agent = %agent_name, error = %e, "LLM client creation failed");
                AgentError::LlmFailed {
                    agent: agent_name.clone(),
                    reason: e.to_string(),
                }
            })?;

        history.validate_for_session(session_id).map_err(|e| {
            tracing::error!(agent = %agent_name, error = %e, "session history validation failed");
            AgentError::LlmFailed {
                agent: agent_name.clone(),
                reason: e.to_string(),
            }
        })?;

        let session_msgs = materialize_messages(&history.for_session(session_id), &ctx)
            .await
            .map_err(|e| {
                tracing::error!(agent = %agent_name, error = %e, "message materialization failed");
                AgentError::LlmFailed {
                    agent: agent_name.clone(),
                    reason: e.to_string(),
                }
            })?;
        tracing::debug!(
            agent = %agent_name,
            session_id = %session_id,
            message_count = session_msgs.len(),
            last_message = ?session_msgs.last(),
            "LLM request history"
        );

        let response =
            client
                .execute(&session_msgs)
                .await
                .map_err(|e| {
                    tracing::error!(agent = %agent_name, error = %e, "LLM execution failed");
                    AgentError::LlmFailed {
                        agent: agent_name.clone(),
                        reason: e.to_string(),
                    }
                })?;

        tracing::debug!(
            agent = %agent_name,
            session_id = %session_id,
            response = ?response,
            "LLM response"
        );

        let usage = response.usage;

        match response.output {
            ClientOutput::Output(val) => {
                tracing::info!(agent = %agent_name, has_usage = usage.is_some(), "agent produced output");
                let content = serde_json::to_string(&val).map_err(AgentError::Serialize)?;
                let msg = if let Some(usage) = usage {
                    Message::assistant(content).with_usage(usage)
                } else {
                    Message::assistant(content)
                };
                history.push(session_id, &agent_name, msg);

                // Agent is done; write its output and clear agent state.
                states.remove_agent_state(node.id);
                if !states.set_state(node.exit, val, Some(node.id)) {
                    return Err(FlowError::Internal {
                        handler: "dispatch_agent",
                        detail: "Output: frame stack empty on set_state".into(),
                    });
                }

                Ok(FlowStep::Continue)
            }

            ClientOutput::ToolCalls { thought, calls } => {
                tracing::debug!(agent = %agent_name, tool_calls = calls.len(), "agent issued tool calls");
                let atc_msg = Message {
                    role: Role::AssistantToolCalls {
                        calls: calls.clone(),
                    },
                    content: thought.unwrap_or_default(),
                    attachments: Vec::new(),
                    usage,
                };
                history.push(session_id, &agent_name, atc_msg);

                // Retrieve existing call_counts (survives Dispatch transitions).
                let existing_counts: HashMap<String, usize> = states
                    .get_agent_state(node.id)
                    .map(|s| s.call_counts.clone())
                    .unwrap_or_default();

                let mut seen_call_ids = HashSet::new();
                let mut active: HashMap<NodeId, (String, String)> =
                    HashMap::with_capacity(calls.len());
                let mut waiting: HashMap<NodeId, VecDeque<WaitingCall>> = HashMap::new();
                let mut new_counts = existing_counts;

                for call in calls {
                    if !seen_call_ids.insert(call.id.clone()) {
                        return Err(AgentError::DuplicateToolCall {
                            agent: agent_name.clone(),
                            tool: call.name.clone(),
                        }
                        .into());
                    }

                    let Some(&(entry_id, exit_id)) = node.tool_lookup.get(&call.name) else {
                        tracing::warn!(
                            agent = %agent_name,
                            tool = %call.name,
                            call_id = %call.id,
                            "unknown tool; returning error result to LLM"
                        );
                        history.push(
                            session_id,
                            &agent_name,
                            Message::tool_output(
                                call.id.clone(),
                                format!(r#"{{"error":"unknown tool '{}'""}}"#, call.name),
                            ),
                        );
                        continue;
                    };

                    // Check per-tool loop limit.
                    let count = new_counts.entry(call.name.clone()).or_insert(0);
                    *count += 1;
                    if let (Some(max), Some(msg)) =
                        (node.max_tool_calls, &node.loop_break_message)
                    {
                        if *count > max {
                            tracing::warn!(
                                agent = %agent_name,
                                tool = %call.name,
                                count = *count,
                                max,
                                "tool call loop limit exceeded; returning error"
                            );
                            history.push(
                                session_id,
                                &agent_name,
                                Message::tool_output(
                                    call.id.clone(),
                                    format!(r#"{{"error":"{msg}"}}"#),
                                ),
                            );
                            continue;
                        }
                    }

                    if active.contains_key(&exit_id) {
                        // The submit sentinel (exit == node.exit) may only appear once.
                        if exit_id == node.exit {
                            return Err(AgentError::DuplicateToolCall {
                                agent: agent_name.clone(),
                                tool: call.name,
                            }
                            .into());
                        }
                        // Queue additional calls for this tool until the slot is free.
                        waiting.entry(exit_id).or_default().push_back(WaitingCall {
                            call_id: call.id,
                            args: call.args,
                            call_name: call.name,
                            entry_id,
                        });
                    } else {
                        if !states.set_state(entry_id, call.args, None) {
                            return Err(FlowError::Internal {
                                handler: "dispatch_agent",
                                detail: "ToolCalls: frame stack empty on set_state".into(),
                            });
                        }
                        active.insert(exit_id, (call.id, call.name));
                    }
                }

                let needs_pending = !active.is_empty();
                let continuation = if needs_pending {
                    AgentContinuation::PendingTool { active, waiting }
                } else {
                    AgentContinuation::Dispatch
                };

                // Update agent state with new continuation and call counts.
                if let Some(s) = states.get_agent_state_mut(node.id) {
                    s.continuation = continuation;
                    s.call_counts = new_counts;
                }

                if needs_pending {
                    states.reinsert_state(node.id);
                }

                Ok(FlowStep::Continue)
            }
        }
    }

    /// Dispatches one executable node from the active frame.
    async fn step_inner(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let total_states = states.len();
        for state_index in 0..total_states {
            let current_node_id = states
                .get_index(state_index)
                .ok_or_else(|| {
                    FlowError::NotFound(
                        "step_inner: current node has not produced a value".to_string(),
                    )
                })?
                .0;
            let current_node = match self.nodes.get(&current_node_id) {
                Some(n) => n,
                None => continue,
            };
            let node_name = self.interner.name_of(current_node_id);
            match current_node {
                FlowNode::Agent(agent) => {
                    tracing::debug!(node = %node_name, kind = "agent", "dispatching node");
                    return Self::handle_agent(agent, self, factory, ctx, history, states).await;
                }
                FlowNode::Either(either) => {
                    tracing::debug!(node = %node_name, kind = "either", "dispatching node");
                    Self::handle_either(either, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Fork(info) => {
                    tracing::debug!(node = %node_name, kind = "fork", "dispatching node");
                    Self::handle_fork(info, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Join(info) => {
                    if !self.can_join(current_node_id, states) {
                        tracing::trace!(node = %node_name, kind = "join", "join waiting on parents");
                        continue;
                    }
                    tracing::debug!(node = %node_name, kind = "join", "dispatching node");
                    Self::handle_join(info, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Work(info) => {
                    tracing::debug!(node = %node_name, kind = "work", "dispatching node");
                    Self::handle_work(info, ctx, states).await?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Map(info) => {
                    tracing::debug!(node = %node_name, kind = "map", "dispatching node");
                    Self::handle_map(info, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Suspend(info) => {
                    tracing::debug!(node = %node_name, kind = "suspend", "dispatching node");
                    return Self::handle_suspend(info, states);
                }
                FlowNode::Flow(inner) => {
                    tracing::debug!(node = %node_name, kind = "flow", "entering sub-flow");
                    let parent_exit = inner.parent_exit.ok_or_else(|| FlowError::Internal {
                        handler: "step_inner",
                        detail: "inner flow missing parent exit".into(),
                    })?;

                    let callable = Callable {
                        parent_entry: current_node_id,
                        parent_exit,
                        exit: inner.exit,
                        entry: inner.entry,
                        index: inner.callable_index,
                        keep_alive: false,
                    };

                    states.call_enter(callable);

                    return Ok(FlowStep::Continue);
                }
            }
        }

        Ok(FlowStep::Continue)
    }

    async fn step(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        mut resumption: Option<Value>,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        match (states.suspension(), &resumption) {
            (Some(_), None) => return Err(FlowError::ResumeRequired),
            (None, Some(_)) => {
                return Err(FlowError::UnexpectedResumption);
            }
            _ => {}
        }
        if let Some(resumption) = resumption.take() {
            if !states.resume(resumption) {
                return Err(FlowError::Internal {
                    handler: "step",
                    detail: "no active suspension or empty frame stack".into(),
                });
            }
        }
        let result = self
            .step_inner(factory, ctx, history, states)
            .await?;
        match result {
            FlowStep::Continue => {
                if let Some(v) = states.call_exit() {
                    return Ok(FlowStep::Done(v));
                }
                Ok(FlowStep::Continue)
            }
            other => Ok(other), // Suspend leaves the frame in place.
        }
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

    /// Registers an agent node keyed by `A::node_id()`.
    pub fn agent<A: Agent>(mut self) -> Self {
        let name_str = A::node_id();
        let name = self.flow.interner.intern(&name_str);
        if self.flow.nodes.contains_key(&name) {
            self.errors
                .push(format!("agent '{}': duplicate node key", name_str));
            return self;
        }
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let input_schema = match serde_json::to_value(schema_gen.root_schema_for::<A>()) {
            Ok(v) => v,
            Err(e) => {
                self.errors
                    .push(format!("agent '{}' input schema: {e}", name_str));
                return self;
            }
        };
        let output_schema = match serde_json::to_value(schema_gen.root_schema_for::<A::Output>()) {
            Ok(v) => v,
            Err(e) => {
                self.errors
                    .push(format!("agent '{}' output schema: {e}", name_str));
                return self;
            }
        };
        let config = A::build();
        let output_str = A::Output::schema_name();
        let output_id = self.flow.interner.intern(&output_str);
        let agent_info = AgentInfo {
            id: name,
            tools: Vec::new(),
            make_message: make_agent_message::<A>,
            preamble: config.preamble,
            input_schema,
            model: config.model_url,
            exit: output_id,
            output_schema,
            tool_lookup: HashMap::new(),
            keep_alive: config.keep_alive,
            loop_break_message: config.loop_break_message,
            max_tool_calls: config.max_tool_calls,
        };
        self.flow
            .nodes
            .insert(name, FlowNode::Agent(Arc::new(agent_info)));
        self
    }

    /// Attaches a tool to agent `A`.
    ///
    /// * `I` is the tool's input type (its `schema_name()` becomes the tool name).
    /// * `O` is the tool's output type. A `Work` node keyed by `I` must be
    ///   registered separately (via `.work::<I, O, _, _>(handler)`) to execute
    ///   the tool.
    ///
    /// The method interns `I`'s schema name (prefixed by the agent name) as the
    /// tool-entry slot and `O`'s schema name as the tool-exit slot, then adds a
    /// `ToolDefinition` to the agent's tool list.
    pub fn tool<A, I, O>(mut self) -> Self
    where
        A: Agent,
        I: 'static + DeserializeOwned + JsonSchema + Send,
        O: ToolOutput,
    {
        let agent_str = A::node_id();
        let agent_id = self.flow.interner.intern(&agent_str);

        let tool_name = pascal_to_snake(&I::schema_name());
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let parameters = match serde_json::to_value(schema_gen.root_schema_for::<I>()) {
            Ok(v) => v,
            Err(e) => {
                self.errors
                    .push(format!("tool '{}' schema: {e}", tool_name));
                return self;
            }
        };
        let definition = ToolDefinition {
            name: tool_name.clone(),
            description: String::new(),
            parameters,
        };

        let entry_id = self.flow.interner.intern(&I::schema_name());
        let exit_id = self.flow.interner.intern(&O::schema_name());
        let to_message: Box<dyn Fn(Value) -> Result<Message, ToolError> + Send + Sync> =
            Box::new(|value: Value| -> Result<Message, ToolError> {
                let o: O = serde_json::from_value(value).map_err(ToolError::Deserialize)?;
                o.to_message()
            });

        match self.flow.nodes.get_mut(&agent_id) {
            Some(FlowNode::Agent(arc)) => match Arc::get_mut(arc) {
                Some(info) => {
                    info.tools.push(ToolInfo { definition, exit_id, to_message });
                    info.tool_lookup.insert(tool_name, (entry_id, exit_id));
                }
                None => {
                    self.errors.push(format!(
                        "tool: agent '{}' Arc is shared; cannot mutate",
                        agent_str
                    ));
                }
            },
            _ => {
                self.errors.push(format!(
                    "tool: agent '{}' not found (register it with .agent::<A>() first)",
                    agent_str
                ));
            }
        }

        self
    }

    /// Registers a pure branch node.
    /// Use `work` first if routing needs I/O or can fail.
    pub fn either<From, A, B, H>(mut self, func: H) -> Self
    where
        From: Serialize + DeserializeOwned + JsonSchema,
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From) -> Either<A, B> + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("either '{}': duplicate node key", from_id_str));
            return self;
        }
        let left_name = self.flow.interner.intern(&A::schema_name());
        let right_name = self.flow.interner.intern(&B::schema_name());
        let shim: Box<dyn Fn(&Value) -> Result<(NodeId, Value), FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                match func(typed) {
                    Either::Left(a) => {
                        let v = serde_json::to_value(&a).map_err(FlowError::Serialize)?;
                        Ok((left_name, v))
                    }
                    Either::Right(b) => {
                        let v = serde_json::to_value(&b).map_err(FlowError::Serialize)?;
                        Ok((right_name, v))
                    }
                }
            });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Either(EitherInfo {
                entry: from_id,
                left_name,
                right_name,
                func: shim,
            }),
        );

        self
    }

    /// Registers a pure 1->2 fan-out node.
    pub fn fork<From, A, B, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From) -> (A, B) + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("fork '{}': duplicate node key", from_id_str));
            return self;
        }
        let shim: Box<dyn Fn(&Value) -> Result<Vec<StateNode>, FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                let (a, b) = func(typed);
                Ok(vec![node(a)?, node(b)?])
            });
        let a_child = self.flow.interner.intern(&A::schema_name());
        let b_child = self.flow.interner.intern(&B::schema_name());
        self.flow.nodes.insert(
            from_id,
            FlowNode::Fork(ForkInfo {
                name: from_id,
                children: vec![a_child, b_child],
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure 2->1 join node.
    pub fn join<A, B, Out, H>(mut self, func: H) -> Self
    where
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(A, B) -> Out + Send + Sync + 'static,
    {
        let a_id_str = A::schema_name();
        let b_id_str = B::schema_name();
        let a_id = self.flow.interner.intern(&a_id_str);
        let b_id = self.flow.interner.intern(&b_id_str);
        for (id, id_str) in [(a_id, &a_id_str), (b_id, &b_id_str)] {
            if self.flow.nodes.contains_key(&id) {
                self.errors
                    .push(format!("join: duplicate node key '{}'", id_str));
                return self;
            }
        }
        let target_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Arc<dyn Fn(&[Value]) -> Result<StateNode, FlowError> + Send + Sync> =
            Arc::new(move |inputs: &[Value]| {
                let a: A = serde_json::from_value(inputs[0].clone())
                    .map_err(FlowError::Deserialize)?;
                let b: B = serde_json::from_value(inputs[1].clone())
                    .map_err(FlowError::Deserialize)?;
                node(func(a, b))
            });
        self.flow.nodes.insert(
            a_id,
            FlowNode::Join(JoinInfo {
                parents: vec![a_id, b_id],
                target: target_id,
                func: Arc::clone(&shim),
            }),
        );
        self.flow.nodes.insert(
            b_id,
            FlowNode::Join(JoinInfo {
                parents: vec![a_id, b_id],
                target: target_id,
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure 1->N fan-out node.
    /// Supported arities come from [`SplitOutputs`].
    pub fn split<From, Out, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: SplitOutputs,
        H: Fn(From) -> Out + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors.push(format!("split '{}': duplicate node key", from_id_str));
            return self;
        }
        let children: Vec<NodeId> = Out::schema_names()
            .into_iter()
            .map(|s| self.flow.interner.intern(&s))
            .collect();
        let shim: Box<dyn Fn(&Value) -> Result<Vec<StateNode>, FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                func(typed).into_nodes()
            });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Fork(ForkInfo {
                name: from_id,
                children,
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure N->1 join node.
    /// Supported arities come from [`MergeInputs`].
    pub fn merge<In, Out, H>(mut self, func: H) -> Self
    where
        In: MergeInputs,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(In) -> Out + Send + Sync + 'static,
    {
        let parent_names = In::schema_names();
        let parent_ids: Vec<NodeId> = parent_names
            .iter()
            .map(|s| self.flow.interner.intern(s))
            .collect();
        for (id, name) in parent_ids.iter().zip(&parent_names) {
            if self.flow.nodes.contains_key(id) {
                self.errors.push(format!("merge: duplicate node key '{}'", name));
                return self;
            }
        }
        let target_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Arc<dyn Fn(&[Value]) -> Result<StateNode, FlowError> + Send + Sync> =
            Arc::new(move |inputs: &[Value]| {
                let typed = In::from_values(inputs)?;
                node(func(typed))
            });
        for &pid in &parent_ids {
            self.flow.nodes.insert(
                pid,
                FlowNode::Join(JoinInfo {
                    parents: parent_ids.clone(),
                    target: target_id,
                    func: Arc::clone(&shim),
                }),
            );
        }
        self
    }

    /// Embeds a child flow.
    /// The child runs in its own frame and writes `F::Output` back to the parent graph.
    pub fn flow<F: Flow>(mut self) -> Self {
        let input_str = F::schema_name();
        let output_str = F::Output::schema_name();
        let input_id = self.flow.interner.intern(&input_str);
        let output_id = self.flow.interner.intern(&output_str);

        if self.flow.nodes.contains_key(&input_id) {
            self.errors
                .push(format!("flow '{}': duplicate node key", input_str));
            return self;
        }

        let mut inner = match FlowGraph::from_flow::<F>() {
            Ok(g) => g,
            Err(e) => {
                self.errors.push(format!("flow '{}': {e}", input_str));
                return self;
            }
        };

        inner.parent_entry = Some(input_id);
        inner.parent_exit = Some(output_id);

        self.flow
            .nodes
            .insert(input_id, FlowNode::Flow(Arc::new(inner)));

        self
    }

    /// Registers an async work node.
    pub fn work<From, Out, Fut, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        Fut: std::future::Future<Output = Result<Out, FlowError>> + Send + 'static,
        H: Fn(From, Context) -> Fut + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("work '{}': duplicate node key", from_id_str));
            return self;
        }
        let exit_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Box<
            dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync,
        > = Box::new(move |value: &Value, ctx: Context| {
            let typed: From = match serde_json::from_value(value.clone()) {
                Ok(v) => v,
                Err(e) => {
                    let err = FlowError::Deserialize(e);
                    return Box::pin(async move { Err(err) });
                }
            };
            let fut = func(typed, ctx);
            Box::pin(async move {
                let out = fut.await?;
                serde_json::to_value(&out).map_err(FlowError::Serialize)
            })
        });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Work(WorkInfo {
                name: from_id,
                exit_name: exit_id,
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure synchronous transform node.
    /// Use `work` if the step can fail or needs I/O.
    pub fn map<From, Out, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From) -> Out + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("map '{}': duplicate node key", from_id_str));
            return self;
        }
        let exit_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Box<dyn Fn(&Value) -> Result<Value, FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                let out = func(typed);
                serde_json::to_value(&out).map_err(FlowError::Serialize)
            });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Map(MapInfo {
                name: from_id,
                exit_name: exit_id,
                func: shim,
            }),
        );
        self
    }

    /// Registers a flow-level suspend point.
    /// `resume()` must later supply a value of type `O`.
    pub fn suspend<I, O>(mut self) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send,
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
    {
        let entry_str = I::schema_name();
        let exit_str = O::schema_name();
        let entry = self.flow.interner.intern(&entry_str);
        let exit = self.flow.interner.intern(&exit_str);
        if self.flow.nodes.contains_key(&entry) {
            self.errors
                .push(format!("suspend '{}': duplicate node key", entry_str));
            return self;
        }
        let output_type = exit_str.to_string();
        let deserialize: Box<
            dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync,
        > = Box::new(|v| serde_json::from_value::<I>(v).map(SuspendedValue::new));
        self.flow.nodes.insert(
            entry,
            FlowNode::Suspend(SuspendInfo {
                entry,
                exit,
                output_type,
                deserialize,
            }),
        );
        self
    }

    /// Validates structural rules and returns the graph.
    /// Entry wiring happens later in `FlowGraph::with_entry`.
    pub fn build(self) -> Result<FlowGraph, FlowError> {
        if !self.errors.is_empty() {
            return Err(BuildError::Invalid(self.errors).into());
        }
        validate_nodes(&self.flow.nodes, &self.flow)?;
        Ok(self.flow)
    }
}




#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    use super::*;
    use crate::clients::{
        Attachment, Client, ClientError, ClientFactory, ClientOptions, ClientOutput,
        ClientResponse, Message, Provider, ToolCall,
    };
    use crate::commons::{Agent, AgentConfig};
    use crate::context::Context;
    use crate::tools::ToolOutput;

    #[derive(Clone)]
    enum ResponseMode {
        Output(Value),
        ToolCall { name: String, args: Value },
    }

    #[derive(Clone)]
    struct CapturingFactory {
        options: Arc<Mutex<Vec<ClientOptions>>>,
        mode: ResponseMode,
    }

    struct CapturingClient {
        mode: ResponseMode,
    }

    impl CapturingFactory {
        fn new(mode: ResponseMode) -> Self {
            Self {
                options: Arc::new(Mutex::new(Vec::new())),
                mode,
            }
        }

        fn captured(&self) -> Vec<ClientOptions> {
            self.options
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
    }

    #[async_trait]
    impl Client for CapturingClient {
        fn provider(&self) -> Provider {
            Provider::OpenAi
        }

        async fn execute(&self, _messages: &[Message]) -> Result<ClientResponse, ClientError> {
            match &self.mode {
                ResponseMode::Output(value) => Ok(ClientResponse::new(
                    Provider::OpenAi,
                    ClientOutput::Output(value.clone()),
                )),
                ResponseMode::ToolCall { name, args } => Ok(ClientResponse::new(
                    Provider::OpenAi,
                    ClientOutput::ToolCalls {
                        thought: None,
                        calls: vec![ToolCall {
                            id: "call-1".into(),
                            name: name.clone(),
                            args: args.clone(),
                            thought_signatures: None,
                        }],
                    },
                )),
            }
        }
    }

    impl ClientFactory for CapturingFactory {
        fn create(
            &self,
            _model_url: &str,
            options: ClientOptions,
        ) -> Result<Box<dyn Client>, ClientError> {
            self.options
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(options);
            Ok(Box::new(CapturingClient {
                mode: self.mode.clone(),
            }))
        }
    }

    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct PlainAgentInput {
        topic: String,
    }

    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct PlainAgentOutput {
        answer: String,
    }

    impl Agent for PlainAgentInput {
        type Output = PlainAgentOutput;

        fn build() -> AgentConfig {
            AgentConfig::new("Answer briefly.", "openai://test-model")
        }
    }

    impl Flow for PlainAgentInput {
        type Output = PlainAgentOutput;

        fn build(builder: FlowBuilder) -> FlowBuilder {
            builder.agent::<PlainAgentInput>()
        }
    }

    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct MessageAgentInput {
        topic: String,
    }

    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct MessageAgentOutput {
        answer: String,
    }

    impl Agent for MessageAgentInput {
        type Output = MessageAgentOutput;

        fn to_message(self, _ctx: &Context) -> Result<Message, FlowError> {
            let mut message = Message::user(format!("Inspect this screenshot about {}", self.topic));
            message.attachments.push(Attachment::Inline {
                mime_type: "image/png".into(),
                data: "aGVsbG8=".into(),
            });
            Ok(message)
        }

        fn build() -> AgentConfig {
            AgentConfig::new("Answer briefly.", "openai://test-model")
        }
    }

    impl Flow for MessageAgentInput {
        type Output = MessageAgentOutput;

        fn build(builder: FlowBuilder) -> FlowBuilder {
            builder.agent::<MessageAgentInput>()
        }
    }

    /// Tool input for agent-with-tool tests.
    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    #[schemars(rename = "lookup")]
    struct LookupInput {
        query: String,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct LookupOutput {
        result: String,
    }

    impl ToolOutput for LookupOutput {}

    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct ToolAgentInput {
        topic: String,
    }

    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct ToolAgentOutput {
        answer: String,
    }

    impl Agent for ToolAgentInput {
        type Output = ToolAgentOutput;

        fn build() -> AgentConfig {
            AgentConfig::new("Use tools before answering.", "openai://test-model")
        }
    }

    impl Flow for ToolAgentInput {
        type Output = ToolAgentOutput;

        fn build(builder: FlowBuilder) -> FlowBuilder {
            builder
                .agent::<ToolAgentInput>()
                .tool::<ToolAgentInput, LookupInput, LookupOutput>()
                .work(|input: LookupInput, _ctx: Context| async move {
                    Ok(LookupOutput { result: input.query })
                })
        }
    }

    /// Agents without tools stay in structured-output mode.
    #[tokio::test]
    async fn schema_and_tools_dispatch_without_tools_uses_structured_output() {
        let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" })));
        let mut runtime = crate::flows::runtime::FlowRuntime::new(PlainAgentInput {
            topic: "rust".into(),
        })
        .expect("runtime should build")
        .with_factory(factory.clone());

        let _ = runtime.next(Context::default()).await.expect("entry step should run");

        let captured = factory.captured();
        assert_eq!(captured.len(), 1);
        let options = &captured[0];
        assert!(options.tools.is_empty());
        assert_eq!(options.tool_choice, crate::clients::ToolChoice::Disabled);
        let expected = serde_json::to_value(schemars::schema_for!(PlainAgentOutput))
            .expect("output schema should serialize");
        assert_eq!(options.output_schema.as_ref(), Some(&expected));
    }

    /// Agent entry uses `to_message` to populate the first user turn.
    #[tokio::test]
    async fn agent_entry_uses_custom_to_message() {
        let mut runtime = crate::flows::runtime::FlowRuntime::new(MessageAgentInput {
            topic: "rust".into(),
        })
        .expect("runtime should build")
        .with_factory(CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" }))));

        let _ = runtime.next(Context::default()).await.expect("entry step should run");

        let entry = runtime
            .inspector()
            .history()
            .entries()
            .iter()
            .find(|entry| entry.message.content == "Inspect this screenshot about rust")
            .expect("custom user message should be recorded");

        assert!(matches!(entry.message.role, Role::User));
        assert!(matches!(
            entry.message.attachments.as_slice(),
            [Attachment::Inline { mime_type, data }]
                if mime_type == "image/png" && data == "aGVsbG8="
        ));
    }

    /// Agents with tools include the tool definition in options.
    #[tokio::test]
    async fn schema_and_tools_dispatch_with_tools_includes_lookup() {
        let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" })));
        let mut runtime = crate::flows::runtime::FlowRuntime::new(ToolAgentInput {
            topic: "rust".into(),
        })
        .expect("runtime should build")
        .with_factory(factory.clone());

        let _ = runtime.next(Context::default()).await.expect("entry step should run");

        let captured = factory.captured();
        assert_eq!(captured.len(), 1);
        let options = &captured[0];
        assert_eq!(options.tool_choice, crate::clients::ToolChoice::Required);
        assert_eq!(options.tools.len(), 1, "should have exactly one tool (lookup)");

        let lookup = &options.tools[0];
        assert_eq!(lookup.name, "lookup");
        assert!(lookup.parameters.is_object());
    }
}
