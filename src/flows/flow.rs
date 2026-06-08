use std::collections::{HashMap, HashSet, VecDeque};

use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::builder::FlowBuilder;
use super::node_api::Node;
use super::compactor::count_complete_turns;
use super::history::FlowHistory;
use super::nodes::{
    AgentInfo, EachInfo, EitherInfo, FlowNode, ForkInfo, JoinInfo, MapInfo, SuspendInfo,
    ToolWorkInfo, WorkInfo,
};
use crate::flows::NodeId;
use crate::flows::errors::{AgentError, BuildError, FlowError};
use crate::flows::interner::Interner;
use crate::flows::state::{AgentContinuation, AgentState, Callable, EachState, FlowState, WaitingCall};
use crate::flows::validation::validate;
use crate::{
    clients::{
        Client, ClientFactory, ClientOptions, ClientOutput, Message, Role,
        ToolChoice, materialize_messages,
    },
    context::Context,
    tools::{SuspendedValue, ToolDefinition},
};

/// Step result returned by the runtime.
pub enum FlowStep<T = serde_json::Value> {
    /// The flow advanced but has not yet produced a final output.
    Continue,
    /// The flow completed and produced output `T`.
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

/// A typed, declarative agent flow.
///
/// Implement this trait to declare a flow graph. Pravah resolves the graph
/// at runtime using `define` and drives it via [`crate::flows::FlowRuntime`].
///
/// # Migration
///
/// Implement [`define`](Flow::define) for new flows. The legacy [`build`](Flow::build)
/// method is deprecated but still works; its default delegates to `define`.
///
/// # Panics
///
/// If neither `define` nor `build` is overridden, any attempt to compile the flow
/// will panic with a message naming the unimplemented type.
pub trait Flow: 'static + JsonSchema + Serialize + DeserializeOwned + Send + Sync {
    /// Final output produced when all nodes complete.
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Constructs the flow graph using the fluent [`Node`] API.
    ///
    /// `root` is a pre-typed entry node for `Self`. Chain node methods on it and
    /// call [`finalize`](Node::finalize) to return the builder.
    fn define(_root: Node<Self>) -> FlowBuilder {
        unimplemented!(
            "`{}` must implement `Flow::define`",
            std::any::type_name::<Self>()
        )
    }

    /// Constructs the flow graph by registering nodes onto `builder`.
    ///
    /// Deprecated: implement [`define`](Flow::define) instead.
    #[deprecated(note = "implement `define(root: Node<Self>)` instead")]
    fn build(builder: FlowBuilder) -> FlowBuilder {
        Self::define(Node::from_builder(builder))
    }

    /// Unique identifier for this flow's entry node; defaults to the JSON Schema name.
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
    pub(crate) fn new() -> Self {
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
        #[allow(deprecated)]
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

    async fn handle_tool_work(
        node: &ToolWorkInfo,
        ctx: Context,
        states: &mut FlowState,
    ) -> Result<(), FlowError> {
        let state = states.get_state(node.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "tool work node '{}' has no input state",
                node.name.0
            ))
        })?;
        match (node.func)(&state, ctx).await {
            Ok(value) => {
                if !states.set_state(node.exit_name, value, Some(node.name)) {
                    return Err(FlowError::Internal {
                        handler: "handle_tool_work",
                        detail: "frame stack empty on set_state".into(),
                    });
                }
            }
            Err(e) if !e.is_fatal() => {
                tracing::warn!(
                    node = %node.name.0,
                    error = %e,
                    kind = %e.error_kind(),
                    "tool work: non-fatal error will be forwarded to model"
                );
                if !states.set_state(node.exit_name, e.to_json(&node.tool_name), Some(node.name)) {
                    return Err(FlowError::Internal {
                        handler: "handle_tool_work",
                        detail: "frame stack empty on set_state".into(),
                    });
                }
                if !states.add_tool_error_exit(node.agent_id, node.exit_name) {
                    return Err(FlowError::Internal {
                        handler: "handle_tool_work",
                        detail: "agent not found for error_exits update".into(),
                    });
                }
            }
            Err(e) => return Err(FlowError::Internal {
                handler: "handle_tool_work",
                detail: e.to_string(),
            }),
        }
        Ok(())
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

    fn handle_either(either: &EitherInfo, states: &mut FlowState) -> Result<(), FlowError> {
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

    /// Fans out over a `Vec<F>` by running the child sub-flow once per item,
    /// collecting `F::Output` results into `Vec<F::Output>`.
    ///
    /// On the first visit, the input slot holds a JSON array of `F` items.
    /// On each subsequent visit, the feedback slot (`info.id`) holds one completed
    /// `F::Output` from the child frame that just exited.
    ///
    /// Fails if the input slot is absent or is not a JSON array.
    fn handle_each(info: &EachInfo, states: &mut FlowState) -> Result<FlowStep, FlowError> {
        let callable = Callable {
            parent_entry: info.id,
            parent_exit: info.id,
            entry: info.inner.entry,
            exit: info.inner.exit,
            index: info.callable_index,
            keep_alive: false,
        };

        if states.each_queue_has(info.id) {
            let result = states.take_state(info.id).ok_or_else(|| FlowError::Internal {
                handler: "handle_each",
                detail: "feedback slot empty on subsequent visit".into(),
            })?;
            let queue = states.each_queue_get_mut(info.id).ok_or_else(|| FlowError::Internal {
                handler: "handle_each",
                detail: "each queue missing after has() returned true".into(),
            })?;
            queue.results.push(result);

            if let Some(next_item) = queue.remaining.pop_front() {
                if !states.set_state(info.id, next_item, None) {
                    return Err(FlowError::Internal {
                        handler: "handle_each",
                        detail: "frame stack empty on set_state for next item".into(),
                    });
                }
                states.call_enter(callable);
            } else {
                let each_state = states.each_queue_remove(info.id).ok_or_else(|| FlowError::Internal {
                    handler: "handle_each",
                    detail: "queue vanished before results could be collected".into(),
                })?;
                if !states.set_state(info.exit, Value::Array(each_state.results), None) {
                    return Err(FlowError::Internal {
                        handler: "handle_each",
                        detail: "frame stack empty on set_state for exit".into(),
                    });
                }
            }
            return Ok(FlowStep::Continue);
        }

        let vec_value = states.take_state(info.id).ok_or_else(|| FlowError::Internal {
            handler: "handle_each",
            detail: "input slot missing on first visit".into(),
        })?;
        let items: Vec<Value> = vec_value
            .as_array()
            .ok_or_else(|| FlowError::Internal {
                handler: "handle_each",
                detail: "input is not a JSON array".into(),
            })?
            .clone();

        if items.is_empty() {
            if !states.set_state(info.exit, Value::Array(vec![]), None) {
                return Err(FlowError::Internal {
                    handler: "handle_each",
                    detail: "frame stack empty on set_state for empty exit".into(),
                });
            }
            return Ok(FlowStep::Continue);
        }

        let mut remaining: VecDeque<Value> = items.into();
        let Some(first) = remaining.pop_front() else {
            return Err(FlowError::Internal {
                handler: "handle_each",
                detail: "deque was empty despite non-empty items check".into(),
            });
        };
        states.each_queue_init(info.id, EachState { remaining, results: Vec::new() });
        if !states.set_state(info.id, first, None) {
            return Err(FlowError::Internal {
                handler: "handle_each",
                detail: "frame stack empty on set_state for first item".into(),
            });
        }
        states.call_enter(callable);
        Ok(FlowStep::Continue)
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
        self.step(factory, ctx, history, None, states).await
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
        let current = states.get_agent_state(node.id).cloned();

        match current {
            None => {
                let session_id = states.get_or_init_session_id(node.id, node.keep_alive);
                let input = states.get_state(node.id).cloned().ok_or_else(|| {
                    FlowError::NotFound(format!(
                        "agent '{}': input state missing",
                        flow.interner.name_of(node.id)
                    ))
                })?;
                let agent_name = flow.interner.name_of(node.id);
                let message = (node.make_message)(input, &ctx)?;
                history.push(&session_id, agent_name, message);
                states.init_agent_state(node.id, session_id.clone());
                Ok(FlowStep::Continue)
            }

            Some(AgentState {
                ref session_id,
                continuation: AgentContinuation::Dispatch,
                ..
            }) => {
                let session_id = session_id.clone();
                Self::dispatch_agent(node, flow, factory, ctx, history, states, &session_id).await
            }

            Some(AgentState {
                ref session_id,
                continuation:
                    AgentContinuation::PendingTool {
                        ref active,
                        ref waiting,
                        ref error_exits,
                    },
                ..
            }) => {
                let session_id = session_id.clone();
                let active = active.clone();
                let waiting = waiting.clone();
                let error_exits = error_exits.clone();
                Self::handle_pending_tools(
                    node, flow, history, states, session_id, active, waiting, error_exits,
                )
            }
        }
    }

    fn handle_pending_tools(
        node: &AgentInfo,
        flow: &FlowGraph,
        history: &mut FlowHistory,
        states: &mut FlowState,
        session_id: String,
        mut active: HashMap<NodeId, (String, String)>,
        mut waiting: HashMap<NodeId, VecDeque<WaitingCall>>,
        mut error_exits: HashSet<String>,
    ) -> Result<FlowStep, FlowError> {
        let agent_name = flow.interner.name_of(node.id);
        let mut completions: Vec<(NodeId, String)> = Vec::new();
        for (&exit_id, (call_id, _)) in &active {
            if states.contains_state(exit_id) {
                completions.push((exit_id, call_id.clone()));
            }
        }

        let completed_count = completions.len();
        for (exit_id, call_id) in completions {
            let value = states
                .take_state(exit_id)
                .ok_or_else(|| FlowError::Internal {
                    handler: "handle_pending_tools",
                    detail: format!("tool exit {:?} missing after contains_state check", exit_id),
                })?;
            let tool_info = node
                .tools
                .iter()
                .find(|t| t.exit_id == exit_id)
                .ok_or_else(|| FlowError::Internal {
                    handler: "handle_pending_tools",
                    detail: format!("no ToolInfo for exit_id {:?}", exit_id),
                })?;
            let mut msg = if error_exits.remove(call_id.as_str()) {
                Message::tool_output(String::new(), value.to_string())
            } else {
                match (tool_info.to_message)(value) {
                    Ok(m) => m,
                    Err(e) if !e.is_fatal() => {
                        tracing::warn!(
                            agent = %agent_name,
                            call_id = %call_id,
                            error = %e,
                            kind = %e.error_kind(),
                            "tool error sent to model"
                        );
                        e.into_error_message(&tool_info.definition.name)
                    }
                    Err(e) => return Err(FlowError::Internal {
                        handler: "handle_pending_tools",
                        detail: e.to_string(),
                    }),
                }
            };
            msg.role = Role::Tool { call_id };
            history.push(&session_id, agent_name, msg);

            active.remove(&exit_id);
            if let Some(queue) = waiting.get_mut(&exit_id) {
                if let Some(next) = queue.pop_front() {
                    if !states.set_state(next.entry_id, next.args, None) {
                        return Err(FlowError::Internal {
                            handler: "handle_pending_tools",
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

        if let Some(s) = states.get_agent_state_mut(node.id) {
            s.continuation = if active.is_empty() {
                AgentContinuation::Dispatch
            } else {
                AgentContinuation::PendingTool { active, waiting, error_exits }
            };
        }
        states.reinsert_state(node.id);
        Ok(FlowStep::Continue)
    }
}

pub(crate) fn maybe_inject_turn_budget_message(
    node: &AgentInfo,
    client: &dyn Client,
    agent_name: &str,
    session_id: &str,
    history: &FlowHistory,
    session_msgs: &mut Vec<Message>,
) {
    if node.tools.is_empty() && !client.uses_exit_tool() {
        return;
    }
    let Some(budget) = node.turn_budget else {
        return;
    };
    let completed = count_complete_turns(&history.session_entries(session_id));
    if completed + 1 < budget as usize {
        return;
    }
    let exit_tool_name = (client.uses_exit_tool() && !node.output_type_name.is_empty())
        .then_some(node.output_type_name.as_str());
    let text = node
        .turn_budget_message
        .as_deref()
        .map(|msg| client.wrap_system_reminder(msg))
        .unwrap_or_else(|| client.default_turn_budget_message(exit_tool_name));
    tracing::warn!(
        agent = %agent_name,
        completed_turns = completed,
        budget = budget,
        "turn budget reached; injecting last-turn reminder"
    );
    session_msgs.push(Message::user(text));
}

impl FlowGraph {
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
        let tool_choice = if defs.is_empty() {
            ToolChoice::Disabled
        } else {
            ToolChoice::Auto
        };
        tracing::info!(
            agent = %agent_name,
            model = %node.model,
            tools = defs.len(),
            session_id = %session_id,
            "LLM dispatch"
        );

        let has_prior_history = !history.session_entries(session_id).is_empty();
        let mut options = ClientOptions::default();
        options.output_type_name = node.output_type_name.clone();
        let options = options
            .with_input_schema(node.input_schema.clone())
            .with_tools(defs)
            .with_tool_choice(tool_choice)
            .with_name(agent_name.clone())
            .with_output_schema(node.output_schema.clone());
        let options = if has_prior_history {
            options
        } else {
            options.with_preamble(node.effective_preamble(&ctx))
        };

        let client = factory.create(&node.model, options).map_err(|e| {
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

        let mut session_msgs = materialize_messages(&history.for_session(session_id), &ctx)
            .await
            .map_err(|e| {
                tracing::error!(agent = %agent_name, error = %e, "message materialization failed");
                AgentError::LlmFailed {
                    agent: agent_name.clone(),
                    reason: e.to_string(),
                }
            })?;

        maybe_inject_turn_budget_message(node, &*client, &agent_name, session_id, history, &mut session_msgs);

        tracing::debug!(
            agent = %agent_name,
            session_id = %session_id,
            message_count = session_msgs.len(),
            last_message = ?session_msgs.last(),
            "LLM request history"
        );

        let response = client.execute(&session_msgs).await.map_err(|e| {
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

                let mut seen_call_ids = HashSet::new();
                let mut active: HashMap<NodeId, (String, String)> =
                    HashMap::with_capacity(calls.len());
                let mut waiting: HashMap<NodeId, VecDeque<WaitingCall>> = HashMap::new();

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
                                format!(r#"{{"error":"unknown tool '{}'"}}"#, call.name),
                            ),
                        );
                        continue;
                    };

                    if active.contains_key(&exit_id) {
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
                    AgentContinuation::PendingTool { active, waiting, error_exits: HashSet::new() }
                } else {
                    AgentContinuation::Dispatch
                };

                if let Some(s) = states.get_agent_state_mut(node.id) {
                    s.continuation = continuation;
                }

                if needs_pending {
                    states.reinsert_state(node.id);
                }

                Ok(FlowStep::Continue)
            }
        }
    }

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
                FlowNode::ToolWork(info) => {
                    tracing::debug!(node = %node_name, kind = "tool_work", "dispatching node");
                    Self::handle_tool_work(info, ctx, states).await?;
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
                FlowNode::Each(info) => {
                    tracing::debug!(node = %node_name, kind = "each", "dispatching node");
                    return Self::handle_each(info, states);
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
        let result = self.step_inner(factory, ctx, history, states).await?;
        match result {
            FlowStep::Continue => {
                if let Some(v) = states.call_exit() {
                    return Ok(FlowStep::Done(v));
                }
                Ok(FlowStep::Continue)
            }
            other => Ok(other),
        }
    }
}