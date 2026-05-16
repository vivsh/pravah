use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::history::FlowHistory;
use super::nary::{MergeInputs, SplitOutputs};
use crate::flows::NodeId;
use crate::flows::errors::{AgentError, BuildError, FlowError};
use crate::flows::interner::Interner;
use crate::flows::phase::Phase;
use crate::flows::state::{Callable, FlowState};
use crate::flows::validation::{validate, validate_nodes};
use crate::{
    clients::{ClientFactory, ClientOptions, ClientOutput, Message, Role},
    commons::Agent,
    context::Context,
    tools::{SuspendedValue, ToolBox, ToolError},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct StateNode {
    name: String,
    value: serde_json::Value,
}

pub(crate) struct AgentInfo {
    pub(crate) id: NodeId,
    pub(crate) tool_box: Arc<ToolBox>,
    pub(crate) preamble: String,
    pub(crate) model: String,
    pub(crate) exit: NodeId,
    pub(crate) output_schema: Value,
    /// Maps tool call name → (entry_id, exit_id) for every tool in `tool_box`.
    pub(crate) tool_lookup: HashMap<String, (NodeId, NodeId)>,
}

/// Metadata for a single tool exposed by an agent, stored as a graph node so the
/// flow engine can dispatch it independently rather than inlining all tool calls.
pub(crate) struct ToolInfo {
    /// State-map key used for this tool: interned `"AgentName::tool_name"`.
    pub(crate) entry: NodeId,

    pub(crate) exit: NodeId,

    tool_index: usize,
    /// Shared toolbox owned by the parent agent.
    tool_box: Arc<ToolBox>,
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

/// A pure synchronous transformation node: `fn(I) -> O`.
/// No context, no error path — if it needs I/O or can fail, use `work`.
pub(crate) struct MapInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
    func: Box<dyn Fn(&Value) -> Result<Value, FlowError> + Send + Sync>,
}

/// A flow-level suspend node. When `I` is present in state the flow pauses
/// and surfaces a [`SuspendedValue`]. On resume a value of type `O` is written
/// to state under `exit`.
pub(crate) struct SuspendInfo {
    pub(crate) entry: NodeId,
    pub(crate) exit: NodeId,
    pub(crate) output_type: String,
    /// Deserialises the raw [`Value`] from state into a type-erased [`SuspendedValue`]
    /// so the caller can downcast it to `I`.
    deserialize: Box<dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync>,
}

/// Constructs a typed [`StateNode`] from an [`Agent`] input value.
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
    /// Pure synchronous infallible transform: `fn(I) -> O`.
    Map(MapInfo),
    /// Flow-level suspend point: pauses when `I` is in state, resumes with `O`.
    Suspend(SuspendInfo),
    /// An embedded sub-flow. Boxed to break the recursive type-size cycle.
    Flow(Arc<FlowGraph>),
    /// A single tool dispatched by a parent agent. Not statically reachable from the
    /// entry node; it enters the state map when the agent issues a tool call.
    Tool(ToolInfo),
    /// An agent invoked as a tool by a parent agent. Entered dynamically when the
    /// parent LLM issues a tool call; runs as a nested frame whose exit wires directly
    /// to the parent agent's output slot.
    AgentTool(Arc<AgentInfo>),
    /// A flow invoked as a tool by a parent agent. Same frame-push semantics as
    /// [`FlowNode::Flow`], but entered dynamically via an LLM tool call.
    FlowTool {
        name: NodeId,
        inner: Arc<FlowGraph>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AgentContinuation {
    Dispatch,
    /// Pending tool calls from the current LLM turn.
    /// Key: tool exit NodeId; value: (call_id, call_name).
    PendingTool(HashMap<NodeId, (String, String)>),
    Exit(Value),
}


/// Typed step result returned by [`FlowRuntime`].
pub enum FlowStep<T = serde_json::Value> {
    Continue,
    Done(T),
    /// The flow paused at a suspend-tool call. Downcast the inner [`SuspendedValue`] to
    /// the concrete input type registered via [`crate::tools::ToolBox::suspend`].
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

    fn build() -> Result<FlowGraph, FlowError>;

    fn node_id() -> String {
        Self::schema_name()
    }
}

/// A complete flow graph with entry/exit points and interner, ready for execution.
/// - All agent history related side-effects must be restricted to only 2 function (handle_child_agent, dispatch_agent) 
/// in the runtime, 
/// - All other handlers will merely moduleate the state
/// 
/// Node: 
/// - Represents data transformation. It always has an entry or input and 
///     one or more (fork) exits or output. 
/// - Ideally when input is processed, it should be removed from the state. 
/// - Input/Output are completely decoupled. 
/// - Within the same graph, no two nodes can have the same entry, but multiple 
///     nodes can have the same exit.
/// 
/// Suspend:
/// - Single deterministic global suspend point for the entire flow.
/// - Only one node can trigger a suspend at any moment - it may or may not be a tool call.
/// - On resume, the value is injected as the output for the suspending node.
/// - Rest of the data-flow is not affected at all.
/// - This is merely a mechanism for external fulfillment for specific nodes.
/// - This can be useful for human-in-the-loop flows, or for integrating with 
///     external systems that require async callbacks (e.g. waiting for an event, 
///     or a long-running computation).
/// 
pub struct FlowGraph {
    pub(crate) nodes: HashMap<NodeId, FlowNode>,

    pub(crate) entry: NodeId,
    pub(crate) exit: NodeId,

    pub(crate) parent_exit: Option<NodeId>,

    /// Input node id when this graph is embedded as a sub-flow; `None` for the root graph.
    pub(crate) parent_entry: Option<NodeId>,

    /// Forward intern map: string name → NodeId.
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
        F::build()?.with_entry(entry, exit)
    }

    /// A join node is ready to execute when all its parent nodes have produced values.
    /// Can only be tested at runtime, not build time, because parent nodes may be dynamically generated agents.
    /// join target can be a terminal and may not be present in the node map at build time.
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

    /// Injects the entry point and runs full graph validation (entry + reachability).
    /// Called by [`FlowRuntime`] after [`Flow::build`] returns the graph.
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

    async fn handle_tool(
        node: &ToolInfo,
        flow: &FlowGraph,
        ctx: Context,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let input = states
            .get_state(node.entry)
            .ok_or_else(|| {
                FlowError::NotFound(format!(
                    "tool '{}': call state missing",
                    flow.interner.name_of(node.entry),
                ))
            })?
            .clone();
        match node
            .tool_box
            .call_at_index(node.tool_index, input, ctx)
            .await
        {
            Ok(value) => {
                if !states.set_state(node.exit, value, Some(node.entry)) {
                    return Err(FlowError::Internal {
                        handler: "handle_tool",
                        detail: "frame stack empty on set_state".into(),
                    });
                }
                Ok(FlowStep::Continue)
            }
            Err(ToolError::Exit(value)) => {
                if !states.remove_state(node.entry){
                    return Err(FlowError::Internal {
                        handler: "handle_tool_exit",
                        detail: "frame stack empty on remove".into(),
                    });
                }

                let continuation = AgentContinuation::Exit(value);

                let content = serde_json::to_value(&continuation).map_err(AgentError::Serialize)?;

                if !states.set_phase(Phase::Continue(Some(content))) {
                    return Err(FlowError::Internal {
                        handler: "handle_tool_exit",
                        detail: "frame stack empty on set_phase".into(),
                    });
                }

                Ok(FlowStep::Continue)
            }
            Err(ToolError::Suspend { value, output_type }) => {
                states.suspend(node.entry, node.exit, output_type);
                Ok(FlowStep::Suspend(value))
            }
            Err(e) => Err(AgentError::ToolFailed {
                tool: node.tool_box.name_at(node.tool_index).to_string(),
                reason: e.to_string(),
            }
            .into()),
        }
    }

    /// Handles [`FlowNode::AgentTool`]: reads the args `Value` from the state slot,
    /// pushes a new frame wired so the inner agent's exit lands directly on the outer
    /// agent's output slot, then sets `Phase::Entry` to start the inner agent.
    fn handle_agent_tool(
        node: &AgentInfo,
        outer: &FlowGraph,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let callable = Callable {
            parent_entry: node.id,
            parent_exit: node.exit,
            exit: node.exit,
            entry: node.id,
            index: outer.callable_index,
        };

        states.call_enter(callable);

        if !states.set_phase(Phase::Entry) {
            return Err(FlowError::Internal {
                handler: "handle_agent_tool",
                detail: "frame stack empty on set_phase".into(),
            });
        }
        Ok(FlowStep::Continue)
    }

    /// Handles [`FlowNode::FlowTool`]: reads `args` from the state slot and
    /// pushes a new frame for the inner flow, wired to the outer agent's output slot.
    fn handle_flow_tool(
        inner: &Arc<FlowGraph>,
        outer: &FlowGraph,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let tool_node_id = inner.parent_entry.ok_or_else(|| FlowError::Internal {
            handler: "handle_flow_tool",
            detail: "inner flow missing parent_entry".into(),
        })?;
        let parent_exit = inner.parent_exit.ok_or_else(|| FlowError::Internal {
            handler: "handle_flow_tool",
            detail: "inner flow missing parent_exit".into(),
        })?;

        let callable = Callable {
            parent_entry: tool_node_id,
            parent_exit,
            exit: inner.exit,
            entry: inner.entry,
            index: inner.callable_index,
        };
        states.call_enter(callable);

        Ok(FlowStep::Continue)
    }

    fn handle_parent_agent(
        node: &AgentInfo,
        flow: &FlowGraph,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let parent_entry = node.id;
        let parent_exit = node.exit;

        let callable = Callable {
            parent_entry,
            parent_exit,
            exit: parent_exit,
            entry: parent_entry,
            index: flow.callable_index,
        };

        states.call_enter(callable);
        if !states.set_phase(Phase::Entry) {
            return Err(FlowError::Internal {
                handler: "handle_parent_agent",
                detail: "frame stack empty after call_enter".into(),
            });
        }

        Ok(FlowStep::Continue)
    }

    async fn handle_child_agent(
        node: &AgentInfo,
        flow: &FlowGraph,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let session_id = states.top_session_id().to_owned();
        let phase = states.phase().cloned().ok_or_else(|| FlowError::Internal {
            handler: "handle_child_agent",
            detail: "called without a phase".into(),
        })?;

        match phase {
            Phase::Entry => {
                // First dispatch: record the agent's input as a user message then call LLM.
                let input = states
                    .get_state(node.id)
                    .ok_or_else(|| {
                        FlowError::NotFound(format!(
                            "agent '{}': input state missing",
                            flow.interner.name_of(node.id)
                        ))
                    })?
                    .clone();

                let agent_name = flow.interner.name_of(node.id);
                history.push(
                    &session_id,
                    agent_name,
                    Message::from_json(Role::User, &input)
                        .map_err(AgentError::Serialize)?,
                );

                // Advance phase: Entry is consumed once the user message is in history.
                // Any future re-entry will find Continue(Dispatch) and skip re-pushing.
                let dispatch_val = serde_json::to_value(AgentContinuation::Dispatch)
                    .map_err(AgentError::Serialize)?;
                if !states.set_phase(Phase::Continue(Some(dispatch_val))) {
                    return Err(FlowError::Internal {
                        handler: "handle_child_agent",
                        detail: "Entry: frame stack empty on set_phase".into(),
                    });
                }

                Self::dispatch_agent(node, flow, factory, history, states).await
            }

            Phase::Continue(val) => {
                let cont = match val {
                    Some(v) => Some(
                        serde_json::from_value::<AgentContinuation>(v)
                            .map_err(AgentError::Deserialize)?,
                    ),
                    None => None,
                };
                match cont {
                    Some(AgentContinuation::Exit(value)) => {
                        // Exit the call with provided value. No dispatch is needed
                        if !states.set_state(node.exit, value, None){
                            return Err(FlowError::Internal {
                                handler: "handle_child_agent_exit",
                                detail: "frame stack empty on set_state".into(),
                            });
                        }
                        Ok(FlowStep::Continue)
                    }
                    Some(AgentContinuation::PendingTool(mut tool_map)) => {
                        let mut completions = HashSet::new();
                        for (t, (call_id, _)) in tool_map.iter() {
                            if let Some(value) = states.take_state(*t) {
                                let call_id = call_id.clone();
                                let agent_id = flow.interner.name_of(node.id);
                                completions.insert(*t);
                                history.push(
                                    &session_id,
                                    agent_id,
                                    Message::tool_output(
                                        call_id,
                                        serde_json::to_string(&value).map_err(AgentError::Serialize)?,
                                    ),
                                );
                            }
                        }

                        tool_map.retain(|k, _| !completions.contains(k));

                        if tool_map.is_empty() {
                            // All tool calls completed within the same turn — skip the pending phase and re-dispatch immediately.

                            // All tool results are now in history — re-dispatch to LLM.
                            let phase = Phase::Continue(Some(
                                serde_json::to_value(AgentContinuation::Dispatch)
                                    .map_err(AgentError::Serialize)?,
                            ));
                            states.set_phase(phase);
                        }else{
                            states.reinsert_state(node.id); // move the agent's input to the end of the state map so that we reach pending tools first on the next step
                            // Some tool calls are still pending — update the phase to reflect the remaining calls.
                            let phase = Phase::Continue(Some(
                                serde_json::to_value(AgentContinuation::PendingTool(tool_map))
                                    .map_err(AgentError::Serialize)?,
                            ));
                            states.set_phase(phase);
                        }
                        return Ok(FlowStep::Continue);
                    }
                    // Explicit Dispatch: call LLM.
                    Some(AgentContinuation::Dispatch) => {
                        Self::dispatch_agent(node, flow, factory, history, states).await
                    }
                    // None: agent output written to exit slot; call_exit will pop the frame.
                    None => Ok(FlowStep::Continue),
                }
            }
        }
    }

    /// Calls the LLM once and handles both structured-output and tool-call responses.
    ///
    /// **Structured output path** (`ClientOutput::Output`):
    /// - Pushes an assistant message to history.
    /// - Writes the output value to `node.exit`, retires `node.id` from the state map.
    /// - Sets phase to `Continue(None)` signalling the frame is ready for `call_exit`.
    ///
    /// **Tool-call path** (`ClientOutput::ToolCalls`):
    /// - Pushes an `AssistantToolCalls` message to history.
    /// - Moves the agent's input slot to the *end* of the state map so that step_inner
    ///   reaches pending tool states (which come first) before triggering this agent again.
    /// - Writes each pending `ToolCall` as a JSON value under its interned tool `NodeId`.
    /// - Sets phase to `Continue(PendingTool(calls))`.
    async fn dispatch_agent(
        node: &AgentInfo,
        flow: &FlowGraph,
        factory: &dyn ClientFactory,
        history: &mut FlowHistory,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let session_id = states.top_session_id().to_owned();
        let agent_name = flow.interner.name_of(node.id).to_string();

        let options = ClientOptions::default()
            .with_preamble(node.preamble.clone())
            .with_tools(node.tool_box.definitions())
            .with_output_schema(node.output_schema.clone())
            .with_name(agent_name.clone());

        let client = factory
            .create(&node.model, options)
            .map_err(|e| AgentError::LlmFailed {
                agent: agent_name.clone(),
                reason: e.to_string(),
            })?;

        history.validate_for_session(&session_id).map_err(|e| AgentError::LlmFailed {
            agent: agent_name.clone(),
            reason: e.to_string(),
        })?;

        let session_msgs = history.for_session(&session_id);

        let response =
            client
                .execute(&session_msgs)
                .await
                .map_err(|e| AgentError::LlmFailed {
                    agent: agent_name.clone(),
                    reason: e.to_string(),
                })?;

        match response.output {
            ClientOutput::Output(val) => {
                let content = serde_json::to_string(&val).map_err(AgentError::Serialize)?;
                let msg = if let Some(usage) = response.usage {
                    Message::assistant(content).with_usage(usage)
                } else {
                    Message::assistant(content)
                };
                history.push(&session_id, &agent_name, msg);

                // Retire the input slot, write the output slot.
                if !states.set_state(node.exit, val, Some(node.id)) {
                    return Err(FlowError::Internal {
                        handler: "dispatch_agent",
                        detail: "Output: frame stack empty on set_state".into(),
                    });
                }
                // Phase::Continue(None): agent is done; call_exit will pop this frame.
                if !states.set_phase(Phase::Continue(None)) {
                    return Err(FlowError::Internal {
                        handler: "dispatch_agent",
                        detail: "Output: frame stack empty on set_phase".into(),
                    });
                }

                Ok(FlowStep::Continue)
            }

            ClientOutput::ToolCalls { thought, calls } => {
                let atc_msg = Message {
                    role: Role::AssistantToolCalls {
                        calls: calls.clone(),
                    },
                    content: thought.unwrap_or_default(),
                    usage: response.usage,
                };
                history.push(&session_id, &agent_name, atc_msg);

                let mut seen_ids = std::collections::HashSet::new();

                // exit_id → (call_id, call_name)
                let mut pending_calls: HashMap<NodeId, (String, String)> =
                    HashMap::with_capacity(calls.len());

                for call in calls {
                    let (entry_id, exit_id) =
                        node.tool_lookup.get(&call.name).copied().ok_or_else(|| {
                            AgentError::UnknownTool {
                                agent: agent_name.clone(),
                                tool: call.name.clone(),
                            }
                        })?;

                    if !seen_ids.insert(entry_id) {
                        return Err(AgentError::DuplicateToolCall {
                            agent: agent_name.clone(),
                            tool: call.name.clone(),
                        }
                        .into());
                    }

                    if !states.set_state(entry_id, call.args, None) {
                        return Err(FlowError::Internal {
                            handler: "dispatch_agent",
                            detail: "ToolCalls: frame stack empty on set_state".into(),
                        });
                    }

                    pending_calls.insert(exit_id, (call.id.clone(), call.name.clone()));
                }

                if !states.set_phase(Phase::Continue(Some(
                    serde_json::to_value(AgentContinuation::PendingTool(pending_calls))
                        .map_err(AgentError::Serialize)?,
                ))) {
                    return Err(FlowError::Internal {
                        handler: "dispatch_agent",
                        detail: "ToolCalls: frame stack empty on set_phase".into(),
                    });
                }

                states.reinsert_state(node.id); // move the agent's input to the end of the state map so that we reach pending tools first on the next step

                Ok(FlowStep::Continue)
            }
        }
    }

    /// Dispatches one step on `self` (the currently active graph for the top frame).
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
            match current_node {
                FlowNode::Agent(agent) => {
                    if let Some(_phase) = states.phase() {
                        return Self::handle_child_agent(
                            agent, self, factory, ctx, history, states,
                        )
                        .await;
                    } else {
                        return Self::handle_parent_agent(agent, self, states);
                    }
                }
                FlowNode::Tool(info) => {
                    return Self::handle_tool(info, self, ctx,  states).await;
                }
                FlowNode::Either(either) => {
                    Self::handle_either(either, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Fork(info) => {
                    Self::handle_fork(info, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Join(info) => {
                    if !self.can_join(current_node_id, states) {
                        continue;
                    }
                    Self::handle_join(info, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Work(info) => {
                    Self::handle_work(info, ctx, states).await?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Map(info) => {
                    Self::handle_map(info, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Suspend(info) => {
                    return Self::handle_suspend(info, states);
                }
                FlowNode::Flow(inner) => {
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
                    };

                    states.call_enter(callable);

                    return Ok(FlowStep::Continue);
                }
                FlowNode::AgentTool(agent) => {
                    if states.callable_entry() == Some(current_node_id) {
                        // We are inside the agent tool's own frame — run it as a child agent.
                        return Self::handle_child_agent(
                            agent, self, factory, ctx, history, states,
                        )
                        .await;
                    } else {
                        // First encounter: push a new frame for the agent tool.
                        return Self::handle_agent_tool(agent, self, states);
                    }
                }
                FlowNode::FlowTool { inner, .. } => {
                    return Self::handle_flow_tool(inner, self, states);
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
            other => Ok(other), // Suspend — do NOT call call_exit
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

    /// Registers an agent node. The node name is derived from `A::node_id()`.
    pub fn agent<A: Agent>(mut self) -> Self {
        let name_str = A::node_id();
        let name = self.flow.interner.intern(&name_str);
        if self.flow.nodes.contains_key(&name) {
            self.errors
                .push(format!("agent '{}': duplicate node key", name_str));
            return self;
        }
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let output_schema = match serde_json::to_value(schema_gen.root_schema_for::<A::Output>()) {
            Ok(v) => v,
            Err(e) => {
                self.errors
                    .push(format!("agent '{}' output schema: {e}", name_str));
                return self;
            }
        };
        let config = A::build();
        let tool_box = Arc::new(config.tool_box.with_agent::<A>(&mut self.flow));
        let output_str = A::Output::schema_name();
        let output_id = self.flow.interner.intern(&output_str);
        let mut tool_lookup: HashMap<String, (NodeId, NodeId)> = HashMap::new();
        // Pre-pass: build tool_lookup before inserting the agent node.
        for i in 0..tool_box.len() {
            let tool_name = tool_box.name_at(i).to_owned();
            let tool_entry =
                self.flow
                    .interner
                    .intern(&format!("{}::{}", name_str, tool_box.input_type_at(i)));
            let tool_exit = self.flow.interner.intern(&tool_box.output_type_at(i));
            tool_lookup.insert(tool_name, (tool_entry, tool_exit));
        }
        let agent_info = AgentInfo {
            id: name,
            tool_box: Arc::clone(&tool_box),
            preamble: config.preamble,
            model: config.model_url,
            exit: output_id,
            output_schema,
            tool_lookup,
        };
        self.flow
            .nodes
            .insert(name, FlowNode::Agent(Arc::new(agent_info)));
        // Register a FlowNode::Tool for each tool in the toolbox (including the submit sentinel).
        // FlowToolDispatcher and AgentToolDispatcher return needs_tool_node() = false because
        // their graph nodes (FlowNode::FlowTool / FlowNode::AgentTool) are already injected by
        // with_agent; registering a plain Tool node for them would overwrite the injected node
        // and cause call_raw to be invoked instead of the frame-push logic.
        for i in 0..tool_box.len() {
            if !tool_box.needs_tool_node_at(i) {
                continue;
            }
            let _tool_name = tool_box.name_at(i);
            let tool_entry =
                self.flow
                    .interner
                    .intern(&format!("{}::{}", name_str, tool_box.input_type_at(i)));
            let tool_exit = self.flow.interner.intern(&tool_box.output_type_at(i));

            self.flow.nodes.insert(
                tool_entry,
                FlowNode::Tool(ToolInfo {
                    entry: tool_entry,
                    exit: tool_exit,
                    tool_index: i,
                    tool_box: Arc::clone(&tool_box),
                }),
            );
        }
        self
    }

    /// Registers a routing node at `From`. The closure receives `From` and returns
    /// `Either<A, B>` — a pure infallible branch decision. No context is available;
    /// if the routing needs I/O or can fail, use a `work` node before it.
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

    /// Registers a fork node at `From`. The closure receives the parent value and returns
    /// two child values placed into state for independent processing. Pure and infallible;
    /// no context is available.
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

    /// Registers a join node that waits for both `A` and `B` states to be present,
    /// combines them into `Out`, and clears the parent states. Pure and infallible;
    /// no context is available.
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

    /// Registers a split node at `From` (1→N fan-out). The closure receives the parent
    /// value and returns an N-tuple; each element becomes an independent branch in the
    /// state map. Supports arities 2–16 via [`SplitOutputs`]. Pure and infallible;
    /// no context is available.
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

    /// Registers a merge node (N→1 fan-in). Waits until all elements of `In`'s tuple
    /// are present in the state map, passes them as a typed tuple to `func`, and writes
    /// the result. Supports arities 2–16 via [`MergeInputs`]. Pure and infallible;
    /// no context is available.
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

    /// Embeds sub-flow `F` as a [`FlowNode::Flow`] node.
    ///
    /// The node key is `F::schema_name()` (the input type). When the flow engine
    /// encounters this node it pushes a new execution frame, seeds it with the input
    /// value, and resumes the inner graph until it produces `F::Output`. The output
    /// is then written to the parent frame under `F::Output::schema_name()`.
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

    /// Registers a work node at `From`. The async closure transforms the input value into
    /// `Out` without LLM involvement.
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

    /// Registers a pure synchronous transformation node at `From`. The closure
    /// receives `From` and returns `Out` \u2014 infallible, no context, no async.
    /// Use `work` instead if the transformation can fail or requires I/O.
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

    /// Registers a flow-level suspend node. When a value of type `I` is present in
    /// state the flow pauses and surfaces it as [`FlowStep::Suspend`]. The caller
    /// resumes by calling [`FlowRuntime::resume`] with a value of type `O`, which is
    /// written into state under `O`'s schema name and the flow continues from there.
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

    /// Validates the graph structure and returns the [`FlowGraph`].
    ///
    /// The entry node is not set here — it is injected later by [`FlowRuntime`] via
    /// the crate-private `FlowGraph::with_entry`. Structural rules (duplicate nodes,
    /// bad model URL, same-type work, fork/join shape) are checked immediately.
    pub fn build(self) -> Result<FlowGraph, FlowError> {
        if !self.errors.is_empty() {
            return Err(BuildError::Invalid(self.errors).into());
        }
        validate_nodes(&self.flow.nodes, &self.flow)?;
        Ok(self.flow)
    }
}
