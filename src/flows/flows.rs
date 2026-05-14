use std::collections::HashMap;
use std::sync::Arc;

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::history::FlowHistory;
use crate::flows::NodeId;
use crate::flows::errors::{BuildError, FlowError};
use crate::flows::interner::Interner;
use crate::flows::phase::Phase;
use crate::flows::state::{Callable, FlowState};
use crate::flows::validation::{validate, validate_nodes};
use crate::{
    clients::{ClientFactory, ClientOptions, ClientOutput, Message, Role, ToolCall},
    commons::Agent,
    context::Context,
    tools::{ToolBox, ToolError},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct StateNode {
    name: String,
    value: serde_json::Value,
}

pub(crate) struct AgentInfo {
    pub(crate) id: NodeId,
    tool_box: Arc<ToolBox>,
    preamble: String,
    pub(crate) model: String,
    pub(crate) exit: NodeId,
    output_schema: Value,
}

/// Metadata for a single tool exposed by an agent, stored as a graph node so the
/// flow engine can dispatch it independently rather than inlining all tool calls.
pub(crate) struct ToolInfo {
    /// State-map key used for this tool: interned `"AgentName::tool_name"`.
    pub(crate) name: NodeId,
    /// Agent state key to return to after the tool completes (agent's input key).
    agent_name: NodeId,
    /// Agent output key — written on `ToolError::Exit` to signal the frame is done.
    agent_exit: NodeId,
    /// Zero-based index into `tool_box.tools`.
    tool_index: usize,
    /// Shared toolbox owned by the parent agent.
    tool_box: Arc<ToolBox>,
}

pub(crate) struct EitherInfo {
    pub(crate) name: NodeId,
    pub(crate) left_name: NodeId,
    pub(crate) right_name: NodeId,
    func: Box<dyn Fn(&Value, Context) -> Result<(NodeId, Value), FlowError> + Send + Sync>,
}

pub(crate) struct ForkInfo {
    pub(crate) name: NodeId,
    pub(crate) children: Vec<NodeId>,
    func: Box<dyn Fn(&Value, Context) -> Result<Vec<StateNode>, FlowError> + Send + Sync>,
}

pub(crate) struct JoinInfo {
    pub(crate) parents: Vec<NodeId>,
    pub(crate) target: NodeId,
    func: Arc<dyn Fn(&[Value], Context) -> Result<StateNode, FlowError> + Send + Sync>,
}

pub(crate) struct WorkInfo {
    pub(crate) name: NodeId,
    pub(crate) exit_name: NodeId,
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

pub(crate) enum FlowNode {
    Agent(Arc<AgentInfo>),
    Either(EitherInfo),
    Fork(ForkInfo),
    Join(JoinInfo),
    Work(WorkInfo),
    /// An embedded sub-flow. Boxed to break the recursive type-size cycle.
    Flow(Arc<FlowGraph>),
    /// A single tool dispatched by a parent agent. Not statically reachable from the
    /// entry node; it enters the state map when the agent issues a tool call.
    Tool(ToolInfo),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AgentContinuation {
    Dispatch,       // history loaded, ready to dispatch to LLM
    PendingTool,    // tool calls issued; waiting for tool nodes to complete
}

/// Typed step result returned by [`FlowRuntime`].
#[derive(Debug)]
pub enum FlowStep {
    Continue,
    Done(Value),
    Suspend { value: Value },
}

pub trait Flow: 'static + JsonSchema + Serialize + DeserializeOwned + Send + Sync {
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;

    fn build() -> Result<FlowGraph, FlowError>;

    fn node_id() -> String {
        Self::schema_name()
    }
}

pub struct FlowGraph {
    pub(crate) nodes: HashMap<NodeId, FlowNode>,

    pub(crate) entry: NodeId,    
    pub(crate) exit: NodeId,

    pub (crate) parent_exit: Option<NodeId>,

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
            return Err(FlowError::Internal("handle_work: frame stack empty".into()));
        }
        Ok(())
    }

    fn handle_fork(node: &ForkInfo, ctx: Context, states: &mut FlowState) -> Result<(), FlowError> {
        let state = states.get_state(node.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "fork parent '{}' has not produced a value",
                node.name.0
            ))
        })?;

        let children = (node.func)(state, ctx)?;
        if children.len() != node.children.len() {
            return Err(BuildError::NodeConflict(format!(
                "fork node '{}' produced {} child states but has {} child nodes",
                node.name.0,
                children.len(),
                node.children.len()
            ))
            .into());
        }
        for (child_node, &child_id) in children.iter().zip(&node.children) {
            if !states.set_state(child_id, child_node.value.clone(), None) {
                return Err(FlowError::Internal("handle_fork: frame stack empty".into()));
            }
        }
        if !states.remove_state(node.name) {
            return Err(FlowError::Internal("handle_fork: frame stack empty on remove".into()));
        }

        Ok(())
    }

    fn handle_join(node: &JoinInfo, ctx: Context, states: &mut FlowState) -> Result<(), FlowError> {
        let mut inputs = Vec::with_capacity(node.parents.len());
        for &p in &node.parents {
            let value = states.get_state(p).ok_or_else(|| {
                FlowError::NotFound(format!("join parent '{}' has not produced a value", p.0))
            })?;
            inputs.push(value.clone());
        }
        let output = (node.func)(&inputs, ctx)?;
        if !states.set_state(node.target, output.value, None) {
            return Err(FlowError::Internal("handle_join: frame stack empty".into()));
        }

        for &p in &node.parents {
            if !states.remove_state(p) {
                return Err(FlowError::Internal("handle_join: frame stack empty on remove".into()));
            }
        }

        Ok(())
    }

    fn handle_either(
        either: &EitherInfo,
        ctx: Context,
        states: &mut FlowState,
    ) -> Result<(), FlowError> {
        let state = states.get_state(either.name).ok_or_else(|| {
            FlowError::NotFound(format!(
                "either parent '{}' has not produced a value",
                either.name.0
            ))
        })?;
        let (out_id, out_val) = (either.func)(&state, ctx)?;
        if !states.set_state(out_id, out_val, Some(either.name)) {
            return Err(FlowError::Internal("handle_either: frame stack empty".into()));
        }
        Ok(())
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
        session_id: &str,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        self.step(factory, ctx, history, session_id, None, states)
            .await
    }

    pub(crate) async fn resume(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        session_id: &str,
        resumption: Value,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        self.step(factory, ctx, history, session_id, Some(resumption), states)
            .await
    }

    fn handle_tool<'a>(
        node: &'a ToolInfo,
        flow: &'a FlowGraph,
        ctx: Context,
        history: &'a mut FlowHistory,
        session_id: &'a str,
        states: &'a mut FlowState,
    ) -> BoxFuture<'a, Result<FlowStep, FlowError>> {
        Box::pin(async move {
            let call_json = states
                .get_state(node.name)
                .ok_or_else(|| {
                    FlowError::NotFound(format!(
                        "tool '{}': call state missing",
                        flow.interner.name_of(node.name),
                    ))
                })?
                .clone();
            let call: ToolCall = serde_json::from_value(call_json)
                .map_err(|e| FlowError::DeserializeError(format!("tool call: {e}")))?;
            match node.tool_box.call_at_index(node.tool_index, &call.id, call.args.clone(), ctx).await {
                Ok(output) => {
                    history.push(
                        Message::tool_output(call.id, output.value.to_string())
                            .with_context(&call.name, session_id),
                    );
                    if !states.remove_state(node.name) {
                        return Err(FlowError::Internal("handle_tool: frame stack empty on remove".into()));
                    }
                    Ok(FlowStep::Continue)
                }
                Err(ToolError::Exit(value)) => {
                    Self::handle_tool_exit(node, flow, &call, value, history, session_id, states)
                }
                Err(ToolError::Suspend) => {
                    // Keep ToolCall JSON in src so `step` can recover call.id on resume.
                    states.suspend(node.name, node.agent_name);
                    Ok(FlowStep::Suspend { value: call.args })
                }
                Err(e) => Err(FlowError::AgentError(format!("tool '{}': {e}", call.name))),
            }
        })
    }

    /// Handles the `submit` sentinel (`ToolError::Exit`).
    ///
    /// - Records the structured output as a tool-result history entry.
    /// - Cancels sibling tool calls from the same LLM turn that have not yet executed,
    ///   writing a `"cancelled"` result for each so history stays well-formed for all providers.
    /// - Writes the output value to `agent_exit` and marks the agent frame done via
    ///   `Phase::Continue(None)`.
    fn handle_tool_exit(
        node: &ToolInfo,
        flow: &FlowGraph,
        call: &ToolCall,
        value: Value,
        history: &mut FlowHistory,
        session_id: &str,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let content = serde_json::to_string(&value)
            .map_err(|e| FlowError::SerializeError(e.to_string()))?;
        history.push(
            Message::tool_output(call.id.clone(), content)
                .with_context(&call.name, session_id),
        );
        let siblings: Vec<(NodeId, Value)> = states
            .keys()
            .filter(|&k| {
                matches!(flow.nodes.get(&k), Some(FlowNode::Tool(t)) if t.agent_name == node.agent_name)
            })
            .filter_map(|k| states.get_state(k).map(|v| (k, v.clone())))
            .collect();
        for (k, call_json) in siblings {
            let sibling: ToolCall = serde_json::from_value(call_json)
                .map_err(|e| FlowError::DeserializeError(format!("sibling tool call: {e}")))? ;
            history.push(
                Message::tool_output(sibling.id, "cancelled: submit was issued".to_string())
                    .with_context(&sibling.name, session_id),
            );
            if !states.remove_state(k) {
                return Err(FlowError::Internal("handle_tool_exit: frame stack empty on sibling remove".into()));
            }
        }
        if !states.set_state(node.agent_exit, value, None) {
            return Err(FlowError::Internal("handle_tool_exit: frame stack empty on set_state".into()));
        }
        if !states.set_phase(Phase::Continue(None)) {
            return Err(FlowError::Internal("handle_tool_exit: frame stack empty on set_phase".into()));
        }
        Ok(FlowStep::Continue)
    }

    fn handle_parent_agent(
        node: &AgentInfo,
        flow: &FlowGraph,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let parent_entry = node.id;
        let parent_exit = node.exit;

        let callable = Callable{
            parent_entry,
            parent_exit,
            exit: parent_exit,
            entry: parent_entry,
            index: flow.callable_index, 
        };

        states.call_enter(callable);
        if !states.set_phase(Phase::Entry) {
            return Err(FlowError::Internal("handle_parent_agent: frame stack empty after call_enter".into()));
        }

        Ok(FlowStep::Continue)
    }

    async fn handle_child_agent(
        node: &AgentInfo,
        flow: &FlowGraph,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        session_id: &str,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let phase = states.phase().cloned().ok_or_else(|| {
            FlowError::Internal("handle_child_agent called without a phase".to_string())
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
                    Message::from_json(Role::User, &input)
                        .map_err(|e| FlowError::SerializeError(e.to_string()))?
                        .with_context(agent_name, session_id),
                );

                // Advance phase: Entry is consumed once the user message is in history.
                // Any future re-entry will find Continue(Dispatch) and skip re-pushing.
                let dispatch_val = serde_json::to_value(AgentContinuation::Dispatch)
                    .map_err(|e| FlowError::SerializeError(e.to_string()))?;
                if !states.set_phase(Phase::Continue(Some(dispatch_val))) {
                    return Err(FlowError::Internal("handle_child_agent Entry: frame stack empty on set_phase".into()));
                }

                Self::dispatch_agent(node, flow, factory, history, session_id, states).await
            }

            Phase::Continue(val) => {
                let cont = match val {
                    Some(v) => Some(
                        serde_json::from_value::<AgentContinuation>(v)
                            .map_err(|e| FlowError::DeserializeError(format!("agent continuation: {e}")))?,
                    ),
                    None => None,
                };
                match cont {
                    Some(AgentContinuation::PendingTool) => {
                        // Move the agent key to the end of the state map so step_inner
                        // reaches tool states (at lower indices) before re-entering this agent.
                        let input_val = states.get_state(node.id).cloned();
                        if !states.remove_state(node.id) {
                            return Err(FlowError::Internal("handle_child_agent PendingTool: frame stack empty on remove".into()));
                        }
                        if let Some(v) = input_val {
                            if !states.set_state(node.id, v, None) {
                                return Err(FlowError::Internal("handle_child_agent PendingTool: frame stack empty on re-insert".into()));
                            }
                        }
                        // If tool states still occupy the map, let step_inner dispatch them first.
                        let any_tools = states
                            .keys()
                            .any(|k| matches!(flow.nodes.get(&k), Some(FlowNode::Tool(_))));
                        if any_tools {
                            return Ok(FlowStep::Continue);
                        }
                        // All tool results are now in history — re-dispatch to LLM.
                        Self::dispatch_agent(node, flow, factory, history, session_id, states).await
                    }
                    // Explicit Dispatch: call LLM.
                    Some(AgentContinuation::Dispatch) => {
                        Self::dispatch_agent(node, flow, factory, history, session_id, states).await
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
        session_id: &str,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let agent_name = flow.interner.name_of(node.id).to_string();

        let options = ClientOptions::default()
            .with_preamble(node.preamble.clone())
            .with_tools(node.tool_box.definitions())
            .with_output_schema(node.output_schema.clone())
            .with_name(agent_name.clone());

        let client = factory
            .create(&node.model, options)
            .map_err(|e| FlowError::AgentError(e.to_string()))?;

        history
            .validate()
            .map_err(|e| FlowError::AgentError(e.to_string()))?;

        let response = client
            .execute(history.as_slice())
            .await
            .map_err(|e| FlowError::AgentError(e.to_string()))?;

        match response.output {
            ClientOutput::Output(val) => {
                let content = serde_json::to_string(&val)
                    .map_err(|e| FlowError::SerializeError(e.to_string()))?;
                let mut msg = Message::assistant(content).with_context(&agent_name, session_id);
                if let Some(usage) = response.usage {
                    msg = msg.with_usage(usage);
                }
                history.push(msg);

                // Retire the input slot, write the output slot.
                if !states.set_state(node.exit, val, Some(node.id)) {
                    return Err(FlowError::Internal("dispatch_agent Output: frame stack empty on set_state".into()));
                }
                // Phase::Continue(None): agent is done; call_exit will pop this frame.
                if !states.set_phase(Phase::Continue(None)) {
                    return Err(FlowError::Internal("dispatch_agent Output: frame stack empty on set_phase".into()));
                }

                Ok(FlowStep::Continue)
            }

            ClientOutput::ToolCalls { thought, calls } => {
                history.push(Message {
                    role: Role::AssistantToolCalls { calls: calls.clone() },
                    content: thought.unwrap_or_default(),
                    usage: response.usage,
                    agent_id: agent_name.clone(),
                    session_id: session_id.to_owned(),
                });

                // Validate: reject unknown tool names and duplicate calls.
                let mut seen_ids = std::collections::HashSet::new();
                for call in &calls {
                    let tool_key = format!("{}::{}", agent_name, call.name);
                    let tool_id = flow.interner.fwd.get(&tool_key).copied().ok_or_else(|| {
                        FlowError::AgentError(format!(
                            "agent '{}': LLM called unknown tool '{}'",
                            agent_name, call.name
                        ))
                    })?;
                    if !seen_ids.insert(tool_id) {
                        return Err(FlowError::AgentError(format!(
                            "agent '{}': duplicate tool call '{}'",
                            agent_name, call.name
                        )));
                    }
                }

                // Insert each pending tool call under its interned NodeId.
                for call in &calls {
                    let tool_key = format!("{}::{}", agent_name, call.name);
                    let tool_id = *flow.interner.fwd.get(&tool_key).unwrap(); // safe: validated above
                    let call_val = serde_json::to_value(call)
                        .map_err(|e| FlowError::SerializeError(e.to_string()))?;
                    if !states.set_state(tool_id, call_val, None) {
                        return Err(FlowError::Internal("dispatch_agent ToolCalls: frame stack empty on set_state".into()));
                    }
                }

                if !states.set_phase(Phase::Continue(Some(
                    serde_json::to_value(AgentContinuation::PendingTool)
                        .map_err(|e| FlowError::SerializeError(e.to_string()))?,
                ))) {
                    return Err(FlowError::Internal("dispatch_agent ToolCalls: frame stack empty on set_phase".into()));
                }

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
        session_id: &str,
        states: &mut FlowState,
    ) -> Result<FlowStep, FlowError> {
        let total_states = states.len();
        for state_index in 0..total_states {
            let current_node_id = states
                .get_index(state_index)
                .ok_or_else(|| {
                    FlowError::NotFound("current node has not produced a value".to_string())
                })?
                .0;
            let current_node = match self.nodes.get(&current_node_id) {
                Some(n) => n,
                None => continue,
            };
            match current_node {
                FlowNode::Agent(agent) => {
                    if let Some(_phase) = states.phase() {
                        return Self::handle_child_agent(agent, self, factory, ctx, history, session_id, states).await;
                    } else {
                        return Self::handle_parent_agent(agent, self, states);
                    }
                }
                FlowNode::Tool(info) => return Self::handle_tool(info, self, ctx, history, session_id, states).await,
                FlowNode::Either(either) => {
                    Self::handle_either(either, ctx, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Fork(info) => {
                    Self::handle_fork(info, ctx, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Join(info) => {
                    if !self.can_join(current_node_id, states) {
                        continue;
                    }
                    Self::handle_join(info, ctx, states)?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Work(info) => {
                    Self::handle_work(info, ctx, states).await?;
                    return Ok(FlowStep::Continue);
                }
                FlowNode::Flow(inner) => {
                    let parent_exit = inner.parent_exit.ok_or_else(|| {
                        FlowError::Internal("inner flow missing parent exit".to_string())
                    })?;

                    let callable = Callable{
                        parent_entry: current_node_id,
                        parent_exit,
                        exit: inner.exit,
                        entry: inner.entry,
                        index: inner.callable_index,
                    };  

                    states.call_enter(callable);

                    return Ok(FlowStep::Continue);
                }
            }
        }

        Ok(FlowStep::Continue)
    }

    /// Entry point for a step. Resolves the active inner graph from the frame stack,
    /// then delegates to [`step_inner`].
    async fn step(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut FlowHistory,
        session_id: &str,
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
            // If resuming from a tool suspension, write the tool result to history
            // before the suspension slot (which holds the original ToolCall JSON) is consumed.
            if let Some(susp) = states.suspension() {
                if let Some(call_json) = states.get_state(susp.src) {
                    if let Ok(tc) = serde_json::from_value::<ToolCall>(call_json.clone()) {
                        history.push(
                            Message::tool_output(tc.id, resumption.to_string())
                                .with_context(&tc.name, session_id),
                        );
                    }
                }
            }
            if !states.resume(resumption) {
                return Err(FlowError::Internal(
                    "resume: no active suspension or empty frame stack".into(),
                ));
            }
        }
        let result = self.step_inner(factory, ctx, history, session_id, states).await?;
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
        let tool_box = Arc::new(config.tool_box.with_agent::<A>());
        let output_str = A::Output::schema_name();
        let output_id = self.flow.interner.intern(&output_str);
        let agent_info = AgentInfo {
            id: name,
            tool_box: Arc::clone(&tool_box),
            preamble: config.preamble,
            model: config.model_url,
            exit: output_id,
            output_schema,
        };
        self.flow
            .nodes
            .insert(name, FlowNode::Agent(Arc::new(agent_info)));
        // Register a FlowNode::Tool for each tool in the toolbox (including the submit sentinel).
        for i in 0..tool_box.len() {
            let tool_name_str = format!("{}::{}", name_str, tool_box.name_at(i));
            let tool_name = self.flow.interner.intern(&tool_name_str);
            self.flow.nodes.insert(
                tool_name,
                FlowNode::Tool(ToolInfo {
                    name: tool_name,
                    agent_name: name,
                    agent_exit: output_id,
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
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("either '{}': duplicate node key", from_id_str));
            return self;
        }
        let left_name = self.flow.interner.intern(&A::schema_name());
        let right_name = self.flow.interner.intern(&B::schema_name());
        let from_id_clone = from_id_str.clone();
        let shim: Box<dyn Fn(&Value, Context) -> Result<(NodeId, Value), FlowError> + Send + Sync> =
            Box::new(move |value: &Value, ctx: Context| {
                let typed: From = serde_json::from_value(value.clone()).map_err(|e| {
                    FlowError::DeserializeError(format!(
                        "transition from '{}': {e}",
                        from_id_clone.clone()
                    ))
                })?;
                match func(typed, ctx)? {
                    Either::Left(a) => {
                        let v = serde_json::to_value(&a)
                            .map_err(|e| FlowError::SerializeError(format!("either left: {e}")))?;
                        Ok((left_name, v))
                    }
                    Either::Right(b) => {
                        let v = serde_json::to_value(&b)
                            .map_err(|e| FlowError::SerializeError(format!("either right: {e}")))?;
                        Ok((right_name, v))
                    }
                }
            });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Either(EitherInfo {
                name: from_id,
                left_name,
                right_name,
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
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("fork '{}': duplicate node key", from_id_str));
            return self;
        }
        let from_id_clone = from_id_str.clone();
        let shim: Box<dyn Fn(&Value, Context) -> Result<Vec<StateNode>, FlowError> + Send + Sync> =
            Box::new(move |value: &Value, ctx: Context| {
                let typed: From = serde_json::from_value(value.clone()).map_err(|e| {
                    FlowError::DeserializeError(format!("fork from '{}': {e}", from_id_clone))
                })?;
                let (a, b) = func(typed, ctx)?;
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
    /// combines them into `Out`, and clears the parent states.
    pub fn join<A, B, Out, H>(mut self, func: H) -> Self
    where
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(A, B, Context) -> Result<Out, FlowError> + Send + Sync + 'static,
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
        let a_id_inner = a_id_str.clone();
        let b_id_inner = b_id_str.clone();
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
        let from_id_clone = from_id_str.clone();
        let exit_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Box<
            dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync,
        > = Box::new(move |value: &Value, ctx: Context| {
            let typed: From = match serde_json::from_value(value.clone()) {
                Ok(v) => v,
                Err(e) => {
                    let err =
                        FlowError::DeserializeError(format!("work from '{}': {e}", from_id_clone));
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
            from_id,
            FlowNode::Work(WorkInfo {
                name: from_id,
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
        validate_nodes(&self.flow.nodes, &self.flow)?;
        Ok(self.flow)
    }
}
