use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::flows::state::FlowState;
use crate::{
    clients::{
        ClientFactory, ClientOptions, DefaultClientFactory, Message, Role, ToolCall, ToolChoice,
    },
    clients::{ClientHistory, ClientOutput},
    commons::Agent,
    context::Context,
    tools::{ToolBox, ToolError},
};

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
}

pub(crate) enum AgentStep {
    Complete,
    Exit {
        value: serde_json::Value,
    },
    Suspend {
        value: serde_json::Value,
        tool_id: String,
        pending_calls: Vec<ToolCall>,
    },
}

struct AgentInfo {
    name: String,
    tool_box: ToolBox,
    preamble: String,
    model: String,
    exit_name: String,
    output_schema: Value,
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
}

pub(crate) enum FlowOut {
    Continue,
    Done(Value),
    Suspend { value: Value, tool_id: String },
}

/// Typed step result returned by [`FlowRuntime`].
#[derive(Debug)]
pub enum RunOut<O: std::fmt::Debug> {
    Continue,
    Done(O),
    Suspend { value: Value, tool_id: String },
}

pub trait Flow: 'static + JsonSchema + Serialize + DeserializeOwned + Send + Sync {
    type Output: std::fmt::Debug
        + JsonSchema
        + Serialize
        + DeserializeOwned
        + Send
        + Sync
        + 'static;

    fn build() -> FlowBuilder;

    fn node_id() -> String {
        Self::schema_name()
    }
}

pub struct FlowGraph {
    nodes: HashMap<String, FlowNode>,
    entry: String,
}

impl FlowGraph {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry: String::new(),
        }
    }

    pub fn builder() -> FlowBuilder {
        FlowBuilder::new()
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
            return Err(FlowError::BuildError(format!(
                "fork node '{}' produced {} child states but has {} child nodes",
                node.name,
                children.len(),
                node.children.len()
            )));
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

    async fn handle_tools(
        agent_id: &str,
        tool_box: &ToolBox,
        ctx: Context,
        calls: &[ToolCall],
        history: &mut ClientHistory,
        resumption: Option<(String, Value)>,
    ) -> Result<AgentStep, FlowError> {
        let mut resumed = false;
        for (i, call) in calls.iter().enumerate() {
            let tool_id = format!("{}::{}", agent_id, call.name);

            if !resumed {
                if let Some((resumption_id, resumption_value)) = &resumption {
                    if resumption_id != &tool_id {
                        return Err(FlowError::ResumeMismatchError(tool_id));
                    }
                    history.push(Message::tool_output(
                        call.id.clone(),
                        resumption_value.to_string(),
                    ));
                    resumed = true;
                    continue;
                }
            }

            match tool_box.call(call, ctx.clone()).await {
                Ok(output) => {
                    history.push(Message::tool_output(
                        output.call.id.clone(),
                        output.value.to_string(),
                    ));
                }
                Err(ToolError::Exit(value)) => {
                    history.push(Message::tool_output(call.id.clone(), value.to_string()));
                    return Ok(AgentStep::Exit { value });
                }
                Err(ToolError::Suspend(value)) => {
                    return Ok(AgentStep::Suspend {
                        value,
                        tool_id,
                        pending_calls: calls[i..].to_vec(),
                    });
                }
                Err(error) => {
                    return Err(FlowError::AgentError(format!("Tool error: {error}")));
                }
            }
        }

        Ok(AgentStep::Complete)
    }

    async fn handle_agent(
        agent: &AgentInfo,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut ClientHistory,
    ) -> Result<AgentStep, FlowError> {
        let options = if agent.tool_box.is_empty() {
            // No tools — use structured-output mode: pass the output schema and disable
            // tool calling so the client returns a typed JSON object directly.
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
            .map_err(|e| FlowError::AgentError(e.to_string()))?;
        let resp = client
            .execute(history.as_slice())
            .await
            .map_err(|e| FlowError::AgentError(e.to_string()))?;
        match resp.output {
            ClientOutput::ToolCalls { thought, calls } => {
                history.push(Message {
                    role: Role::AssistantToolCalls {
                        calls: calls.clone(),
                    },
                    content: thought.unwrap_or_default(),
                    usage: resp.usage,
                });
                Self::handle_tools(&agent.name, &agent.tool_box, ctx, &calls, history, None).await
            }
            ClientOutput::Output(value) => Ok(AgentStep::Exit { value }),
        }
    }

    /// Skips the LLM call when resuming from a suspension; injects the resume payload directly.
    async fn handle_resume(
        agent_id: &str,
        tool_box: &ToolBox,
        ctx: Context,
        history: &mut ClientHistory,
        pending_calls: &[ToolCall],
        payload: (String, Value),
    ) -> Result<AgentStep, FlowError> {
        Self::handle_tools(
            agent_id,
            tool_box,
            ctx,
            pending_calls,
            history,
            Some(payload),
        )
        .await
    }

    /// Pushes the initial user message exactly once per agent entry.
    fn handle_init(
        node_id: &str,
        states: &mut FlowState,
        history: &mut ClientHistory,
    ) -> Result<(), FlowError> {
        let initialized = states.agent_started();
        if !initialized {
            let state = states.get_state(node_id).ok_or_else(|| {
                FlowError::NotFound("initial state node has not produced a value".to_string())
            })?;
            history.push(Message::user(state.to_string()));
        }
        Ok(())
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

    fn validate(&self) -> Result<(), FlowError> {
        validate(&self.nodes, &self.entry)
    }

    pub(crate) async fn next(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut ClientHistory,
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        self.step(factory, ctx, history, None, states).await
    }

    pub(crate) async fn resume(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut ClientHistory,
        resumption: (String, Value),
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        self.step(factory, ctx, history, Some(resumption), states)
            .await
    }

    fn agent_step_into_flow_out(
        outcome: AgentStep,
        current_node: &str,
        exit_name: &str,
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        match outcome {
            AgentStep::Complete => Ok(FlowOut::Continue),
            AgentStep::Exit { value } => {
                states.agent_stopped();
                states.set_state(exit_name, value, Some(current_node));
                Ok(FlowOut::Continue)
            }
            AgentStep::Suspend {
                value,
                tool_id,
                pending_calls,
            } => {
                states.suspend(&tool_id, Some(pending_calls));
                Ok(FlowOut::Suspend { value, tool_id })
            }
        }
    }

    async fn step(
        &self,
        factory: &dyn ClientFactory,
        ctx: Context,
        history: &mut ClientHistory,
        resumption: Option<(String, Value)>,
        states: &mut FlowState,
    ) -> Result<FlowOut, FlowError> {
        // Handle suspend/resume consistency before doing any work
        match (states.suspension(), &resumption) {
            (Some(tool_id), None) => return Err(FlowError::ResumeRequired(tool_id.clone())),
            (None, Some((tool_id, _))) => {
                return Err(FlowError::UnexpectedResumption(tool_id.clone()));
            }
            _ => {}
        }

        let total_states = states.len();

        // Iteration over states is needed to support join nodes, which require checking all states for readiness.
        for state_index in 0..total_states {
            let current_node_id = states
                .get_index(state_index)
                .ok_or_else(|| {
                    FlowError::NotFound("current node has not produced a value".to_string())
                })?
                .0
                .clone();

            // Skip terminal states. This is also needed because join might still be waiting for them to produce values, and we don't want to error out on missing nodes.
            let current_node = match self.nodes.get(&current_node_id) {
                Some(n) => n,
                None => continue, // terminal state — skip it
            };

            if let Some(payload) = resumption {
                let pending = states
                    .take_pending_calls()
                    .ok_or_else(|| FlowError::AgentError("resume without pending calls".into()))?;

                let agent = match current_node {
                    FlowNode::Agent(a) => a,
                    _ => {
                        return Err(FlowError::AgentError(format!(
                            "resume on non-agent node '{}'",
                            current_node_id
                        )));
                    }
                };

                let outcome = Self::handle_resume(
                    &agent.name,
                    &agent.tool_box,
                    ctx,
                    history,
                    &pending,
                    payload,
                )
                .await?;
                states.clear_suspension();
                return Self::agent_step_into_flow_out(
                    outcome,
                    &current_node_id.clone(),
                    &agent.exit_name,
                    states,
                );
            }

            match current_node {
                FlowNode::Agent(agent) => {
                    Self::handle_init(&agent.name, states, history)?;
                    let outcome = Self::handle_agent(agent, factory, ctx, history).await?;
                    return Self::agent_step_into_flow_out(
                        outcome,
                        &current_node_id.clone(),
                        &agent.exit_name,
                        states,
                    );
                }
                FlowNode::Either(either) => {
                    Self::handle_either(either, ctx, states)?; // ctx moved
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
            }
        }

        if states.keys().all(|k| self.is_terminal(k)) {
            let value = states
                .keys()
                .next()
                .and_then(|k| states.get_state(k))
                .cloned()
                .unwrap_or(Value::Null);
            return Ok(FlowOut::Done(value));
        }
        let stuck: Vec<&str> = states
            .keys()
            .filter(|k| !self.is_terminal(k))
            .map(String::as_str)
            .collect();
        Err(FlowError::Deadlock(stuck.join(", ")))
    }
}

fn validate(nodes: &HashMap<String, FlowNode>, entry: &str) -> Result<(), FlowError> {
    let mut problems: Vec<String> = Vec::new();

    // --- Per-node rules ---
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
                // Both parent keys register the same join — only validate once per group.
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
                    if !nodes.contains_key(p.as_str()) {
                        problems.push(format!(
                            "join (target '{}'): parent '{}' is not a registered node",
                            info.target, p
                        ));
                    }
                }
                // target may be a terminal (absent from nodes) — that is valid
            }
            FlowNode::Either(info) => {
                if info.left_name == info.right_name {
                    problems.push(format!(
                        "either '{}': both branches resolve to the same schema name '{}'",
                        key, info.left_name
                    ));
                }
            }
        }
    }

    // --- Entry check ---
    if entry.is_empty() {
        problems.push(
            "flow has no entry node — call .entry(Self::node_id()) on the builder".to_string(),
        );
    } else if !nodes.contains_key(entry) {
        problems.push(format!("entry '{}' is not a registered node", entry));
    }

    // --- Reachability (only when entry is valid) ---
    if !entry.is_empty() && nodes.contains_key(entry) {
        // Map each node to the set of keys it produces output into.
        let successors: HashMap<&str, Vec<&str>> = nodes
            .iter()
            .map(|(key, node)| {
                let succs: Vec<&str> = match node {
                    FlowNode::Agent(info) => vec![info.exit_name.as_str()],
                    FlowNode::Work(info) => vec![info.exit_name.as_str()],
                    FlowNode::Fork(info) => info.children.iter().map(String::as_str).collect(),
                    FlowNode::Join(info) => vec![info.target.as_str()],
                    FlowNode::Either(info) => {
                        vec![info.left_name.as_str(), info.right_name.as_str()]
                    }
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
        Err(FlowError::Invalid(problems))
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

    /// Sets the entry node key. Required for graph validation. Typically called as
    /// `.entry(Self::node_id())` at the end of `Flow::build()`.
    pub fn entry(mut self, key: impl Into<String>) -> Self {
        self.flow.entry = key.into();
        self
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
        let agent_info = AgentInfo {
            name: name.clone(),
            tool_box: A::tool_box().with_agent::<A>(),
            preamble: A::preamble(),
            model: A::model_url(),
            exit_name: A::Output::schema_name(),
            output_schema,
        };
        self.flow.nodes.insert(name, FlowNode::Agent(agent_info));
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

    /// Builds the [`FlowGraph`], validating it before returning.
    pub(crate) fn build(self) -> Result<FlowGraph, FlowError> {
        if !self.errors.is_empty() {
            return Err(FlowError::Invalid(self.errors));
        }
        self.flow.validate()?;
        Ok(self.flow)
    }
}

pub struct FlowRuntime<I: Flow> {
    state: FlowState,
    graph: FlowGraph,
    history: ClientHistory,
    factory: Arc<dyn ClientFactory>,
    _marker: std::marker::PhantomData<I>,
}

impl<I: Flow> FlowRuntime<I> {
    pub fn new(flow: I) -> Result<Self, FlowError> {
        let graph = I::build().build()?;
        let value = serde_json::to_value(&flow).map_err(|e| {
            FlowError::SerializeError(format!("start node '{}': {e}", I::node_id()))
        })?;
        let mut state = FlowState::new();
        state.set_state(&I::node_id(), value, None);
        let mut history = ClientHistory::new(None);
        history.push(Message::user(format!("Starting flow: {}", I::node_id())));
        Ok(Self {
            state,
            graph,
            history,
            factory: Arc::new(DefaultClientFactory),
            _marker: std::marker::PhantomData,
        })
    }

    /// Overrides the default [`ClientFactory`]. Useful for testing or selecting a provider.
    pub fn with_factory(mut self, factory: impl ClientFactory + 'static) -> Self {
        self.factory = Arc::new(factory);
        self
    }

    pub async fn next(&mut self, ctx: Context) -> Result<RunOut<I::Output>, FlowError> {
        let factory = Arc::clone(&self.factory);
        let out = self
            .graph
            .next(factory.as_ref(), ctx, &mut self.history, &mut self.state)
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
}

impl<I: Flow> std::fmt::Debug for FlowRuntime<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowRuntime").finish_non_exhaustive()
    }
}
