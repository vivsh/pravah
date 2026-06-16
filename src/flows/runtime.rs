use std::sync::Arc;

use crate::{
    Context,
    clients::{DefaultClientFactory, Message, Role},
    flows::{
        ClientFactory, Flow, FlowError, FlowGraph, FlowHistory, FlowStep, NodeId,
        compactor::{DynHistoryCompactor, NoopCompactor},
        inspect::FlowInspector,
        memory::{DynMemoryFactory, NoopMemoryFactory},
        state::{AgentContinuation, Callable, FlowState},
        store::{DynHistoryStore, NoopHistoryStore},
    },
    tools::SuspendedValue,
};
use super::nodes::FlowNode;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

/// Converts an internal `FlowStep<Value>` into the public `FlowStep<T>`.
/// Only the `Done` payload is deserialized.
fn lift_step<T: DeserializeOwned>(step: FlowStep<Value>) -> Result<FlowStep<T>, FlowError> {
    match step {
        FlowStep::Continue => Ok(FlowStep::Continue),
        FlowStep::Done(val) => serde_json::from_value(val)
            .map(FlowStep::Done)
            .map_err(FlowError::Deserialize),
        FlowStep::Suspend(s) => Ok(FlowStep::Suspend(s)),
    }
}

/// Limits enforced by [`FlowRuntime::run_until`].
#[derive(Debug, Clone, Default)]
pub struct RunLimits {
    /// Stops after this many steps.
    pub max_steps: Option<usize>,
    /// Stops after this many model turns.
    pub max_turns: Option<usize>,
    /// Stops when the frame stack reaches this depth.
    pub max_depth: Option<usize>,
    /// Stops after this wall-clock duration.
    pub max_duration: Option<std::time::Duration>,
}

impl RunLimits {
    /// Creates an unconstrained limit set. Add constraints with the builder methods.
    pub fn new() -> Self {
        Self::default()
    }
    /// Stops the run after `n` engine steps regardless of how many LLM calls are made.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = Some(n);
        self
    }
    /// Stops the run after `n` model turns (assistant messages).
    pub fn max_turns(mut self, n: usize) -> Self {
        self.max_turns = Some(n);
        self
    }
    /// Stops the run when the nested-flow call stack reaches depth `n`.
    pub fn max_depth(mut self, n: usize) -> Self {
        self.max_depth = Some(n);
        self
    }
    /// Stops the run after the given wall-clock duration has elapsed.
    pub fn max_duration(mut self, d: std::time::Duration) -> Self {
        self.max_duration = Some(d);
        self
    }
}

/// Reason a `run_until` loop stopped before completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitKind {
    /// [`RunLimits::max_steps`] was reached.
    MaxSteps,
    /// [`RunLimits::max_turns`] was reached.
    MaxTurns,
    /// [`RunLimits::max_depth`] was reached.
    MaxDepth,
    /// [`RunLimits::max_duration`] elapsed.
    MaxDuration,
}

/// Outcome of [`FlowRuntime::run_until`].
#[derive(Debug)]
pub enum RunOutcome<T> {
    /// The flow finished normally and produced a value.
    Done(T),
    /// The flow hit a suspend point. Resume with [`FlowRuntime::resume`].
    Suspend(SuspendedValue),
    /// A [`RunLimits`] constraint was hit before the flow finished.
    LimitExceeded(LimitKind),
}

pub(crate) struct FlowCall(pub(crate) Arc<FlowGraph>);

/// Drives a [`Flow`] step by step, owning all execution state.
///
/// Create with [`FlowRuntime::new`] and advance with [`FlowRuntime::next`] or
/// [`FlowRuntime::run_until`]. Use [`FlowRuntime::inspector`] to observe state
/// between steps.
pub struct FlowRuntime<I: Flow> {
    state: FlowState,
    callables: Vec<FlowCall>,
    history: FlowHistory,
    factory: Arc<dyn ClientFactory>,
    memory: Arc<dyn DynMemoryFactory>,
    compactor: Box<dyn DynHistoryCompactor>,
    store: Box<dyn DynHistoryStore>,
    _marker: std::marker::PhantomData<I>,
}

impl<I: Flow> FlowRuntime<I> {
    fn build_graph() -> Result<FlowGraph, FlowError> {
        FlowGraph::from_flow::<I>()
    }

    /// Builds the graph and initialises execution state for the given input.
    /// Fails if the flow graph is invalid.
    pub fn new(flow: I) -> Result<Self, FlowError> {
        let graph: FlowGraph = Self::build_graph()?;
        let exit = graph.exit;

        let value = serde_json::to_value(&flow).map_err(FlowError::Serialize)?;
        let entry_id = graph.entry;

        let mut state = FlowState::new();
        let mut history = FlowHistory::new();
        history.push(
            "__root__",
            "__runtime__",
            Message::user(format!("Starting flow: {}", I::node_id())),
        );

        let (root_callable_index, callables) = Self::make_callables(graph)?;

        let callable = Callable {
            parent_entry: entry_id,
            parent_exit: exit,
            exit,
            entry: entry_id,
            index: root_callable_index,
            keep_alive: false,
        };

        state.call_enter(callable);
        state.set_state(entry_id, value, None);

        Ok(Self {
            state,
            callables,
            history,
            factory: Arc::new(DefaultClientFactory),
            memory: Arc::new(NoopMemoryFactory),
            compactor: Box::new(NoopCompactor),
            store: Box::new(NoopHistoryStore),
            _marker: std::marker::PhantomData,
        })
    }

    fn make_callables(mut graph: FlowGraph) -> Result<(usize, Vec<FlowCall>), FlowError> {
        let mut callables = vec![];
        Self::collect_callables(&mut graph, &mut callables)?;
        graph.callable_index = callables.len();
        let root_callable_index = graph.callable_index;
        callables.push(FlowCall(Arc::new(graph)));
        return Ok((root_callable_index, callables));
    }

    /// Collects callable sub-graphs so frames can jump to the correct graph at runtime.
    fn collect_callables(
        root: &mut FlowGraph,
        callables: &mut Vec<FlowCall>,
    ) -> Result<(), FlowError> {
        for i in 0..root.interner.rev.len() {
            let id = NodeId(i);
            if let Some(node) = root.nodes.get_mut(&id) {
                match node {
                    FlowNode::Flow(inner) => {
                        if let Some(inner_mut) = Arc::get_mut(inner) {
                            Self::collect_callables(inner_mut, callables)?;
                            inner_mut.callable_index = callables.len();
                            callables.push(FlowCall(inner.clone()));
                        } else {
                            return Err(FlowError::Internal {
                                handler: "collect_callables",
                                detail: "failed to get exclusive Arc reference to inner flow"
                                    .into(),
                            });
                        }
                    }
                    FlowNode::Each(info_arc) => {
                        if let Some(info_mut) = Arc::get_mut(info_arc) {
                            if let Some(inner_mut) = Arc::get_mut(&mut info_mut.inner) {
                                Self::collect_callables(inner_mut, callables)?;
                                let idx = callables.len();
                                inner_mut.callable_index = idx;
                                info_mut.callable_index = idx;
                                let inner_clone = info_mut.inner.clone();
                                callables.push(FlowCall(inner_clone));
                            } else {
                                return Err(FlowError::Internal {
                                    handler: "collect_callables",
                                    detail: "failed to get exclusive Arc reference to each inner flow".into(),
                                });
                            }
                        } else {
                            return Err(FlowError::Internal {
                                handler: "collect_callables",
                                detail: "failed to get exclusive Arc reference to EachInfo".into(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Replaces the default [`ClientFactory`].
    pub fn with_factory(mut self, factory: impl ClientFactory + 'static) -> Self {
        self.factory = Arc::new(factory);
        self
    }

    /// Sets a [`MemoryFactory`](crate::flows::memory::MemoryFactory) for dynamic context injection.
    ///
    /// The factory is called once per agent invocation (result cached for the
    /// lifetime of the invocation) to retrieve memories or other dynamic content
    /// that is prepended to the agent's system prompt between the static preamble
    /// and the input-schema hint.
    pub fn with_memory(
        mut self,
        memory: impl crate::flows::memory::MemoryFactory + Send + Sync + 'static,
    ) -> Self {
        self.memory = Arc::new(memory);
        self
    }

    /// Replaces the history compactor.
    pub fn with_compactor(
        mut self,
        c: impl crate::flows::compactor::HistoryCompactor + 'static,
    ) -> Self {
        self.compactor = Box::new(c);
        self
    }

    /// Replaces the history store.
    pub fn with_store(mut self, s: impl crate::flows::store::HistoryStore + 'static) -> Self {
        self.store = Box::new(s);
        self
    }

    /// Replaces the conversation history.
    /// Call this after [`FlowRuntime::from_snapshot`] when you need the old LLM context back.
    pub fn with_history(mut self, history: FlowHistory) -> Self {
        self.history = history;
        self
    }

    /// Returns a read-only view of the runtime state and history.
    pub fn inspector(&self) -> FlowInspector<'_> {
        FlowInspector::new(&self.state, &self.callables, &self.history)
    }

    /// Total number of LLM calls made across all agents since the flow started.
    /// Counts each `Assistant` and `AssistantToolCalls` history entry.
    pub fn agent_call_count(&self) -> usize {
        self.history
            .entries()
            .iter()
            .filter(|e| {
                !e.evicted
                    && matches!(
                        e.message.role,
                        Role::Assistant | Role::AssistantToolCalls { .. }
                    )
            })
            .count()
    }

    /// Injects a user message into the current agent session's history.
    /// Only call this when [`FlowInspector::is_agent_dispatch_ready`] returns `true`.
    /// Returns an error if no frame is active or no agent is at a dispatch boundary.
    pub fn inject_message(&mut self, content: impl Into<String>) -> Result<(), FlowError> {
        let frame = self.state.frames_slice().last().ok_or_else(|| FlowError::Internal {
            handler: "inject_message",
            detail: "no active frame".into(),
        })?;
        let callable = self.callables.get(frame.callable.index).ok_or_else(|| FlowError::Internal {
            handler: "inject_message",
            detail: format!("callable index {} out of range", frame.callable.index),
        })?;
        let (session_id, agent_name) = frame
            .agent_states
            .iter()
            .find_map(|(&agent_id, agent_state)| {
                matches!(agent_state.continuation, AgentContinuation::Dispatch).then(|| {
                    let name = callable.0.interner.name_of(agent_id).to_owned();
                    (agent_state.session_id.clone(), name)
                })
            })
            .ok_or_else(|| FlowError::Internal {
                handler: "inject_message",
                detail: "no agent is at a dispatch boundary; only call inject_message when is_agent_dispatch_ready() returns true".into(),
            })?;
        self.history.push(&session_id, &agent_name, Message::user(content));
        Ok(())
    }

    /// Advances the flow by one engine step.
    /// Returns [`FlowStep::Continue`] until the flow finishes or suspends.
    /// Fails with [`FlowError::ResumeRequired`] if the flow is already suspended.
    pub async fn next(&mut self, ctx: Context) -> Result<FlowStep<I::Output>, FlowError> {
        let factory = Arc::clone(&self.factory);
        let memory = Arc::clone(&self.memory);
        let callable_index = self
            .state
            .callable_index()
            .ok_or_else(|| FlowError::Internal {
                handler: "next",
                detail: "called after flow completed".into(),
            })?;
        let graph = self
            .callables
            .get(callable_index)
            .ok_or_else(|| FlowError::Internal {
                handler: "next",
                detail: format!("callable index {callable_index} out of range"),
            })?;
        tracing::debug!(
            flow = %I::node_id(),
            callable_index,
            depth = self.state.depth(),
            "runtime step"
        );
        let out = graph
            .0
            .next(factory.as_ref(), memory.as_ref(), ctx, &mut self.history, &mut self.state)
            .await?;
        self.run_compaction_and_flush().await;
        let step = lift_step(out)?;
        match &step {
            FlowStep::Continue => {}
            FlowStep::Done(_) => {
                tracing::info!(flow = %I::node_id(), "flow completed");
            }
            FlowStep::Suspend(_) => {
                tracing::info!(
                    flow = %I::node_id(),
                    output_type = ?self.state.suspension().map(|s| s.output_type.as_str()),
                    "flow suspended"
                );
            }
        }
        Ok(step)
    }

    /// Runs until the flow finishes, suspends, or hits a limit.
    pub async fn run_until(
        &mut self,
        ctx: Context,
        limits: RunLimits,
    ) -> Result<RunOutcome<I::Output>, FlowError> {
        let deadline = limits.max_duration.map(|d| std::time::Instant::now() + d);
        let mut steps = 0usize;
        let mut turns = 0usize;
        let log_limit = |limit: &str, steps: usize, depth: usize| {
            tracing::warn!(flow = %I::node_id(), limit, steps, depth, "run limit exceeded");
        };
        loop {
            if let Some(max) = limits.max_steps {
                if steps >= max {
                    log_limit("max_steps", steps, self.state.depth());
                    return Ok(RunOutcome::LimitExceeded(LimitKind::MaxSteps));
                }
            }
            if let Some(max) = limits.max_turns {
                if turns >= max {
                    log_limit("max_turns", steps, self.state.depth());
                    return Err(FlowError::LimitExceeded(format!(
                        "max_turns ({max}) exceeded"
                    )));
                }
            }
            if let Some(max) = limits.max_depth {
                if self.state.depth() >= max {
                    log_limit("max_depth", steps, self.state.depth());
                    return Ok(RunOutcome::LimitExceeded(LimitKind::MaxDepth));
                }
            }
            if let Some(dl) = deadline {
                if std::time::Instant::now() >= dl {
                    log_limit("max_duration", steps, self.state.depth());
                    return Ok(RunOutcome::LimitExceeded(LimitKind::MaxDuration));
                }
            }
            let was_dispatch = self.inspector().is_agent_dispatch_ready();
            steps += 1;
            match self.next(ctx.clone()).await? {
                FlowStep::Continue => {
                    if was_dispatch {
                        turns += 1;
                    }
                }
                FlowStep::Done(v) => return Ok(RunOutcome::Done(v)),
                FlowStep::Suspend(sv) => return Ok(RunOutcome::Suspend(sv)),
            }
        }
    }

    /// Provides the resumption value to a suspended flow and advances it one step.
    /// Fails with [`FlowError::UnexpectedResumption`] if the flow is not suspended,
    /// or [`FlowError::ResumptionTypeMismatch`] if `R` does not match the expected type.
    pub async fn resume<R: Serialize + JsonSchema>(
        &mut self,
        ctx: Context,
        value: R,
    ) -> Result<FlowStep<I::Output>, FlowError> {
        let suspension = self
            .state
            .suspension()
            .ok_or(FlowError::UnexpectedResumption)?;
        let expected = suspension.output_type.clone();
        let got: String = R::schema_name().into();
        if got != expected {
            return Err(FlowError::ResumptionTypeMismatch { expected, got });
        }
        let resumption = serde_json::to_value(value).map_err(FlowError::Serialize)?;
        let factory = Arc::clone(&self.factory);
        let memory = Arc::clone(&self.memory);
        let callable_index = self
            .state
            .callable_index()
            .ok_or_else(|| FlowError::Internal {
                handler: "resume",
                detail: "called after flow completed".into(),
            })?;
        let graph = self
            .callables
            .get(callable_index)
            .ok_or_else(|| FlowError::Internal {
                handler: "resume",
                detail: format!("callable index {callable_index} out of range"),
            })?;
        let out = graph
            .0
            .resume(
                factory.as_ref(),
                memory.as_ref(),
                ctx,
                &mut self.history,
                resumption,
                &mut self.state,
            )
            .await?;
        self.run_compaction_and_flush().await;
        lift_step(out)
    }

    /// Compacts each active session and then flushes the store.
    async fn run_compaction_and_flush(&mut self) {
        let session_ids: Vec<String> = self
            .state
            .active_session_ids()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for session_id in &session_ids {
            let owned: Vec<crate::flows::history::HistoryEntry> = self
                .history
                .session_entries(session_id)
                .into_iter()
                .cloned()
                .collect();
            let refs: Vec<&crate::flows::history::HistoryEntry> = owned.iter().collect();
            let result = self.compactor.compact_dyn(session_id, &refs).await;
            let _ = self.history.apply_compaction(session_id, &refs, result);
        }
        let _ = self.store.flush_dyn(&mut self.history).await;
    }

    /// Captures runtime state as a serializable [`FlowSnapshot`].
    /// History is stored separately.
    pub fn snapshot(&self) -> FlowSnapshot {
        FlowSnapshot {
            state: self.state.clone(),
        }
    }

    /// Rebuilds a runtime from a [`FlowSnapshot`].
    /// The graph is rebuilt from `I::build()`, so closures are never serialized.
    /// Re-attach history separately if you need the old conversation context.
    pub fn from_snapshot(snapshot: FlowSnapshot) -> Result<Self, FlowError> {
        let graph = Self::build_graph()?;
        let history = FlowHistory::new();

        let (_root_callable_index, callables) = Self::make_callables(graph)?;

        Ok(Self {
            state: snapshot.state,
            history,
            callables,
            factory: Arc::new(DefaultClientFactory),
            memory: Arc::new(NoopMemoryFactory),
            compactor: Box::new(NoopCompactor),
            store: Box::new(NoopHistoryStore),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<I: Flow> std::fmt::Debug for FlowRuntime<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowRuntime").finish_non_exhaustive()
    }
}

/// Serializable runtime state captured from a [`FlowRuntime`].
/// History is managed separately and can be re-attached with [`FlowRuntime::with_history`].
#[derive(Serialize, Deserialize)]
pub struct FlowSnapshot {
    state: FlowState,
}
