use std::sync::Arc;

use crate::{
    Context,
    clients::{DefaultClientFactory, Message},
    flows::{
        compactor::{DynHistoryCompactor, NoopCompactor},
        store::{DynHistoryStore, NoopHistoryStore},
        ClientFactory, Flow, FlowError, FlowGraph, FlowHistory, FlowStep, NodeId, flows::FlowNode,
        state::{Callable, FlowState},
    },
    tools::SuspendedValue,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

/// Converts an internal `FlowStep<Value>` to the public `FlowStep<T>` by deserializing
/// the `Done` payload. `Continue` and `Suspend` pass through unchanged.
fn lift_step<T: DeserializeOwned>(step: FlowStep<Value>) -> Result<FlowStep<T>, FlowError> {
    match step {
        FlowStep::Continue => Ok(FlowStep::Continue),
        FlowStep::Done(val) => {
            serde_json::from_value(val).map(FlowStep::Done).map_err(FlowError::Deserialize)
        }
        FlowStep::Suspend(s) => Ok(FlowStep::Suspend(s)),
    }
}

/// Limits for [`FlowRuntime::run_until`]. All fields are optional; unset fields are unchecked.
#[derive(Debug, Clone, Default)]
pub struct RunLimits {
    /// Maximum number of [`FlowRuntime::next`] calls.
    pub max_steps: Option<usize>,
    /// Maximum number of LLM turns. Currently shares the step counter; will track
    /// API calls independently once per-agent instrumentation is added.
    pub max_turns: Option<usize>,
    /// Maximum frame-stack depth (guards against pathologically deep sub-flow nesting).
    pub max_depth: Option<usize>,
    /// Maximum wall-clock time before returning [`LimitKind::MaxDuration`].
    pub max_duration: Option<std::time::Duration>,
}

impl RunLimits {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = Some(n);
        self
    }
    pub fn max_turns(mut self, n: usize) -> Self {
        self.max_turns = Some(n);
        self
    }
    pub fn max_depth(mut self, n: usize) -> Self {
        self.max_depth = Some(n);
        self
    }
    pub fn max_duration(mut self, d: std::time::Duration) -> Self {
        self.max_duration = Some(d);
        self
    }
}

/// Reason a [`FlowRuntime::run_until`] loop was interrupted before completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitKind {
    MaxSteps,
    MaxTurns,
    MaxDepth,
    MaxDuration,
}

/// Outcome of [`FlowRuntime::run_until`].
#[derive(Debug)]
pub enum RunOutcome<T> {
    Done(T),
    Suspend(SuspendedValue),
    LimitExceeded(LimitKind),
}

struct FlowCall(Arc<FlowGraph>);

pub struct FlowRuntime<I: Flow> {
    state: FlowState,
    callables: Vec<FlowCall>,
    history: FlowHistory,
    factory: Arc<dyn ClientFactory>,
    compactor: Box<dyn DynHistoryCompactor>,
    store: Box<dyn DynHistoryStore>,
    _marker: std::marker::PhantomData<I>,
}

impl<I: Flow> FlowRuntime<I> {

    fn build_graph() -> Result<FlowGraph, FlowError> {
        FlowGraph::from_flow::<I>()
    }

    pub fn new(flow: I) -> Result<Self, FlowError> {
        let graph: FlowGraph = Self::build_graph()?;
        let exit = graph.exit;

        let value = serde_json::to_value(&flow).map_err(FlowError::Serialize)?;
        let entry_id = graph.entry;

        let mut state = FlowState::new();
        // 0 is callable index as root is the first callable

        let mut history = FlowHistory::new();
        history.push("__root__", "__runtime__", Message::user(format!("Starting flow: {}", I::node_id())));

        let (root_callable_index, callables) = Self::make_callables(graph)?;

        let callable = Callable{
            parent_entry: entry_id,
            parent_exit: exit,
            exit,
            entry: entry_id,
            index: root_callable_index,
        };

        state.call_enter(callable);
        state.set_state(entry_id, value, None);

        Ok(Self {
            state,
            callables,
            history,
            factory: Arc::new(DefaultClientFactory),
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

    // walk the graph to collect all AgentInfo and inner FlowGraph for later dispatch during step execution
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
                                detail: "failed to get exclusive Arc reference to inner flow".into(),
                            });
                        }
                    }
                    FlowNode::FlowTool { inner, .. } => {
                        if let Some(inner_mut) = Arc::get_mut(inner) {
                            Self::collect_callables(inner_mut, callables)?;
                            inner_mut.callable_index = callables.len();
                            callables.push(FlowCall(inner.clone()));
                        } else {
                            return Err(FlowError::Internal {
                                handler: "collect_callables",
                                detail: "failed to get exclusive Arc reference to flow tool inner graph".into(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Overrides the default [`ClientFactory`]. Useful for testing or selecting a provider.
    pub fn with_factory(mut self, factory: impl ClientFactory + 'static) -> Self {
        self.factory = Arc::new(factory);
        self
    }

    /// Sets the [`HistoryCompactor`] used to evict old turns after each step.
    pub fn with_compactor(mut self, c: impl crate::flows::compactor::HistoryCompactor + 'static) -> Self {
        self.compactor = Box::new(c);
        self
    }

    /// Sets the [`HistoryStore`] used to persist history after each step.
    pub fn with_store(mut self, s: impl crate::flows::store::HistoryStore + 'static) -> Self {
        self.store = Box::new(s);
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

    pub async fn next(&mut self, ctx: Context) -> Result<FlowStep<I::Output>, FlowError> {
        let factory = Arc::clone(&self.factory);
        let callable_index = self.state.callable_index()
            .ok_or_else(|| FlowError::Internal { handler: "next", detail: "called after flow completed".into() })?;
        let graph = self.callables.get(callable_index)
            .ok_or_else(|| FlowError::Internal { handler: "next", detail: format!("callable index {callable_index} out of range") })?;
        let out = graph
            .0
            .next(
                factory.as_ref(),
                ctx,
                &mut self.history,
                &mut self.state,
            )
            .await?;
        self.run_compaction_and_flush().await;
        lift_step(out)
    }

    /// Drives the flow to completion, checking `limits` before each step.
    ///
    /// Returns [`RunOutcome::Done`] when the flow completes, [`RunOutcome::Suspend`] when
    /// the flow suspends waiting for external input, or [`RunOutcome::LimitExceeded`] when
    /// any configured limit is reached.
    pub async fn run_until(
        &mut self,
        ctx: Context,
        limits: RunLimits,
    ) -> Result<RunOutcome<I::Output>, FlowError> {
        let deadline = limits.max_duration.map(|d| std::time::Instant::now() + d);
        let mut steps = 0usize;
        loop {
            if let Some(max) = limits.max_steps {
                if steps >= max {
                    return Ok(RunOutcome::LimitExceeded(LimitKind::MaxSteps));
                }
            }
            if let Some(max) = limits.max_turns {
                if steps >= max {
                    return Ok(RunOutcome::LimitExceeded(LimitKind::MaxTurns));
                }
            }
            if let Some(max) = limits.max_depth {
                if self.state.depth() >= max {
                    return Ok(RunOutcome::LimitExceeded(LimitKind::MaxDepth));
                }
            }
            if let Some(dl) = deadline {
                if std::time::Instant::now() >= dl {
                    return Ok(RunOutcome::LimitExceeded(LimitKind::MaxDuration));
                }
            }
            steps += 1;
            match self.next(ctx.clone()).await? {
                FlowStep::Continue => {}
                FlowStep::Done(v) => return Ok(RunOutcome::Done(v)),
                FlowStep::Suspend(sv) => return Ok(RunOutcome::Suspend(sv)),
            }
        }
    }

    pub async fn resume<R: Serialize + JsonSchema>(
        &mut self,
        ctx: Context,
        value: R,
    ) -> Result<FlowStep<I::Output>, FlowError> {
        let suspension = self.state.suspension()
            .ok_or(FlowError::UnexpectedResumption)?;
        let expected = suspension.output_type.clone();
        let got: String = R::schema_name().into();
        if got != expected {
            return Err(FlowError::ResumptionTypeMismatch { expected, got });
        }
        let resumption = serde_json::to_value(value).map_err(FlowError::Serialize)?;
        let factory = Arc::clone(&self.factory);
        let callable_index = self.state.callable_index()
            .ok_or_else(|| FlowError::Internal { handler: "resume", detail: "called after flow completed".into() })?;
        let graph = self.callables.get(callable_index)
            .ok_or_else(|| FlowError::Internal { handler: "resume", detail: format!("callable index {callable_index} out of range") })?;
        let out = graph
            .0
            .resume(
                factory.as_ref(),
                ctx,
                &mut self.history,
                resumption,
                &mut self.state,
            )
            .await?;
        self.run_compaction_and_flush().await;
        lift_step(out)
    }

    /// Runs per-session compaction then calls the store's flush.
    async fn run_compaction_and_flush(&mut self) {
        let session_ids: Vec<String> = self
            .state
            .active_session_ids()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for session_id in &session_ids {
            // Clone entries so we can release the immutable borrow before calling apply_compaction.
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
        let history = FlowHistory::new();

        let (_root_callable_index, callables) = Self::make_callables(graph)?;

        Ok(Self {
            state: snapshot.state,
            history,
            callables,
            factory: Arc::new(DefaultClientFactory),
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
