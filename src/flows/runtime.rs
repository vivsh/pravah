use std::sync::Arc;

use schemars::JsonSchema;
use crate::{
    Context,
    clients::{DefaultClientFactory, Message},
    flows::{
        ClientFactory, Flow, FlowError, FlowGraph, FlowHistory, FlowStep, NodeId, flows::FlowNode,
        state::{Callable, FlowState},
    },
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

fn new_session_id() -> String {
    Uuid::now_v7().to_string()
}

struct FlowCall(Arc<FlowGraph>);

pub struct FlowRuntime<I: Flow> {
    state: FlowState,
    callables: Vec<FlowCall>,
    history: FlowHistory,
    session_id: String,
    factory: Arc<dyn ClientFactory>,
    _marker: std::marker::PhantomData<I>,
}

impl<I: Flow> FlowRuntime<I> {

    fn build_graph() -> Result<FlowGraph, FlowError> {
        FlowGraph::from_flow::<I>()
    }

    pub fn new(flow: I) -> Result<Self, FlowError> {
        let graph: FlowGraph = Self::build_graph()?;
        let exit = graph.exit;

        let value = serde_json::to_value(&flow).map_err(|e| {
            FlowError::SerializeError(format!("start node '{}': {e}", I::node_id()))
        })?;
        let entry_id = graph.entry;

        let mut state = FlowState::new();
        // 0 is callable index as root is the first callable

        let mut history = FlowHistory::new(None);
        history.push(Message::user(format!("Starting flow: {}", I::node_id())));

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
            session_id: new_session_id(),
            factory: Arc::new(DefaultClientFactory),
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
                            return Err(FlowError::Internal(
                                "Failed to get mutable reference to inner flow".into(),
                            ));
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

    /// Replaces the conversation history used by agent nodes.
    ///
    /// Call after [`FlowRuntime::from_snapshot`] to restore the LLM context from a
    /// previously persisted [`FlowHistory`].
    pub fn with_history(mut self, history: FlowHistory) -> Self {
        self.history = history;
        self
    }

    pub async fn next(&mut self, ctx: Context) -> Result<FlowStep, FlowError> {
        let factory = Arc::clone(&self.factory);
        let callable_index = self.state.callable_index()
            .ok_or_else(|| FlowError::Internal("next() called after flow completed".into()))?;
        let graph = self.callables.get(callable_index)
            .ok_or_else(|| FlowError::Internal(format!("callable index {callable_index} out of range")))?;
        let out = graph
            .0
            .next(
                factory.as_ref(),
                ctx,
                &mut self.history,
                &self.session_id,
                &mut self.state,
            )
            .await?;
        Ok(out)
    }

    pub async fn resume(
        &mut self,
        ctx: Context,
        resumption: Value,
    ) -> Result<FlowStep, FlowError> {
        let factory = Arc::clone(&self.factory);
        let callable_index = self.state.callable_index()
            .ok_or_else(|| FlowError::Internal("resume() called after flow completed".into()))?;
        let graph = self.callables.get(callable_index)
            .ok_or_else(|| FlowError::Internal(format!("callable index {callable_index} out of range")))?;
        let out = graph
            .0
            .resume(
                factory.as_ref(),
                ctx,
                &mut self.history,
                &self.session_id,
                resumption,
                &mut self.state,
            )
            .await?;
        Ok(out)
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

        let (_root_callable_index, callables) = Self::make_callables(graph)?;

        Ok(Self {
            state: snapshot.state,
            history,
            callables,
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
