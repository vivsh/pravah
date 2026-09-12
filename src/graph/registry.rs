use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::Context;
use crate::clients::{ClientOptions, Message};
use crate::history::{
    DynHistoryPreparer, DynHistoryStore, FlowHistory, HistoryEntry, HistoryPreparation,
    HistoryPreparer, HistoryStore, NoopHistoryStore, count_complete_turns, protected_start,
    validate_message_groups,
};

use super::error::GraphError;
use super::ids::{EdgeId, HandlerKey};
use super::model::TypeSpec;
use super::value::Value;

#[derive(Clone)]
/// Runtime-owned service bundle shared by edge handlers.
///
/// Configure history behavior through `Runtime::with_history_preparer` and
/// `Runtime::with_store`; graph serialization never contains these services.
pub struct RuntimeServices {
    preparer: Option<Arc<dyn DynHistoryPreparer>>,
    store: Arc<dyn DynHistoryStore>,
}

impl Default for RuntimeServices {
    fn default() -> Self {
        Self {
            preparer: None,
            store: Arc::new(NoopHistoryStore),
        }
    }
}

impl RuntimeServices {
    /// Creates services without a preparation policy and with a no-op history store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs the fallible policy run before each model execution, never after output.
    pub fn with_history_preparer(mut self, preparer: impl HistoryPreparer + 'static) -> Self {
        self.preparer = Some(Arc::new(preparer));
        self
    }

    /// Replaces the history store used to record appended history entries.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.store = Arc::new(store);
        self
    }

    pub(crate) fn store(&self) -> &dyn DynHistoryStore {
        self.store.as_ref()
    }
}

#[derive(Clone)]
/// Context passed to continuation handlers while they advance.
///
/// It carries the execution's bound `Context`, runtime services, and controlled
/// history access without smuggling services into `Context::deps()`.
pub struct ContinuationContext {
    ctx: Context,
    services: Arc<RuntimeServices>,
    history: Arc<Mutex<FlowHistory>>,
}

impl ContinuationContext {
    pub(crate) fn new(
        ctx: Context,
        services: Arc<RuntimeServices>,
        history: Arc<Mutex<FlowHistory>>,
    ) -> Self {
        Self {
            ctx,
            services,
            history,
        }
    }

    /// Returns the ordinary per-call request context.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Returns runtime-owned services available to continuation handlers.
    pub fn services(&self) -> &RuntimeServices {
        self.services.as_ref()
    }

    /// Appends one message to runtime history after recording it in the store.
    pub async fn push_history(
        &self,
        session_id: &str,
        agent_id: &str,
        message: Message,
    ) -> Result<(), GraphError> {
        self.push_history_batch(session_id, agent_id, vec![message])
            .await
    }

    /// Appends a validated message batch without exposing partial in-memory history.
    ///
    /// Stores must deduplicate retries by the stable entry position. A store can
    /// observe a prefix when a later write fails, but the runtime commits the
    /// batch only after every entry has been accepted.
    pub async fn push_history_batch(
        &self,
        session_id: &str,
        agent_id: &str,
        messages: Vec<Message>,
    ) -> Result<(), GraphError> {
        let mut history = self.history.lock().await;
        let mut staged = history.clone();
        let mut entries = Vec::with_capacity(messages.len());
        for message in messages {
            let entry = staged.prepare_entry(session_id, agent_id, message);
            staged.commit_entry(entry.clone());
            entries.push(entry);
        }
        for entry in &entries {
            self.services
                .store()
                .record_dyn(entry)
                .await
                .map_err(|err| GraphError::HistoryPersistence(err.to_string()))?;
        }
        *history = staged;
        Ok(())
    }

    /// Returns live messages for one history session.
    pub async fn history_for_session(&self, session_id: &str) -> Vec<Message> {
        self.history.lock().await.for_session(session_id)
    }

    /// Returns live history rows for one session.
    pub async fn history_entries_for_session(&self, session_id: &str) -> Vec<HistoryEntry> {
        self.history
            .lock()
            .await
            .session_entries(session_id)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Validates that one history session is provider-safe to dispatch.
    pub async fn validate_history_for_session(&self, session_id: &str) -> Result<(), GraphError> {
        self.history
            .lock()
            .await
            .validate_for_session(session_id)
            .map_err(|err| GraphError::Invalid(format!("agent history is invalid: {err}")))
    }

    /// Counts complete model turns in one history session.
    pub async fn complete_turn_count(&self, session_id: &str) -> usize {
        let entries = self.history_entries_for_session(session_id).await;
        let refs = entries.iter().collect::<Vec<_>>();
        count_complete_turns(&refs)
    }

    /// Prepares and atomically replaces completed session history for one model request.
    /// The returned messages are exactly the successfully prepared history for dispatch.
    pub(crate) async fn prepare_history(
        &self,
        session_id: &str,
        model: &str,
        options: &ClientOptions,
        framework_messages: &[Message],
    ) -> Result<Vec<Message>, GraphError> {
        let invalid = |reason| GraphError::HistoryPreparationValidation {
            session_id: session_id.to_owned(),
            reason,
        };
        let Some(preparer) = &self.services.preparer else {
            let messages = self.history_for_session(session_id).await;
            validate_message_groups(&messages).map_err(invalid)?;
            return Ok(messages);
        };
        let owned = self.history_entries_for_session(session_id).await;
        let refs = owned.iter().collect::<Vec<_>>();
        validate_message_groups(refs.iter().map(|entry| &entry.message)).map_err(invalid)?;
        let (committed, protected) = refs.split_at(protected_start(&refs));
        let request = HistoryPreparation {
            session_id,
            model,
            options,
            framework_messages,
            committed,
            protected,
        };
        let result = preparer.prepare_dyn(request).await?;
        self.history
            .lock()
            .await
            .replace_prepared_history(session_id, &refs, result)
            .map_err(invalid)
    }
}

/// Intermediate edge write emitted by a continuation transition.
#[derive(Debug, Clone)]
pub struct EdgeWrite {
    /// Edge to write.
    pub edge: EdgeId,
    /// Value to write to the edge.
    pub value: Value,
}

/// Event delivered to an active continuation checkpoint.
#[derive(Debug, Clone)]
pub enum ContinuationEvent {
    /// A child graph completed and produced an output for this call id.
    ChildResult { call_id: String, output: Value },
    /// External input supplied to a continuation-owned suspension.
    Resume { input: Value },
    /// No child result is pending; the handler may do more internal work.
    Poll,
}

/// External suspension requested by an active continuation handler.
#[derive(Debug, Clone)]
pub struct ContinuationSuspension {
    /// Expected resume type and schema exposed at invocation boundaries.
    pub resume_type: TypeSpec,
    /// Serializable payload returned to the external caller.
    pub payload: Value,
}

/// Child graph invocation requested by a continuation transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationChildCall {
    /// Index into the continuation node's embedded child graph list.
    pub child_index: usize,
    /// Stable call id returned later with the child result.
    pub call_id: String,
    /// Value written to the child graph entry edge.
    pub input: Value,
}

/// Result of starting/advancing a generic multi-step continuation node.
#[derive(Debug, Clone, Default)]
pub struct ContinuationTransition {
    /// Serialized checkpoint to keep the continuation active.
    pub checkpoint: Option<Value>,
    /// Optional opaque handler state stored beside the checkpoint.
    pub state: Option<Value>,
    /// Completion outputs written to the node output edges.
    pub outputs: Vec<Value>,
    /// Extra edge writes emitted before completion or continuation.
    pub writes: Vec<EdgeWrite>,
    /// Child graph calls requested by this transition.
    pub child_calls: Vec<ContinuationChildCall>,
    /// Optional external suspension owned by this continuation checkpoint.
    pub suspension: Option<ContinuationSuspension>,
}

/// Synchronous value handler for pure edge transforms.
pub trait ValueHandler: Send + Sync {
    /// Converts input values into output values.
    fn call(&self, inputs: Vec<Value>) -> Result<Vec<Value>, GraphError>;
}

impl<F> ValueHandler for F
where
    F: Fn(Vec<Value>) -> Result<Vec<Value>, GraphError> + Send + Sync,
{
    fn call(&self, inputs: Vec<Value>) -> Result<Vec<Value>, GraphError> {
        self(inputs)
    }
}

/// Async one-shot handler for work nodes.
///
/// Use this for operations that complete within one VM dispatch.
pub trait WorkHandler: Send + Sync {
    /// Runs the work node with its input values and request context.
    fn call<'a>(
        &'a self,
        inputs: Vec<Value>,
        ctx: Context,
    ) -> BoxFuture<'a, Result<Vec<Value>, GraphError>>;
}

impl<F> WorkHandler for F
where
    F: Fn(Vec<Value>, Context) -> BoxFuture<'static, Result<Vec<Value>, GraphError>> + Send + Sync,
{
    fn call<'a>(
        &'a self,
        inputs: Vec<Value>,
        ctx: Context,
    ) -> BoxFuture<'a, Result<Vec<Value>, GraphError>> {
        Box::pin(self(inputs, ctx))
    }
}

/// Multi-step handler for continuation nodes.
///
/// Use this for agents, external protocols, or other state machines that need
/// checkpoints, polling, or child graph calls.
pub trait ContinuationHandler: Send + Sync {
    /// Validates serialized payload metadata against this registered handler.
    ///
    /// Implementations with payload-bound runtime capabilities should reject a
    /// graph whose serialized declaration does not match those capabilities.
    fn validate_payload(&self, _payload: &Value) -> Result<(), GraphError> {
        Ok(())
    }

    /// Starts the continuation from ready input values.
    fn start<'a>(
        &'a self,
        payload: &'a Value,
        state: Option<Value>,
        inputs: Vec<Value>,
        ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>>;

    /// Advances an active continuation checkpoint with a VM event.
    fn advance<'a>(
        &'a self,
        payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>>;
}

#[derive(Default, Clone)]
/// Runtime registry for all non-builtin handlers referenced by a graph.
///
/// Graphs store only handler keys; callers must provide the matching registry
/// before building an `Runtime`.
pub struct HandlerRegistry {
    value_handlers: HashMap<String, Arc<dyn ValueHandler>>,
    work_handlers: HashMap<String, Arc<dyn WorkHandler>>,
    continuation_handlers: HashMap<String, Arc<dyn ContinuationHandler>>,
}

impl HandlerRegistry {
    /// Creates an empty handler registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a pure value handler under a unique key.
    pub fn insert_value<H>(
        &mut self,
        key: impl Into<String>,
        handler: H,
    ) -> Result<&mut Self, GraphError>
    where
        H: ValueHandler + 'static,
    {
        let key = key.into();
        if self.value_handlers.contains_key(&key) {
            return Err(GraphError::Invalid(format!(
                "duplicate value handler key '{key}'"
            )));
        }
        self.value_handlers.insert(key, Arc::new(handler));
        Ok(self)
    }

    /// Registers a one-shot async work handler under a unique key.
    pub fn insert_work<H>(
        &mut self,
        key: impl Into<String>,
        handler: H,
    ) -> Result<&mut Self, GraphError>
    where
        H: WorkHandler + 'static,
    {
        let key = key.into();
        if self.work_handlers.contains_key(&key) {
            return Err(GraphError::Invalid(format!(
                "duplicate work handler key '{key}'"
            )));
        }
        self.work_handlers.insert(key, Arc::new(handler));
        Ok(self)
    }

    /// Registers a multi-step continuation handler under a unique key.
    pub fn insert_continuation<H>(
        &mut self,
        key: impl Into<String>,
        handler: H,
    ) -> Result<&mut Self, GraphError>
    where
        H: ContinuationHandler + 'static,
    {
        let key = key.into();
        if self.continuation_handlers.contains_key(&key) {
            return Err(GraphError::Invalid(format!(
                "duplicate continuation handler key '{key}'"
            )));
        }
        self.continuation_handlers.insert(key, Arc::new(handler));
        Ok(self)
    }

    /// Resolves a pure value handler by graph key.
    pub fn value(&self, key: &HandlerKey) -> Option<Arc<dyn ValueHandler>> {
        self.value_handlers.get(key.as_str()).cloned()
    }

    /// Resolves an async work handler by graph key.
    pub fn work(&self, key: &HandlerKey) -> Option<Arc<dyn WorkHandler>> {
        self.work_handlers.get(key.as_str()).cloned()
    }

    /// Resolves a continuation handler by graph key.
    pub fn continuation(&self, key: &HandlerKey) -> Option<Arc<dyn ContinuationHandler>> {
        self.continuation_handlers.get(key.as_str()).cloned()
    }

    /// Returns whether a value handler key is registered.
    pub fn has_value(&self, key: &str) -> bool {
        self.value_handlers.contains_key(key)
    }

    /// Returns whether a work handler key is registered.
    pub fn has_work(&self, key: &str) -> bool {
        self.work_handlers.contains_key(key)
    }

    /// Returns whether a continuation handler key is registered.
    pub fn has_continuation(&self, key: &str) -> bool {
        self.continuation_handlers.contains_key(key)
    }

    /// Merges another registry, rejecting duplicate keys within a handler class.
    pub fn extend_from(&mut self, other: &Self) -> Result<(), GraphError> {
        for key in other.value_handlers.keys() {
            if self.value_handlers.contains_key(key) {
                return Err(GraphError::Invalid(format!(
                    "duplicate value handler key '{key}'"
                )));
            }
        }
        for key in other.work_handlers.keys() {
            if self.work_handlers.contains_key(key) {
                return Err(GraphError::Invalid(format!(
                    "duplicate work handler key '{key}'"
                )));
            }
        }
        for key in other.continuation_handlers.keys() {
            if self.continuation_handlers.contains_key(key) {
                return Err(GraphError::Invalid(format!(
                    "duplicate continuation handler key '{key}'"
                )));
            }
        }

        self.value_handlers.extend(
            other
                .value_handlers
                .iter()
                .map(|(key, handler)| (key.clone(), Arc::clone(handler))),
        );
        self.work_handlers.extend(
            other
                .work_handlers
                .iter()
                .map(|(key, handler)| (key.clone(), Arc::clone(handler))),
        );
        self.continuation_handlers.extend(
            other
                .continuation_handlers
                .iter()
                .map(|(key, handler)| (key.clone(), Arc::clone(handler))),
        );
        Ok(())
    }

    pub(crate) fn extend_namespaced(&mut self, prefix: &str, other: &Self) {
        self.value_handlers.extend(
            other
                .value_handlers
                .iter()
                .map(|(key, handler)| (namespaced_handler_key(prefix, key), Arc::clone(handler))),
        );
        self.work_handlers.extend(
            other
                .work_handlers
                .iter()
                .map(|(key, handler)| (namespaced_handler_key(prefix, key), Arc::clone(handler))),
        );
        self.continuation_handlers.extend(
            other
                .continuation_handlers
                .iter()
                .map(|(key, handler)| (namespaced_handler_key(prefix, key), Arc::clone(handler))),
        );
    }
}

fn namespaced_handler_key(prefix: &str, key: &str) -> String {
    format!("{prefix}::{key}")
}
