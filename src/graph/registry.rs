use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::history::{CompactionResult, HistoryEntry, MessageHistory};

use super::error::GraphError;
use super::ids::{EdgeId, HandlerKey};
use super::model::TypeSpec;
use super::value::Value;

/// Immutable execution information supplied to a synchronous continuation.
#[derive(Clone, Copy)]
pub struct ContinuationContext<'a> {
    execution_id: uuid::Uuid,
    history: &'a MessageHistory,
    output_validator: Option<&'a jsonschema::Validator>,
    history_policy: &'a crate::history::HistoryPolicy,
    loaded_keys: &'a std::collections::BTreeSet<String>,
}

impl<'a> ContinuationContext<'a> {
    pub(crate) fn new(
        execution_id: uuid::Uuid,
        history: &'a MessageHistory,
        output_validator: Option<&'a jsonschema::Validator>,
        history_policy: &'a crate::history::HistoryPolicy,
        loaded_keys: &'a std::collections::BTreeSet<String>,
    ) -> Self {
        Self {
            execution_id,
            history,
            output_validator,
            history_policy,
            loaded_keys,
        }
    }
    /// Returns the execution namespace used for deterministic identities.
    pub fn execution_id(&self) -> uuid::Uuid {
        self.execution_id
    }
    /// Borrows the sole runtime-owned history. Handlers return changes rather than mutating it.
    pub fn history(&self) -> &'a MessageHistory {
        self.history
    }

    pub(crate) fn output_validator(&self) -> Option<&'a jsonschema::Validator> {
        self.output_validator
    }
    pub(crate) fn history_policy(&self) -> &crate::history::HistoryPolicy {
        self.history_policy
    }
    pub(crate) fn loaded_keys(&self) -> &std::collections::BTreeSet<String> {
        self.loaded_keys
    }
}

/// Concrete history changes committed atomically with a continuation transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HistoryChange {
    /// Appends an acknowledged batch at its prevalidated stable positions.
    Append(Vec<HistoryEntry>),
    /// Replaces a validated completed prefix without changing cumulative usage.
    Compact {
        session_id: String,
        decision: CompactionResult,
    },
}

/// Intermediate edge write emitted by a continuation transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeWrite {
    /// Edge to write.
    pub edge: EdgeId,
    /// Value to write to the edge.
    pub value: Value,
}

/// Event delivered to an active continuation checkpoint.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "accepted completions stay inline without an extra allocation or ownership wrapper"
)]
pub enum ContinuationEvent {
    /// An external outcome already accepted durably by the runtime.
    Agent {
        request: super::agent_request::AgentRequest,
        response: super::agent_request::AgentResponse,
    },
    /// A child graph completed and produced an output for this call id.
    ChildResult { call_id: String, output: Value },
    /// External input supplied to a continuation-owned suspension.
    Resume { input: Value },
    /// No child result is pending; the handler may do more internal work.
    Poll,
}

/// External suspension requested by an active continuation handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContinuationTransition {
    /// External operation to emit with this checkpoint, mutually exclusive with suspension and child calls.
    pub agent: Option<super::agent_request::AgentRequest>,
    /// History edits validated and committed together with the VM transition.
    pub history: Vec<HistoryChange>,
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
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError>;

    /// Advances an active continuation checkpoint with a VM event.
    fn advance<'a>(
        &'a self,
        payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError>;
}

#[derive(Default, Clone)]
/// Runtime registry for all non-builtin handlers referenced by a graph.
///
/// Graphs store only handler keys; callers must provide the matching registry
/// before building an `Runtime`.
pub struct HandlerRegistry {
    value_handlers: HashMap<String, Arc<dyn ValueHandler>>,
    agent_handlers: HashMap<String, Arc<dyn super::agent_request::DynAgentHandler>>,
    continuation_handlers: HashMap<String, Arc<dyn ContinuationHandler>>,
}

impl HandlerRegistry {
    /// Registers one immutable implementation for VM transitions and external execution.
    pub(crate) fn insert_effect_continuation<H>(
        &mut self,
        key: &str,
        handler: H,
    ) -> Result<&mut Self, GraphError>
    where
        H: ContinuationHandler + super::agent_request::DynAgentHandler + 'static,
    {
        if self.continuation_handlers.contains_key(key) || self.agent_handlers.contains_key(key) {
            return Err(GraphError::GraphValidation(
                "duplicate effect handler key".into(),
            ));
        }
        let handler = Arc::new(handler);
        self.continuation_handlers
            .insert(key.into(), handler.clone());
        self.agent_handlers.insert(key.into(), handler);
        Ok(self)
    }
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

    /// Registers an external callback; it is never called by synchronous stepping.
    pub fn insert_agent<H: super::agent_request::DynAgentHandler + 'static>(
        &mut self,
        key: impl Into<String>,
        handler: H,
    ) -> Result<&mut Self, GraphError> {
        let key = key.into();
        if self.agent_handlers.contains_key(&key) {
            return Err(GraphError::GraphValidation(
                "duplicate agent worker handler key".into(),
            ));
        }
        self.agent_handlers.insert(key, Arc::new(handler));
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

    /// Resolves an external callback for the executor.
    pub fn agent(
        &self,
        key: &HandlerKey,
    ) -> Option<Arc<dyn super::agent_request::DynAgentHandler>> {
        self.agent_handlers.get(key.as_str()).cloned()
    }

    /// Resolves a continuation handler by graph key.
    pub fn continuation(&self, key: &HandlerKey) -> Option<Arc<dyn ContinuationHandler>> {
        self.continuation_handlers.get(key.as_str()).cloned()
    }

    /// Returns whether a value handler key is registered.
    pub fn has_value(&self, key: &str) -> bool {
        self.value_handlers.contains_key(key)
    }

    /// Returns whether a agent worker handler key is registered.
    pub fn has_agent(&self, key: &str) -> bool {
        self.agent_handlers.contains_key(key)
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
        for key in other.agent_handlers.keys() {
            if self.agent_handlers.contains_key(key) {
                return Err(GraphError::Invalid(format!(
                    "duplicate agent worker handler key '{key}'"
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
        self.agent_handlers.extend(
            other
                .agent_handlers
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
        self.agent_handlers.extend(
            other
                .agent_handlers
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
