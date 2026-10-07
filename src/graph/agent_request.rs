//! Self-contained agent work descriptions; routing never requires a mutable runtime.

use serde::{Deserialize, Serialize};
use std::{fmt, sync::Arc};
use uuid::Uuid;

use super::{HandlerKey, Value};
use crate::clients::{Message, TokenUsage};
use crate::history::HistoryEntry;

pub(crate) mod client_response;
mod executor;
mod failure;
pub(crate) mod options;
mod response;
pub use executor::{AgentExecutor, DynAgentHandler};
pub use response::{AgentError, AgentResponse};

/// Version of the direct agent request protocol, independent of Rath's implementation.
pub const AGENT_REQUEST_VERSION: u32 = 2;

/// Serializable inputs for one externally scheduled agent operation.
/// Runtime dependencies, credentials supplied through Context, and task scheduling
/// metadata are not attached by Pravah. Supplied payloads may themselves be sensitive.
#[derive(Clone, Serialize)]
pub struct AgentRequest {
    version: u32,
    id: Uuid,
    pub(crate) persist: Option<Arc<[HistoryEntry]>>,
    #[serde(flatten)]
    pub(crate) operation: Arc<AgentOperation>,
}

/// Private operation data. Public inspection borrows only the requested routing fields.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum AgentOperation {
    Configure {
        execution_id: Uuid,
        handler: HandlerKey,
        definition: Value,
        input: Value,
        load_history: bool,
        loaded_keys: std::collections::BTreeSet<String>,
    },
    Control {
        handler: HandlerKey,
        definition: Value,
        observation: Value,
    },
    Tool {
        handler: HandlerKey,
        input: Value,
    },
    Generate {
        session_id: String,
        model: String,
        options: Value,
        entries: Vec<HistoryEntry>,
        guidance: Vec<Message>,
        budget_conclusion: bool,
        compact: bool,
        last_usage: Option<TokenUsage>,
        total_input: Option<u32>,
        total_output: Option<u32>,
    },
    PersistHistory,
}

/// Constructs shared protocol objects without allocating static field names.
pub(crate) fn object<const N: usize>(
    fields: [(&'static str, Value); N],
) -> Result<Value, super::GraphError> {
    Value::from_shared_object(
        fields
            .into_iter()
            .map(|(key, value)| (std::borrow::Cow::Borrowed(key), value))
            .collect(),
    )
    .map_err(|error| super::GraphError::AgentRequestValidation(error.to_string()))
}

impl AgentRequest {
    /// Constructs immutable work at the VM's checked request boundary.
    pub(crate) fn new(id: Uuid, operation: AgentOperation) -> Self {
        Self {
            version: AGENT_REQUEST_VERSION,
            id,
            persist: None,
            operation: Arc::new(operation),
        }
    }

    /// Assigns the runtime's deterministic identity before exposing a request.
    pub(crate) fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }
    /// Returns the agent protocol version checked when this request was decoded.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the stable identity used for completion recording and delivery.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returns the operation discriminator without inspecting or decoding payloads.
    pub fn kind(&self) -> &'static str {
        match self.operation.as_ref() {
            AgentOperation::Configure { .. } => "configure",
            AgentOperation::Control { .. } => "control",
            AgentOperation::Tool { .. } => "tool",
            AgentOperation::Generate { .. } => "generate",
            AgentOperation::PersistHistory => "persist_history",
        }
    }

    /// Borrows the model locator for generation or request-aware compaction.
    /// Configuration and tool operations do not necessarily have a resolved model.
    pub fn model(&self) -> Option<&str> {
        match self.operation.as_ref() {
            AgentOperation::Generate { model, .. } => Some(model),
            _ => None,
        }
    }

    /// Borrows the logical provider scheme, not a resolved factory or network endpoint.
    /// Returns None when no model is known or its locator lacks a scheme.
    pub fn provider(&self) -> Option<&str> {
        self.model()?
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .filter(|scheme| !scheme.is_empty())
    }

    /// Borrows the registered Rust handler identity when this operation requires one.
    pub fn handler(&self) -> Option<&HandlerKey> {
        match self.operation.as_ref() {
            AgentOperation::Configure { handler, .. }
            | AgentOperation::Control { handler, .. }
            | AgentOperation::Tool { handler, .. } => Some(handler),
            _ => None,
        }
    }

    /// Borrows the application conversation key when one is resolved.
    /// A persistence batch spanning several sessions has no single routing key.
    pub fn conversation_key(&self) -> Option<&str> {
        let session = match self.operation.as_ref() {
            AgentOperation::Generate { session_id, .. } => session_id.as_str(),
            AgentOperation::PersistHistory => {
                let entries = self.persist.as_deref()?;
                let first = entries.first()?;
                if entries
                    .iter()
                    .any(|entry| entry.session_id != first.session_id)
                {
                    return None;
                }
                &first.session_id
            }
            _ => return None,
        };
        session.strip_prefix("key:")
    }
}

impl<'de> Deserialize<'de> for AgentRequest {
    /// Checks the outer protocol version before accepting operation data.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            version: u32,
            id: Uuid,
            persist: Option<Arc<[HistoryEntry]>>,
            #[serde(flatten)]
            operation: AgentOperation,
        }
        let wire = Wire::deserialize(deserializer)?;
        if wire.version != AGENT_REQUEST_VERSION {
            return Err(serde::de::Error::custom(
                "unsupported agent request version",
            ));
        }
        Ok(Self {
            version: wire.version,
            id: wire.id,
            persist: wire.persist,
            operation: Arc::new(wire.operation),
        })
    }
}

impl fmt::Debug for AgentRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentRequest")
            .field("id", &self.id)
            .field("operation", &self.kind())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "tests/agent_request.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/agent_options.rs"]
mod options_tests;
