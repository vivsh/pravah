use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::Context;
use crate::history::{Compactor, HistoryStore};

use super::agent::Agent;
use super::error::GraphError;
use super::ids::{NodeId, VarId};
use super::runtime::{Runtime, Snapshot};
use super::state::Step;
use super::typed::build_chat_graph;
use super::value::{Value, from_value, to_value};

mod builder;
mod request;
#[cfg(test)]
mod tests;
pub use builder::ChatBuilder;
pub use request::ChatRequest;

type InputValidator<I> = fn(&I, &Value) -> Result<(), GraphError>;

/// A Chat input and optional application key carried by ordinary VM values.
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatSubmission<I> {
    pub(crate) input: I,
    pub(crate) key: Option<String>,
}

impl ChatSubmission<Value> {
    /// Checks the envelope while retaining shared invocation values for typed decoding.
    pub(crate) fn decode(value: &Value) -> Result<Self, super::value::ValueError> {
        use super::value::ValueError;
        let invalid = || ValueError::Unsupported("invalid chat submission envelope".into());
        let mut fields = value.object_entries().ok_or_else(invalid)?;
        if fields.any(|(key, _)| key != "input" && key != "key") {
            return Err(invalid());
        }
        let input = value.get("input").ok_or_else(invalid)?.clone();
        let key = match value.get("key") {
            None => None,
            Some(value) if value.is_null() => None,
            Some(value) => Some(value.as_str().ok_or_else(invalid)?.to_owned()),
        };
        Ok(Self { input, key })
    }
}

/// One assistant response produced by graph chat.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTurn<O> {
    /// Decoded assistant response.
    pub output: O,
}

impl<O> ChatTurn<O> {
    /// Consumes the turn and returns the response value.
    pub fn into_output(self) -> O {
        self.output
    }
}

/// Typed conversation and application state backed by one graph runtime.
///
/// Builder chats always retain history; function-defined agents must enable
/// `keep_alive`. Application state persists independently and is never added to prompts.
pub struct Chat<I, O, S = ()> {
    runtime: Runtime,
    state_var: VarId,
    input_boundaries: [NodeId; 2],
    validate_input: Option<InputValidator<I>>,
    _marker: PhantomData<fn(I, S) -> O>,
}

impl<I, O> Chat<I, O>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Creates a snapshot-ready chat with unit application state and fixed context.
    ///
    /// Graph validation may fail; construction executes no agent or external work.
    pub async fn new(
        agent: fn(Agent<I>) -> Agent<O>,
        context: Context,
    ) -> Result<Self, GraphError> {
        Self::with_state(agent, (), context).await
    }
}

impl<I, O, S> Chat<I, O, S>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Creates a snapshot-ready chat with initial application state stored in the VM.
    ///
    /// Graph validation and state conversion can fail. Only the initial input
    /// suspension executes; no agent configuration, history or external calls occur.
    pub async fn with_state(
        agent: fn(Agent<I>) -> Agent<O>,
        state: S,
        context: Context,
    ) -> Result<Self, GraphError> {
        Self::from_definition(agent(Agent::root()), state, context, None).await
    }

    /// Initializes the common graph for function-defined and builder-created chats.
    async fn from_definition(
        agent: Agent<O>,
        state: S,
        context: Context,
        validate_input: Option<InputValidator<I>>,
    ) -> Result<Self, GraphError> {
        let value = encode_state(state)?;
        let (prepared, state_var, input_boundaries) = build_chat_graph::<I, O, S>(agent)?;
        let mut runtime = prepared.start(Value::NULL, context)?;
        runtime.write_chat_state(state_var, value)?;
        let [bootstrap, _] = input_boundaries;
        if !matches!(runtime.next().await?, Step::Suspend(_)) || !runtime.chat_ready(&[bootstrap]) {
            return Err(GraphError::Invalid(
                "chat did not reach its initial input boundary".into(),
            ));
        }
        Ok(Self {
            runtime,
            state_var,
            input_boundaries,
            validate_input,
            _marker: PhantomData,
        })
    }

    /// Restores execution and application state using freshly supplied runtime context.
    ///
    /// Validates the graph, VM state and decoding into `S` without running handlers.
    /// Reattach history policies and stores after restoration. Old chat graphs are rejected.
    pub fn from_snapshot(
        agent: fn(Agent<I>) -> Agent<O>,
        snapshot: Snapshot,
        context: Context,
    ) -> Result<Self, GraphError> {
        Self::restore_definition(agent(Agent::root()), snapshot, context, None)
    }

    /// Restores without calling application handlers and checks retained submission values.
    fn restore_definition(
        agent: Agent<O>,
        snapshot: Snapshot,
        context: Context,
        validate_input: Option<InputValidator<I>>,
    ) -> Result<Self, GraphError> {
        let (prepared, state_var, input_boundaries) = build_chat_graph::<I, O, S>(agent)?;
        let runtime = prepared.restore(snapshot, context)?;
        let chat = Self {
            runtime,
            state_var,
            input_boundaries,
            validate_input,
            _marker: PhantomData,
        };
        chat.get().map_err(|error| {
            GraphError::SnapshotValidation(format!("chat state is invalid: {error}"))
        })?;
        chat.validate_restored_inputs()?;
        Ok(chat)
    }

    /// Decodes application state from the VM, including during an unfinished turn.
    ///
    /// Decoding may allocate and fail; no typed copy is retained by the chat.
    pub fn get(&self) -> Result<S, GraphError> {
        from_value(self.runtime.chat_state(self.state_var)?).map_err(|error| {
            GraphError::ValueConversion {
                target: "chat state".into(),
                reason: error.to_string(),
            }
        })
    }

    /// Replaces application state between turns, including before the first message.
    ///
    /// Conversion, graph shape validation or epoch failure leaves state unchanged.
    /// An unfinished turn rejects mutation; state is never automatically exposed to tools.
    pub fn set(&mut self, state: S) -> Result<(), GraphError> {
        self.require_ready("set")?;
        self.runtime
            .write_chat_state(self.state_var, encode_state(state)?)
    }

    /// Sets the fallible pre-request history policy; reattach it after snapshot restore.
    pub fn with_compactor(mut self, compactor: impl Compactor + 'static) -> Self {
        self.runtime = self.runtime.with_compactor(compactor);
        self
    }

    /// Replaces the history store used to record chat messages.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.runtime = self.runtime.with_store(store);
        self
    }

    /// Captures execution, conversation history and application state together.
    ///
    /// Available before the first message and during unfinished turns. A busy history
    /// lock can cause failure; the application owns durable storage of the snapshot.
    pub fn snapshot(&self) -> Result<Snapshot, GraphError> {
        self.runtime.snapshot()
    }

    /// Converts input at a chat boundary and runs until the next assistant response.
    ///
    /// Builder chats accept strings, messages or `ChatRequest` directly.
    /// Failed or cancelled unfinished turns reject another send; this method never
    /// retries them. Agent/tool suspension returns `GraphError::ChatSuspended`.
    pub async fn send(&mut self, input: impl Into<I>) -> Result<ChatTurn<O>, GraphError> {
        self.submit(input, None, "send").await
    }

    /// Converts input and sends it with a key overriding the configured user-message key.
    /// The key is durable before activation; readiness and conversion failures accept no input.
    pub async fn send_with_key(
        &mut self,
        input: impl Into<I>,
        key: impl Into<String>,
    ) -> Result<ChatTurn<O>, GraphError> {
        self.submit(input, Some(key.into()), "send_with_key").await
    }

    /// Validates and converts before changing the suspended execution.
    async fn submit(
        &mut self,
        input: impl Into<I>,
        key: Option<String>,
        operation: &'static str,
    ) -> Result<ChatTurn<O>, GraphError> {
        self.require_ready(operation)?;
        let input = input.into();
        if let Some(validate) = self.validate_input {
            validate(
                &input,
                self.runtime.chat_agent_payload(&self.input_boundaries)?,
            )?;
        }
        let mut step = self.runtime.resume(ChatSubmission { input, key }).await?;
        loop {
            match step {
                Step::Continue => step = self.runtime.next().await?,
                Step::Suspend(value) => return self.decode_response(value),
                Step::Done(_) => {
                    return Err(GraphError::Invalid(
                        "chat runtime completed unexpectedly".into(),
                    ));
                }
            }
        }
    }

    /// Rechecks persisted input separately from VM shape and frame-relationship validation.
    fn validate_restored_inputs(&self) -> Result<(), GraphError> {
        let validate = |value: &Value| -> Result<(), GraphError> {
            let request: ChatSubmission<I> = from_value(value.clone())
                .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
            if let Some(check) = self.validate_input {
                check(
                    &request.input,
                    self.runtime.chat_agent_payload(&self.input_boundaries)?,
                )
                .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
            }
            Ok(())
        };
        self.runtime
            .validate_chat_inputs(&self.input_boundaries, validate)
    }

    fn require_ready(&self, operation: &'static str) -> Result<(), GraphError> {
        if self.runtime.chat_ready(&self.input_boundaries) {
            Ok(())
        } else {
            Err(GraphError::ChatNotReady { operation })
        }
    }

    /// Decodes only the authored response boundary, never a tool or controller payload.
    fn decode_response(&self, value: Value) -> Result<ChatTurn<O>, GraphError> {
        let [_, response] = self.input_boundaries;
        if !self.runtime.chat_ready(&[response]) {
            return Err(GraphError::ChatSuspended);
        }
        from_value(value)
            .map(|output| ChatTurn { output })
            .map_err(|error| GraphError::ValueConversion {
                target: "chat response".into(),
                reason: error.to_string(),
            })
    }
}

fn encode_state<S: Serialize>(state: S) -> Result<Value, GraphError> {
    to_value(state).map_err(|error| GraphError::ValueConversion {
        target: "chat state".into(),
        reason: error.to_string(),
    })
}
