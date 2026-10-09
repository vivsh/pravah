use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

use super::{AgentExecutor, AgentRequest, AgentResponse};
use crate::Context;
use crate::history::{Compactor, HistoryStore};
use uuid::Uuid;

use super::agent::Agent;
use super::error::GraphError;
use super::ids::{NodeId, VarId};
use super::runtime::{Runtime, Snapshot};
use super::state::Step;
use super::typed::build_chat_graph;
use super::value::{Value, from_value, to_value};

mod builder;
mod request;
mod streaming;
#[cfg(test)]
mod tests;
pub use builder::ChatBuilder;
pub use request::ChatRequest;

/// One assistant response produced by graph chat.
#[derive(Debug, Clone)]
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
/// Chats retain one conversation using an explicit configuration key or a default
/// derived from their execution UUID. Application state is never added to prompts.
///
/// Inputs cannot bypass the chat's fixed type, even through a request envelope:
/// ```compile_fail
/// use pravah::{Chat, ChatRequest};
/// async fn wrong_input(chat: &mut Chat<u32, String>) {
///     chat.send(ChatRequest::from(String::from("not a number"))).await;
/// }
/// ```
/// ```compile_fail
/// use pravah::Chat;
/// async fn wrong_input(chat: &mut Chat<u32, String>) {
///     chat.send("not a number").await;
/// }
/// ```
pub struct Chat<I, O, S = ()> {
    runtime: Runtime,
    executor: AgentExecutor,
    state_var: VarId,
    input_boundaries: [NodeId; 2],
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
    pub fn new(agent: fn(Agent<I>) -> Agent<O>, context: Context) -> Result<Self, GraphError> {
        Self::with_state(agent, (), context)
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
    pub fn with_state(
        agent: fn(Agent<I>) -> Agent<O>,
        state: S,
        context: Context,
    ) -> Result<Self, GraphError> {
        Self::from_definition(agent(Agent::root()), state, context)
    }

    /// Initializes the common graph for function-defined and builder-created chats.
    fn from_definition(agent: Agent<O>, state: S, context: Context) -> Result<Self, GraphError> {
        let value = encode_state(state)?;
        let (prepared, state_var, input_boundaries) = build_chat_graph::<I, O, S>(agent)?;
        let executor = prepared.executor(context);
        let mut runtime = prepared.start(Value::NULL, Uuid::now_v7())?;
        runtime.write_chat_state(state_var, value)?;
        let [bootstrap, _] = input_boundaries;
        if !matches!(runtime.next()?, Step::Suspend(_)) || !runtime.chat_ready(&[bootstrap]) {
            return Err(GraphError::Invalid(
                "chat did not reach its initial input boundary".into(),
            ));
        }
        Ok(Self {
            runtime,
            executor,
            state_var,
            input_boundaries,
            _marker: PhantomData,
        })
    }

    /// Restores execution and application state using freshly supplied runtime context.
    ///
    /// Validates the graph, VM state and decoding into `S` without running handlers.
    /// Saved history intent is retained; reattach stores and compactors after restoration.
    /// Old chat graphs are rejected.
    pub fn from_snapshot(
        agent: fn(Agent<I>) -> Agent<O>,
        snapshot: Snapshot,
        context: Context,
    ) -> Result<Self, GraphError> {
        Self::restore_definition(agent(Agent::root()), snapshot, context)
    }

    /// Restores without calling application handlers and checks retained submission values.
    fn restore_definition(
        agent: Agent<O>,
        snapshot: Snapshot,
        context: Context,
    ) -> Result<Self, GraphError> {
        let (prepared, state_var, input_boundaries) = build_chat_graph::<I, O, S>(agent)?;
        let executor = prepared.executor(context);
        let runtime = prepared.restore(snapshot)?;
        let chat = Self {
            runtime,
            executor,
            state_var,
            input_boundaries,
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

    /// Installs compaction dependencies and opts this chat into request-time working memory.
    /// Reinstall dependencies after restore; checkpointed intent and acknowledgements remain in the VM.
    pub fn with_compactor(mut self, compactor: impl Compactor + 'static) -> Self {
        self.runtime.enable_chat_history(false, true);
        self.executor = self.executor.with_compactor(compactor);
        self
    }

    /// Installs persistence/retrieval dependencies and enables keyed conversation loading.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.runtime.enable_chat_history(true, false);
        self.executor = self.executor.with_store(store);
        self
    }

    /// Captures execution, conversation history and application state together.
    ///
    /// Available before the first message and during unfinished turns. The application
    /// owns durable storage of the snapshot; invalid runtime state causes failure.
    pub fn snapshot(&self) -> Result<Snapshot, GraphError> {
        self.runtime.snapshot()
    }

    /// Reports whether execution references this history session ID; inspection allocates nothing.
    pub fn conversation_is_active(&self, conversation_id: &str) -> bool {
        self.runtime.conversation_is_active(conversation_id)
    }

    /// Drops inactive working conversation history without deleting its archive or requiring persistence.
    /// Use the exact history session ID. Active turns reject removal; a later keyed turn may reload.
    pub fn drop_conversation(&mut self, conversation_id: &str) -> Result<(), GraphError> {
        self.runtime.drop_conversation(conversation_id)
    }

    /// Converts input at a chat boundary and runs until the next assistant response.
    ///
    /// Accepts the fixed input type or a `ChatRequest<I>` with invocation overrides.
    /// Failed or cancelled unfinished turns reject another send; this method never
    /// retries them. Agent/tool suspension returns `GraphError::ChatSuspended`.
    pub async fn send(
        &mut self,
        input: impl Into<ChatRequest<I>>,
    ) -> Result<ChatTurn<O>, GraphError> {
        self.accept_submission(input, None, "send")?;
        self.finish_send().await
    }

    /// Converts input and sends it with a key overriding the configured user-message key.
    /// The key is durable before activation; readiness and conversion failures accept no input.
    pub async fn send_with_key(
        &mut self,
        input: impl Into<ChatRequest<I>>,
        key: impl Into<String>,
    ) -> Result<ChatTurn<O>, GraphError> {
        self.accept_submission(input, Some(key.into()), "send_with_key")?;
        self.finish_send().await
    }

    /// Validates and converts before changing the suspended execution.
    fn accept_submission(
        &mut self,
        input: impl Into<ChatRequest<I>>,
        key: Option<String>,
        operation: &'static str,
    ) -> Result<(), GraphError> {
        self.require_ready(operation)?;
        let mut input = input.into();
        if let Some(key) = key {
            input.key = Some(key);
        }
        input.validate(self.runtime.chat_agent_payload(&self.input_boundaries)?)?;
        self.runtime.resume(input)
    }

    /// Accepts a typed submission without executing effects or advancing the turn.
    pub fn submit(&mut self, input: impl Into<ChatRequest<I>>) -> Result<(), GraphError> {
        self.accept_submission(input, None, "submit")
    }

    /// Accepts a durable keyed submission without executing effects.
    pub fn submit_with_key(
        &mut self,
        input: impl Into<ChatRequest<I>>,
        key: impl Into<String>,
    ) -> Result<(), GraphError> {
        self.accept_submission(input, Some(key.into()), "submit_with_key")
    }

    /// Performs one synchronous VM step, distinguishing the response boundary from internal suspension.
    #[expect(
        clippy::should_implement_trait,
        reason = "explicit fallible VM stepping is not iteration"
    )]
    pub fn next(&mut self) -> Result<ChatStep<O>, GraphError> {
        match self.runtime.next()? {
            Step::Continue => Ok(ChatStep::Continue),
            Step::Agent(request) => Ok(ChatStep::Agent(request)),
            Step::Suspend(value) if self.runtime.chat_ready(&[self.input_boundaries[1]]) => {
                Ok(ChatStep::Done(self.decode_response(value)?))
            }
            Step::Suspend(value) => Ok(ChatStep::Suspend(value)),
            Step::Done(_) => Err(GraphError::Invalid(
                "chat runtime completed unexpectedly".into(),
            )),
        }
    }

    /// Borrows the external executor; it retains no conversation or pending-operation state.
    pub fn executor(&self) -> &AgentExecutor {
        &self.executor
    }

    /// Borrows the pending agent operation without acquiring or retaining runtime access.
    pub fn pending_agent(&self) -> Option<&AgentRequest> {
        self.runtime.pending_agent()
    }

    /// Accepts a worker completion; next interprets its outcome and recorded acknowledgements.
    pub fn resume_agent(&mut self, response: AgentResponse) -> Result<(), GraphError> {
        self.runtime.resume_agent(response)
    }

    /// Accepts a typed value for an internal agent or tool suspension.
    pub fn resume<T: Serialize>(&mut self, input: T) -> Result<(), GraphError> {
        self.runtime.resume(input)
    }

    /// Drives synchronous steps and asynchronous workers without retrying failed operations.
    async fn finish_send(&mut self) -> Result<ChatTurn<O>, GraphError> {
        loop {
            match self.next()? {
                ChatStep::Continue => {}
                ChatStep::Agent(request) => {
                    let response = self.executor.execute(&request).await;
                    self.resume_agent(response)?;
                }
                ChatStep::Suspend(_) => return Err(GraphError::ChatSuspended),
                ChatStep::Done(turn) => return Ok(turn),
            }
        }
    }

    /// Rechecks persisted input separately from VM shape and frame-relationship validation.
    fn validate_restored_inputs(&self) -> Result<(), GraphError> {
        let validate = |value: &Value| -> Result<(), GraphError> {
            let request: ChatRequest<I> = from_value(value.clone())
                .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
            request
                .validate(self.runtime.chat_agent_payload(&self.input_boundaries)?)
                .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
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

/// Result of one manually driven Chat step; errors use the ordinary Result channel.
#[derive(Debug, Clone)]
pub enum ChatStep<O> {
    /// Internal synchronous progress.
    Continue,
    /// One external request awaiting delivery.
    Agent(AgentRequest),
    /// Agent or tool input is required, not a completed assistant response.
    Suspend(Value),
    /// The assistant response is complete and the chat is ready for another submission.
    Done(ChatTurn<O>),
}
