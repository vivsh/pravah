use std::any::{TypeId, type_name};
use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

use crate::clients::{
    Client, ClientError, ClientOptions, ClientOutput, Message, ResponseFormat, Role,
    TokenUsage, materialize_messages,
};
use crate::context::Context;
use crate::flows::compactor::{DynHistoryCompactor, HistoryCompactor, NoopCompactor};
use crate::flows::memory::{DynMemoryFactory, MemoryFactory, MemoryQuery, NoopMemoryFactory};
use crate::flows::store::{DynHistoryStore, HistoryStore, NoopHistoryStore};
use crate::flows::{FlowHistory, HistoryEntry};

/// Agent-id label used when pushing messages into [`FlowHistory`].
const CHAT_AGENT_ID: &str = "chat";

/// Error returned by [`Chat`] operations.
#[derive(Debug, Error)]
pub enum ChatError {
    #[error(transparent)]
    Client(#[from] ClientError),
    /// `send_message` received a message whose role is not [`Role::User`].
    #[error("message role must be User")]
    NonUserMessage,
    /// Text sessions only accept string model output.
    #[error("model returned non-text output for a text chat session")]
    UnexpectedOutput,
    /// Model returned tool calls. Tools are not supported in [`Chat`].
    #[error("model returned tool calls; tools are not supported in Chat")]
    ToolCallsNotSupported,
    /// The input type could not be represented as a plain string.
    #[error("text chat input type {ty} did not serialize to a string")]
    InvalidTextInput { ty: &'static str },
    /// Failed to serialize a chat input value.
    #[error("failed to serialize chat input for {ty}: {source}")]
    InputSerialize {
        ty: &'static str,
        #[source]
        source: serde_json::Error,
    },
    /// Failed to decode a typed chat output value.
    #[error("failed to decode chat output as {ty}: {source}")]
    OutputDeserialize {
        ty: &'static str,
        #[source]
        source: serde_json::Error,
    },
    /// Failed to build a JSON Schema for a typed chat value.
    #[error("failed to build chat schema for {ty}: {source}")]
    Schema {
        ty: &'static str,
        #[source]
        source: serde_json::Error,
    },
    /// The history store flush reported an error.
    ///
    /// The store interface erases the concrete error type.
    /// When a flush fails the in-memory history is already updated; the store
    /// may be behind. There are no rollback semantics.
    #[error("history store flush failed: {0}")]
    Store(Box<dyn std::error::Error + Send + Sync>),
    /// Memory retrieval failed before dispatch.
    #[error("memory retrieval failed: {0}")]
    Memory(Box<dyn std::error::Error + Send + Sync>),
}

/// Provider-agnostic wire format for chat input/output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatWireKind {
    /// Plain text content.
    Text,
    /// JSON content.
    Json,
}

impl ChatWireKind {
    fn response_format(self) -> ResponseFormat {
        match self {
            Self::Text => ResponseFormat::Text,
            Self::Json => ResponseFormat::Json,
        }
    }
}

/// Chat payload contract used by [`Chat`].
///
/// All `Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static`
/// types implement this automatically. `String` stays in plain-text mode; all
/// other types use JSON with schema-driven client settings.
pub trait ChatType:
    Serialize + DeserializeOwned + JsonSchema + Send + Sync + Sized + 'static
{
    /// Returns the provider-agnostic wire format for this type.
    fn wire_kind() -> ChatWireKind {
        if TypeId::of::<Self>() == TypeId::of::<String>() {
            ChatWireKind::Text
        } else {
            ChatWireKind::Json
        }
    }

    /// Returns the JSON Schema used to configure the provider, when needed.
    fn schema() -> Result<Option<Value>, ChatError> {
        if Self::wire_kind() == ChatWireKind::Text {
            return Ok(None);
        }
        serde_json::to_value(schemars::schema_for!(Self)).map(Some).map_err(|source| {
            ChatError::Schema {
                ty: type_name::<Self>(),
                source,
            }
        })
    }

    /// Serializes the value into the user-visible message content.
    fn encode_input(&self) -> Result<String, ChatError> {
        let value = serde_json::to_value(self).map_err(|source| ChatError::InputSerialize {
            ty: type_name::<Self>(),
            source,
        })?;
        match Self::wire_kind() {
            ChatWireKind::Text => match value {
                Value::String(text) => Ok(text),
                _ => Err(ChatError::InvalidTextInput {
                    ty: type_name::<Self>(),
                }),
            },
            ChatWireKind::Json => Ok(value.to_string()),
        }
    }

    /// Formats a decoded provider value for history persistence.
    fn history_content(value: &Value) -> Result<String, ChatError> {
        match Self::wire_kind() {
            ChatWireKind::Text => value
                .as_str()
                .map(str::to_owned)
                .ok_or(ChatError::UnexpectedOutput),
            ChatWireKind::Json => Ok(value.to_string()),
        }
    }

    /// Decodes the provider value into the requested output type.
    fn decode_output(value: Value) -> Result<Self, ChatError> {
        if Self::wire_kind() == ChatWireKind::Text && !value.is_string() {
            return Err(ChatError::UnexpectedOutput);
        }
        serde_json::from_value(value).map_err(|source| ChatError::OutputDeserialize {
            ty: type_name::<Self>(),
            source,
        })
    }
}

impl<T> ChatType for T where
    T: Serialize + DeserializeOwned + JsonSchema + Send + Sync + Sized + 'static
{
}

/// One completed conversation turn.
#[derive(Debug)]
pub struct ChatTurn<Output = String> {
    /// Assistant reply value.
    pub output: Output,
    /// Token usage reported by the provider, if available.
    pub usage: Option<TokenUsage>,
}

impl<Output> ChatTurn<Output> {
    /// Consumes the turn and returns the output value.
    pub fn into_output(self) -> Output {
        self.output
    }
}

impl ChatTurn<String> {
    /// Borrows the assistant reply text.
    pub fn text(&self) -> &str {
        &self.output
    }

    /// Consumes the turn and returns the reply text.
    pub fn into_text(self) -> String {
        self.output
    }
}

/// Snapshot of session state for in-process restoration.
///
/// `options` captures the full [`ClientOptions`] so that provider settings,
/// schemas, and response mode survive snapshot/restore cycles.
///
/// **Not included:** compactor and store. Re-attach them after
/// [`Chat::from_snapshot`] with [`Chat::with_compactor`] and
/// [`Chat::with_store`]. For durable persistence, attach a
/// [`HistoryStore`] to the session before calling [`Chat::snapshot`].
#[derive(Clone)]
pub struct ChatSnapshot<Input = String, Output = String>
where
    Input: ChatType,
    Output: ChatType,
{
    pub session_id: String,
    pub url: String,
    pub options: ClientOptions,
    pub history: FlowHistory,
    _types: PhantomData<(Input, Output)>,
}

/// Builder for [`Chat`].
pub struct ChatBuilder<Input = String, Output = String>
where
    Input: ChatType,
    Output: ChatType,
{
    url: String,
    options: ClientOptions,
    environment: Option<String>,
    session_id: Option<String>,
    compactor: Box<dyn DynHistoryCompactor>,
    store: Box<dyn DynHistoryStore>,
    memory: Box<dyn DynMemoryFactory>,
    _types: PhantomData<(Input, Output)>,
}

impl<Input: ChatType, Output: ChatType> ChatBuilder<Input, Output> {
    /// Sets the static system preamble sent before conversation history.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.options.preamble = Some(preamble.into());
        self
    }

    /// Appends runtime environment text to the system prompt.
    ///
    /// The environment is appended after the preamble (if any) and before the
    /// input-schema hint. It is baked in at build time.
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.environment = Some(env.into());
        self
    }

    /// Sets the sampling temperature. Higher values increase output randomness.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.options.temperature = Some(temperature);
        self
    }

    /// Overrides the auto-generated session UUID.
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Attaches a history compactor to the session.
    pub fn with_compactor(mut self, compactor: impl HistoryCompactor + 'static) -> Self {
        self.compactor = Box::new(compactor);
        self
    }

    /// Attaches a history store to the session.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.store = Box::new(store);
        self
    }

    /// Attaches a memory factory for per-turn dynamic context injection.
    ///
    /// Called once per `send`/`send_message` call; the result is prepended to
    /// the outgoing messages as a system prompt but is never stored in history.
    pub fn with_memory(mut self, memory: impl MemoryFactory + Send + Sync + 'static) -> Self {
        self.memory = Box::new(memory);
        self
    }

    /// Builds the session, connecting to the provider specified by `url`.
    pub fn build(mut self) -> Result<Chat<Input, Output>, ChatError> {
        if let Some(env) = self.environment {
            self.options.preamble = Some(match self.options.preamble.take() {
                Some(p) => format!("{p}\n\n{env}"),
                None => env,
            });
        }
        let options = build_typed_options::<Input, Output>(self.options)?;
        let client = options.clone().create(&self.url)?;
        Ok(Chat {
            session_id: self.session_id.unwrap_or_else(|| Uuid::now_v7().to_string()),
            url: self.url,
            options,
            client,
            history: FlowHistory::new(),
            compactor: self.compactor,
            store: self.store,
            memory: self.memory,
            _types: PhantomData,
        })
    }
}

/// Stateful single-conversation chat session backed by [`FlowHistory`].
///
/// `Chat<String, String>` is plain text chat. Any non-`String` input or output
/// type switches the session into JSON mode for that side.
///
/// One instance owns exactly one conversation. For multi-user or multi-agent
/// scenarios use [`FlowRuntime`](crate::flows::FlowRuntime).
///
/// **Supported:** text and typed JSON sessions, plus multimodal input via
/// [`send_message`](Chat::send_message) on `Chat<String, Output>`.
/// **Not supported:** tool calls. Use [`Flow`](crate::flows::Flow) for that.
pub struct Chat<Input = String, Output = String>
where
    Input: ChatType,
    Output: ChatType,
{
    session_id: String,
    url: String,
    options: ClientOptions,
    client: Box<dyn Client>,
    history: FlowHistory,
    compactor: Box<dyn DynHistoryCompactor>,
    store: Box<dyn DynHistoryStore>,
    memory: Box<dyn DynMemoryFactory>,
    _types: PhantomData<(Input, Output)>,
}

impl<Input: ChatType, Output: ChatType> Chat<Input, Output> {
    /// Creates a builder for a session targeting the given model URL.
    pub fn builder(url: impl Into<String>) -> ChatBuilder<Input, Output> {
        ChatBuilder {
            url: url.into(),
            options: ClientOptions::default(),
            environment: None,
            session_id: None,
            compactor: Box::new(NoopCompactor),
            store: Box::new(NoopHistoryStore),
            memory: Box::new(NoopMemoryFactory),
            _types: PhantomData,
        }
    }

    /// Restores a session from a [`ChatSnapshot`].
    ///
    /// The provider client is re-created fresh; it is never serialized.
    /// Compactor and store revert to no-ops; re-attach with
    /// [`with_compactor`](Self::with_compactor) and [`with_store`](Self::with_store).
    pub fn from_snapshot(snap: ChatSnapshot<Input, Output>) -> Result<Self, ChatError> {
        let client = snap.options.clone().create(&snap.url)?;
        Ok(Self {
            session_id: snap.session_id,
            url: snap.url,
            options: snap.options,
            client,
            history: snap.history,
            compactor: Box::new(NoopCompactor),
            store: Box::new(NoopHistoryStore),
            memory: Box::new(NoopMemoryFactory),
            _types: PhantomData,
        })
    }

    /// Replaces the history compactor. Useful after [`from_snapshot`](Self::from_snapshot).
    pub fn with_compactor(mut self, compactor: impl HistoryCompactor + 'static) -> Self {
        self.compactor = Box::new(compactor);
        self
    }

    /// Replaces the history store. Useful after [`from_snapshot`](Self::from_snapshot).
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.store = Box::new(store);
        self
    }

    /// Replaces the memory factory. Useful after [`from_snapshot`](Self::from_snapshot).
    pub fn with_memory(mut self, memory: impl MemoryFactory + Send + Sync + 'static) -> Self {
        self.memory = Box::new(memory);
        self
    }

    /// Returns the session id used to tag history entries.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Borrows the full conversation history for this session.
    pub fn history(&self) -> &FlowHistory {
        &self.history
    }

    /// Returns a snapshot of the current session state.
    pub fn snapshot(&self) -> ChatSnapshot<Input, Output> {
        ChatSnapshot {
            session_id: self.session_id.clone(),
            url: self.url.clone(),
            options: self.options.clone(),
            history: self.history.clone(),
            _types: PhantomData,
        }
    }

    /// Sends typed input and returns the typed assistant reply.
    pub async fn send(
        &mut self,
        ctx: Context,
        input: impl Into<Input>,
    ) -> Result<ChatTurn<Output>, ChatError> {
        let input = input.into();
        let input_value = serde_json::to_value(&input).unwrap_or(Value::Null);
        let env = self.retrieve_memory(&ctx, &input_value).await?;
        let msg = Message::user(input.encode_input()?);
        let turn = self.dispatch(&ctx, &msg, env).await?;
        let reply = build_reply(&turn.reply_content, turn.usage);
        self.history.push(&self.session_id, CHAT_AGENT_ID, msg);
        self.history.push(&self.session_id, CHAT_AGENT_ID, reply);
        self.compact_and_flush().await?;
        Ok(turn.into_turn())
    }

    async fn retrieve_memory(&self, ctx: &Context, input: &Value) -> Result<Option<String>, ChatError> {
        self.memory
            .retrieve_dyn(&MemoryQuery { agent_name: CHAT_AGENT_ID, input, ctx })
            .await
            .map_err(ChatError::Memory)
    }

    async fn dispatch(
        &self,
        ctx: &Context,
        msg: &Message,
        env: Option<String>,
    ) -> Result<DispatchTurn<Output>, ChatError> {
        let mut msgs = self.history.for_session(&self.session_id);
        if let Some(text) = env {
            msgs.insert(0, Message { role: Role::System, content: text, attachments: Vec::new(), usage: None });
        }
        msgs.push(msg.clone());
        let materialized = materialize_messages(&msgs, ctx).await?;
        let resp = self.client.execute(&materialized).await?;
        let decoded = decode_client_output::<Output>(resp.output)?;
        Ok(DispatchTurn {
            output: decoded.output,
            reply_content: decoded.reply_content,
            usage: resp.usage,
        })
    }

    /// Clones session entries to satisfy the compactor's borrowing contract,
    /// then flushes the store. The clone is O(n) in history length; a
    /// [`SlidingWindowCompactor`](crate::flows::SlidingWindowCompactor) keeps
    /// n small.
    async fn compact_and_flush(&mut self) -> Result<(), ChatError> {
        let owned: Vec<HistoryEntry> = self
            .history
            .session_entries(&self.session_id)
            .into_iter()
            .cloned()
            .collect();
        let refs: Vec<&HistoryEntry> = owned.iter().collect();
        let result = self.compactor.compact_dyn(&self.session_id, &refs).await;
        if let Err(e) = self.history.apply_compaction(&self.session_id, &refs, result) {
            tracing::warn!(
                session_id = %self.session_id,
                error = %e,
                "compaction failed; history unchanged"
            );
        }
        self.store
            .flush_dyn(&mut self.history)
            .await
            .map_err(ChatError::Store)
    }
}

impl<Output: ChatType> Chat<String, Output> {
    /// Sends an explicit user [`Message`] and returns the assistant reply.
    ///
    /// Fails with [`ChatError::NonUserMessage`] if `msg.role` is not [`Role::User`].
    /// File attachments are resolved via `ctx` before dispatch.
    pub async fn send_message(
        &mut self,
        ctx: Context,
        msg: Message,
    ) -> Result<ChatTurn<Output>, ChatError> {
        if !matches!(msg.role, Role::User) {
            return Err(ChatError::NonUserMessage);
        }
        let input_value = Value::String(msg.content.clone());
        let env = self.retrieve_memory(&ctx, &input_value).await?;
        let turn = self.dispatch(&ctx, &msg, env).await?;
        let reply = build_reply(&turn.reply_content, turn.usage);
        self.history.push(&self.session_id, CHAT_AGENT_ID, msg);
        self.history.push(&self.session_id, CHAT_AGENT_ID, reply);
        self.compact_and_flush().await?;
        Ok(turn.into_turn())
    }
}

struct DecodedOutput<Output> {
    output: Output,
    reply_content: String,
}

struct DispatchTurn<Output> {
    output: Output,
    reply_content: String,
    usage: Option<TokenUsage>,
}

impl<Output> DispatchTurn<Output> {
    fn into_turn(self) -> ChatTurn<Output> {
        ChatTurn {
            output: self.output,
            usage: self.usage,
        }
    }
}

fn build_typed_options<Input: ChatType, Output: ChatType>(
    mut options: ClientOptions,
) -> Result<ClientOptions, ChatError> {
    if let Some(schema) = Input::schema()? {
        options = options.with_input_schema(schema);
    }
    if let Some(schema) = Output::schema()? {
        options = options.with_output_schema(schema);
    }
    Ok(options.with_response_format(Output::wire_kind().response_format()))
}

fn build_reply(text: &str, usage: Option<TokenUsage>) -> Message {
    match usage {
        Some(u) => Message::assistant(text).with_usage(u),
        None => Message::assistant(text),
    }
}

fn decode_client_output<Output: ChatType>(
    output: ClientOutput,
) -> Result<DecodedOutput<Output>, ChatError> {
    match output {
        ClientOutput::Output(value) => Ok(DecodedOutput {
            reply_content: Output::history_content(&value)?,
            output: Output::decode_output(value)?,
        }),
        ClientOutput::ToolCalls { .. } => Err(ChatError::ToolCallsNotSupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
    struct DraftInput {
        topic: String,
    }

    #[derive(Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
    struct DraftOutput {
        answer: String,
    }

    /// String chat stays in plain-text mode and sends unquoted content.
    #[test]
    fn string_chat_type_stays_text() {
        assert_eq!(String::wire_kind(), ChatWireKind::Text);
        assert!(String::schema().unwrap().is_none());
        assert_eq!("hello", "hello".to_string().encode_input().unwrap());
    }

    /// Non-string chat types use JSON mode and derive a schema.
    #[test]
    fn json_chat_type_uses_json_wire_format() {
        let input = DraftInput {
            topic: "ownership".into(),
        };
        assert_eq!(DraftInput::wire_kind(), ChatWireKind::Json);
        assert!(DraftInput::schema().unwrap().is_some());
        assert_eq!(input.encode_input().unwrap(), r#"{"topic":"ownership"}"#);
    }

    /// Typed options use the output type to decide provider response mode.
    #[test]
    fn typed_options_use_output_type_for_response_mode() {
        let text = build_typed_options::<DraftInput, String>(ClientOptions::default()).unwrap();
        assert!(text.input_schema.is_some());
        assert!(text.output_schema.is_none());
        assert_eq!(text.response_format, ResponseFormat::Text);

        let json = build_typed_options::<String, DraftOutput>(ClientOptions::default()).unwrap();
        assert!(json.input_schema.is_none());
        assert!(json.output_schema.is_some());
        assert_eq!(json.response_format, ResponseFormat::Json);
    }

    /// Text output rejects structured provider values.
    #[test]
    fn text_output_rejects_structured_values() {
        let err = String::decode_output(serde_json::json!({ "answer": "ok" })).unwrap_err();
        assert!(matches!(err, ChatError::UnexpectedOutput));
    }

    /// JSON output decodes from the provider value into the typed result.
    #[test]
    fn json_output_decodes_from_value() {
        let output = DraftOutput::decode_output(serde_json::json!({ "answer": "ok" }))
            .unwrap();
        assert_eq!(
            output,
            DraftOutput {
                answer: "ok".into(),
            }
        );
    }
}
