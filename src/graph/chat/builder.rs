use super::Chat;
use crate::history::{DynCompactor, DynHistoryStore};
use crate::{
    Agent, AgentDecision, AgentLoop, Compactor, Context, GraphError, HistoryStore, McpResourceRef,
    Snapshot, Toolset,
};
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use std::sync::Arc;
use std::{error::Error, future::Future, marker::PhantomData};

/// Builds a graph-backed Chat retaining one keyed conversation without a configure callback.
/// All settings except instructions are graph data and must match when restoring.
/// New instructions apply only when an invocation has not committed configuration.
pub struct ChatBuilder<I, O, S = ()> {
    agent: Agent<I>,
    state: S,
    store: Option<Arc<dyn DynHistoryStore>>,
    compactor: Option<Arc<dyn DynCompactor>>,
    _marker: PhantomData<fn() -> O>,
}

impl Chat<(), ()> {
    /// Starts a conversation with fixed input `I` rendered as JSON and output `O`.
    pub fn builder<I, O>() -> ChatBuilder<I, O> {
        ChatBuilder {
            agent: Agent::root(),
            state: (),
            store: None,
            compactor: None,
            _marker: PhantomData,
        }
    }
}

impl<I, O, S> ChatBuilder<I, O, S> {
    /// Selects this Chat's conversation key, replacing any earlier value.
    /// Empty keys fail at build/restore; omission derives an isolated key from the execution UUID.
    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.agent = self.agent.key(key);
        self
    }

    /// Sets initial application state for a new chat, replacing any previous value or state type.
    /// State is converted only at build and lives exclusively in the runtime afterward.
    pub fn state<T>(self, state: T) -> ChatBuilder<I, O, T> {
        ChatBuilder {
            agent: self.agent,
            state,
            store: self.store,
            compactor: self.compactor,
            _marker: PhantomData,
        }
    }

    /// Installs the pre-request compactor, replacing any previous one; never serialized.
    pub fn compactor(mut self, compactor: impl Compactor + 'static) -> Self {
        self.compactor = Some(Arc::new(compactor));
        self
    }

    /// Installs the message history store, replacing any previous one; never serialized.
    pub fn store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.store = Some(Arc::new(store));
        self
    }

    /// Sets the required provider/model URL, replacing any previous value.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.agent = self.agent.model(model);
        self
    }

    /// Sets instructions for future invocations, replacing any previous value.
    /// On restore, already-configured invocations keep their checkpointed instructions.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.agent = self.agent.instructions(text);
        self
    }

    /// Sets provider options; credentials should remain in runtime Context dependencies.
    pub fn provider_config(mut self, config: impl Into<JsonValue>) -> Self {
        self.agent = self.agent.provider_config(config);
        self
    }

    /// Declares candidate tools; duplicate identities fail at build or restore.
    /// The builder runs immediately and may consume captured catalogue definitions.
    pub fn tools(mut self, build: impl FnOnce(Toolset) -> Toolset) -> Self {
        self.agent = self.agent.tools(build);
        self
    }

    /// Sets default MCP references; requests may replace them, including with an empty list.
    pub fn resources(mut self, refs: impl IntoIterator<Item = McpResourceRef>) -> Self {
        self.agent = self.agent.resources(refs);
        self
    }

    /// Registers the ordinary asynchronous agent-loop controller, at most once.
    pub fn control<Fut, E>(mut self, control: fn(AgentLoop<I>, Context) -> Fut) -> Self
    where
        I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
        Fut: Future<Output = Result<AgentDecision, E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        self.agent = self.agent.control(control);
        self
    }

    /// Caps output per model request, replacing the previous cap; a final zero is invalid.
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.agent = self.agent.max_output_tokens(tokens);
        self
    }

    /// Caps ordinary model turns; exhausted agents use the existing forced-conclusion path.
    pub fn turn_budget(mut self, turns: u32) -> Self {
        self.agent = self.agent.turn_budget(turns);
        self
    }

    /// Caps accepted calls for an explicit tool name, including a JSON alias.
    /// Invalid, duplicate or unknown budgets fail build or restore.
    pub fn tool_budget_named(mut self, name: impl Into<String>, calls: u32) -> Self {
        self.agent = self.agent.tool_budget_named(name, calls);
        self
    }

    /// Caps accepted calls for the canonical tool input identity; errors accumulate until build.
    pub fn tool_budget<T: JsonSchema>(mut self, calls: u32) -> Self {
        self.agent = self.agent.tool_budget::<T>(calls);
        self
    }
}

impl<I, O, S> ChatBuilder<I, O, S>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Validates and initializes a snapshot-ready chat without external calls or history writes.
    /// Consumes configuration, moving initial state and services into their existing owners.
    pub fn build(self, ctx: Context) -> Result<Chat<I, O, S>, GraphError> {
        let chat = Chat::from_definition(self.agent.build_checked()?, self.state, ctx)?;
        Ok(attach_services(chat, self.store, self.compactor))
    }
}

impl<I, O> ChatBuilder<I, O>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Restores with matching settings and fresh context; instructions alone may differ.
    /// Does not step, resolve resources, or replace committed configuration.
    /// Configure services beforehand. State comes exclusively from the snapshot;
    /// restoration is unavailable on builders carrying non-unit initial state.
    pub fn restore<S>(self, snapshot: Snapshot, ctx: Context) -> Result<Chat<I, O, S>, GraphError>
    where
        S: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let chat = Chat::restore_definition(self.agent.build_checked()?, snapshot, ctx)?;
        Ok(attach_services(chat, self.store, self.compactor))
    }
}

/// Attaches runtime-only services without duplicating declaration settings or history ownership.
fn attach_services<I, O, S>(
    mut chat: Chat<I, O, S>,
    store: Option<Arc<dyn DynHistoryStore>>,
    compactor: Option<Arc<dyn DynCompactor>>,
) -> Chat<I, O, S> {
    chat.runtime
        .enable_chat_history(store.is_some(), compactor.is_some());
    chat.executor.store = store;
    chat.executor.compactor = compactor;
    chat
}
