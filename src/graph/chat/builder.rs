use super::Chat;
use super::request::validate_resources;
use crate::clients::Message;
use crate::graph::agent::{RequestedToolBudget, agent_tool_identity};
use crate::history::{DynCompactor, DynHistoryStore};
use crate::{
    Agent, AgentConfig, AgentDecision, AgentLoop, Compactor, Context, GraphError, HistoryStore,
    McpResourceRef, Snapshot, Toolset,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use std::sync::Arc;
use std::{error::Error, future::Future, marker::PhantomData};

/// Builds a graph-backed Chat retaining one keyed conversation without a configure callback.
/// All settings except instructions are graph data and must match when restoring.
/// New instructions apply only when an invocation has not committed configuration.
pub struct ChatBuilder<I, O, S = ()> {
    settings: ChatSettings,
    instructions: String,
    agent: Agent<I>,
    errors: Vec<String>,
    state: S,
    store: Option<Arc<dyn DynHistoryStore>>,
    compactor: Option<Arc<dyn DynCompactor>>,
    _marker: PhantomData<fn() -> O>,
}

/// Immutable definition data, decoded only when an invocation configures its agent.
#[derive(Default, Serialize, Deserialize, JsonSchema)]
struct ChatSettings {
    key: Option<String>,
    model: String,
    provider_config: Option<JsonValue>,
    max_output_tokens: Option<u32>,
    turn_budget: Option<u32>,
    tool_budgets: Vec<RequestedToolBudget>,
    resources: Vec<McpResourceRef>,
}

impl Chat<(), ()> {
    /// Starts a conversation with fixed input `I` rendered as JSON and output `O`.
    pub fn builder<I, O>() -> ChatBuilder<I, O> {
        ChatBuilder {
            settings: ChatSettings::default(),
            instructions: String::new(),
            agent: Agent::root(),
            errors: Vec::new(),
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
        self.settings.key = Some(key.into());
        self
    }
    /// Sets initial application state for a new chat, replacing any previous value or state type.
    /// State is converted only at build and lives exclusively in the runtime afterward.
    pub fn state<T>(self, state: T) -> ChatBuilder<I, O, T> {
        ChatBuilder {
            settings: self.settings,
            instructions: self.instructions,
            agent: self.agent,
            errors: self.errors,
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
        self.settings.model = model.into();
        self
    }
    /// Sets instructions for future invocations, replacing any previous value.
    /// On restore, already-configured invocations keep their checkpointed instructions.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = text.into();
        self
    }
    /// Sets provider options; credentials must remain in runtime Context dependencies.
    pub fn provider_config(mut self, config: impl Into<JsonValue>) -> Self {
        self.settings.provider_config = Some(config.into());
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
        self.settings.resources = refs.into_iter().collect();
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
        self.settings.max_output_tokens = Some(tokens);
        self
    }
    /// Caps ordinary model turns; exhausted agents use the existing forced-conclusion path.
    pub fn turn_budget(mut self, turns: u32) -> Self {
        set_budget(
            &mut self.settings.turn_budget,
            turns,
            "turn budget",
            &mut self.errors,
        );
        self
    }
    /// Caps accepted calls for an explicit tool name, including a JSON alias.
    /// Invalid, duplicate or unknown budgets fail build or restore.
    pub fn tool_budget_named(mut self, name: impl Into<String>, calls: u32) -> Self {
        let name = name.into();
        if calls == 0
            || self
                .settings
                .tool_budgets
                .iter()
                .any(|budget| budget.name == name)
        {
            self.errors
                .push(format!("invalid or repeated tool budget for '{name}'"));
        } else {
            self.settings
                .tool_budgets
                .push(RequestedToolBudget { name, limit: calls });
        }
        self
    }

    /// Caps accepted calls for the canonical tool input identity; errors accumulate until build.
    pub fn tool_budget<T: JsonSchema>(mut self, calls: u32) -> Self {
        match agent_tool_identity::<T>() {
            Ok(name) if calls > 0 && !self.settings.tool_budgets.iter().any(|b| b.name == name) => {
                self.settings
                    .tool_budgets
                    .push(RequestedToolBudget { name, limit: calls });
            }
            Ok(name) => self
                .errors
                .push(format!("invalid or repeated tool budget for '{name}'")),
            Err(error) => self.errors.push(error),
        }
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
    /// Consumes all configuration, moving initial state and services into the existing runtime.
    pub fn build(self, ctx: Context) -> Result<Chat<I, O, S>, GraphError> {
        let agent = finish(self.settings, self.instructions, self.agent, self.errors)?;
        let chat = Chat::from_definition(agent, self.state, ctx)?;
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
    /// Configure services beforehand. Application state comes exclusively from the snapshot;
    /// restoration is unavailable on builders carrying non-unit initial state.
    pub fn restore<S>(self, snapshot: Snapshot, ctx: Context) -> Result<Chat<I, O, S>, GraphError>
    where
        S: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let agent = finish(self.settings, self.instructions, self.agent, self.errors)?;
        let chat = Chat::restore_definition(agent, snapshot, ctx)?;
        Ok(attach_services(chat, self.store, self.compactor))
    }
}

/// Rejects definition errors before lowering, resource resolution, or any VM mutation.
fn finish<I, O>(
    settings: ChatSettings,
    instructions: String,
    agent: Agent<I>,
    mut errors: Vec<String>,
) -> Result<Agent<O>, GraphError>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + DeserializeOwned + JsonSchema + Send + Sync,
{
    if settings.model.trim().is_empty() {
        errors.push("model must not be empty".into());
    }
    if settings
        .key
        .as_ref()
        .is_some_and(|key| key.trim().is_empty())
    {
        errors.push("conversation key must not be empty".into());
    }
    if settings.max_output_tokens == Some(0) {
        errors.push("max output tokens must be positive".into());
    }
    if let Err(error) = validate_resources(&settings.resources) {
        errors.push(error);
    }
    for budget in &settings.tool_budgets {
        if !agent.tool_names().any(|name| name == budget.name) {
            errors.push(format!("unknown agent tool '{}'", budget.name));
        }
    }
    if !errors.is_empty() {
        return Err(GraphError::AgentConfigValidation(errors.join("; ")));
    }
    Ok(agent.configure_with(settings, instructions, configure_chat))
}

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

fn set_budget(target: &mut Option<u32>, value: u32, label: &str, errors: &mut Vec<String>) {
    if value == 0 || target.is_some() {
        errors.push(format!("invalid or repeated {label}"));
    } else {
        *target = Some(value);
    }
}

/// Combines a single request with graph-owned settings using ordinary agent configuration.
async fn configure_chat<I: Serialize>(
    input: I,
    settings: ChatSettings,
    instructions: String,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    let content = serde_json::to_string(&input).map_err(|error| GraphError::ValueConversion {
        target: "chat user message".into(),
        reason: error.to_string(),
    })?;
    let message = Message::user(content);
    let mut config = AgentConfig::new(settings.model, instructions, message);
    config.key = settings.key;
    config.provider_config = settings.provider_config;
    config.max_output_tokens = settings.max_output_tokens;
    config.turn_budget = settings.turn_budget;
    config.tool_budgets = settings.tool_budgets;
    config.resources = settings.resources;
    Ok(config)
}
