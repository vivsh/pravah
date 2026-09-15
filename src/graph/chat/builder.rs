use super::request::validate_resources;
use super::{Chat, ChatRequest};
use crate::graph::RuntimeServices;
use crate::graph::agent::{RequestedToolBudget, agent_tool_identity};
use crate::{
    Agent, AgentConfig, AgentDecision, AgentLoop, Compactor, Context, GraphError, HistoryStore,
    McpResourceRef, Snapshot, ToolFilter, Toolset,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use std::{error::Error, future::Future, marker::PhantomData};

/// Builds a graph-backed, keep-alive Chat without an application configure callback.
/// Settings are serialized graph data; use an equivalent builder to restore snapshots.
pub struct ChatBuilder<O, S = ()> {
    settings: ChatSettings,
    agent: Agent<ChatRequest>,
    errors: Vec<String>,
    state: S,
    services: Option<RuntimeServices>,
    _marker: PhantomData<fn() -> O>,
}

/// Immutable definition data, decoded only when an invocation configures its agent.
#[derive(Default, Serialize, Deserialize, JsonSchema)]
struct ChatSettings {
    model: String,
    instructions: String,
    provider_config: Option<JsonValue>,
    max_output_tokens: Option<u32>,
    turn_budget: Option<u32>,
    tool_budgets: Vec<RequestedToolBudget>,
    resources: Vec<McpResourceRef>,
}

impl Chat<(), ()> {
    /// Starts an ordinary keep-alive Chat definition with structured output type `O`.
    pub fn builder<O>() -> ChatBuilder<O> {
        ChatBuilder {
            settings: ChatSettings::default(),
            agent: Agent::root(),
            errors: Vec::new(),
            state: (),
            services: None,
            _marker: PhantomData,
        }
    }
}

impl<O, S> ChatBuilder<O, S> {
    /// Sets initial application state for a new chat, replacing any previous value or state type.
    /// State is converted only at build and lives exclusively in the runtime afterward.
    pub fn state<T>(self, state: T) -> ChatBuilder<O, T> {
        ChatBuilder {
            settings: self.settings,
            agent: self.agent,
            errors: self.errors,
            state,
            services: self.services,
            _marker: PhantomData,
        }
    }

    /// Installs the pre-request compactor, replacing any previous one; never serialized.
    pub fn compactor(mut self, compactor: impl Compactor + 'static) -> Self {
        self.services = Some(self.services.unwrap_or_default().with_compactor(compactor));
        self
    }

    /// Installs the message history store, replacing any previous one; never serialized.
    pub fn store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.services = Some(self.services.unwrap_or_default().with_store(store));
        self
    }

    /// Sets the required provider/model URL, replacing any previous value.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.settings.model = model.into();
        self
    }
    /// Sets system instructions, replacing any previous value.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.settings.instructions = text.into();
        self
    }
    /// Sets provider options; credentials must remain in runtime Context dependencies.
    pub fn provider_config(mut self, config: impl Into<JsonValue>) -> Self {
        self.settings.provider_config = Some(config.into());
        self
    }
    /// Declares candidate tools; duplicate identities fail at build or restore.
    pub fn tools(mut self, build: fn(Toolset) -> Toolset) -> Self {
        self.agent = self.agent.tools(build);
        self
    }
    /// Sets default MCP references; requests may replace them, including with an empty list.
    pub fn resources(mut self, refs: impl IntoIterator<Item = McpResourceRef>) -> Self {
        self.settings.resources = refs.into_iter().collect();
        self
    }

    /// Registers the ordinary asynchronous agent-loop controller, at most once.
    pub fn control<Fut, E>(mut self, control: fn(AgentLoop<ChatRequest>, Context) -> Fut) -> Self
    where
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
    /// Caps accepted calls for the canonical tool input identity; errors accumulate until build.
    pub fn tool_budget<I: JsonSchema>(mut self, calls: u32) -> Self {
        match agent_tool_identity::<I>() {
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

impl<O, S> ChatBuilder<O, S>
where
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Validates and initializes a snapshot-ready chat without external calls or history writes.
    /// Consumes all configuration, moving initial state and services into the existing runtime.
    pub async fn build(self, ctx: Context) -> Result<Chat<ChatRequest, O, S>, GraphError> {
        let agent = finish(self.settings, self.agent, self.errors)?;
        let chat =
            Chat::from_definition(agent, self.state, ctx, Some(ChatRequest::validate)).await?;
        Ok(attach_services(chat, self.services))
    }
}

impl<O> ChatBuilder<O>
where
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Restores with matching settings and fresh context, without stepping or resolving resources.
    /// Configure services beforehand. Application state comes exclusively from the snapshot;
    /// restoration is unavailable on builders carrying non-unit initial state.
    pub fn restore<S>(
        self,
        snapshot: Snapshot,
        ctx: Context,
    ) -> Result<Chat<ChatRequest, O, S>, GraphError>
    where
        S: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let agent = finish(self.settings, self.agent, self.errors)?;
        let chat = Chat::restore_definition(agent, snapshot, ctx, Some(ChatRequest::validate))?;
        Ok(attach_services(chat, self.services))
    }
}

/// Rejects definition errors before lowering, resource resolution, or any VM mutation.
fn finish<O>(
    settings: ChatSettings,
    agent: Agent<ChatRequest>,
    mut errors: Vec<String>,
) -> Result<Agent<O>, GraphError>
where
    O: 'static + DeserializeOwned + JsonSchema + Send + Sync,
{
    if settings.model.trim().is_empty() {
        errors.push("model must not be empty".into());
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
    Ok(agent.configure_with(settings, configure_chat))
}

fn attach_services<I, O, S>(
    mut chat: Chat<I, O, S>,
    services: Option<RuntimeServices>,
) -> Chat<I, O, S> {
    if let Some(services) = services {
        chat.runtime = chat.runtime.with_runtime_services(services);
    }
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
async fn configure_chat(
    request: ChatRequest,
    settings: ChatSettings,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    let ChatRequest {
        message,
        memory,
        tools,
        resources,
    } = request;
    let mut config = AgentConfig::new(settings.model, settings.instructions, message).keep_alive();
    config.memory = memory;
    config.provider_config = settings.provider_config;
    config.max_output_tokens = settings.max_output_tokens;
    config.turn_budget = settings.turn_budget;
    config.tool_budgets = settings.tool_budgets;
    config.resources = resources.unwrap_or(settings.resources);
    if let Some(tools) = tools {
        config = config.tool_filter(ToolFilter::only(tools));
    }
    Ok(config)
}
