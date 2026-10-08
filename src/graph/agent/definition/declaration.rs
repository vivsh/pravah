use super::*;
use crate::clients::Message;
use crate::graph::agent::config::validate_resources;
use crate::graph::agent::{McpResourceRef, RequestedToolBudget, agent_tool_identity};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[cfg(test)]
#[path = "tests/declaration.rs"]
mod tests;

/// Graph-owned settings, decoded only when an invocation configures its agent.
#[derive(Default, Serialize, Deserialize, JsonSchema)]
pub(super) struct AgentSettings {
    key: Option<String>,
    model: String,
    provider_config: Option<JsonValue>,
    max_output_tokens: Option<u32>,
    turn_budget: Option<u32>,
    tool_budgets: Vec<RequestedToolBudget>,
    resources: Vec<McpResourceRef>,
}

impl<T> Agent<T> {
    /// Sets the required provider/model URL; later values replace earlier ones.
    /// Missing or empty models fail compilation. Cannot be combined with custom configure.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        if let Some(settings) = self.settings_mut() {
            settings.model = model.into();
        }
        self
    }

    /// Sets instructions for future invocations, replacing any earlier value.
    /// Changed instructions do not invalidate snapshots; committed invocations retain theirs.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        if self.settings_mut().is_some() {
            self.definition.instructions = Some(instructions.into());
        }
        self
    }

    /// Sets a runtime-wide conversation key; omission leaves graph agents frame-local.
    /// Empty keys fail compilation. Chat supplies its own default key when omitted.
    pub fn key(mut self, key: impl Into<String>) -> Self {
        if let Some(settings) = self.settings_mut() {
            settings.key = Some(key.into());
        }
        self
    }

    /// Sets opaque provider options, replacing any earlier value.
    /// Supplied options are graph data; keep credentials in runtime dependencies where possible.
    pub fn provider_config(mut self, config: impl Into<JsonValue>) -> Self {
        if let Some(settings) = self.settings_mut() {
            settings.provider_config = Some(config.into());
        }
        self
    }

    /// Caps generated tokens per model request; later values replace earlier ones.
    /// A final zero cap fails compilation.
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        if let Some(settings) = self.settings_mut() {
            settings.max_output_tokens = Some(tokens);
        }
        self
    }

    /// Selects MCP text references in supplied order, replacing earlier selections.
    /// Malformed or duplicate references fail compilation; resolution occurs at activation.
    pub fn resources(mut self, refs: impl IntoIterator<Item = McpResourceRef>) -> Self {
        if let Some(settings) = self.settings_mut() {
            settings.resources = refs.into_iter().collect();
        }
        self
    }

    /// Caps ordinary model turns, using existing forced conclusion at exhaustion.
    /// Zero or repeated budgets fail compilation; the budget resets per invocation.
    pub fn turn_budget(mut self, turns: u32) -> Self {
        if let Some(settings) = self.settings_mut() {
            if turns == 0 || settings.turn_budget.is_some() {
                self.definition
                    .errors
                    .push("invalid or repeated turn budget".into());
            } else {
                settings.turn_budget = Some(turns);
            }
        }
        self
    }

    /// Caps accepted calls for the canonical tool input identity.
    /// Zero, repeated or undeclared budgets fail compilation, including aliases by name.
    pub fn tool_budget<I: JsonSchema>(mut self, calls: u32) -> Self {
        match agent_tool_identity::<I>() {
            Ok(name) => self.tool_budget_named(name, calls),
            Err(error) => {
                self.definition.errors.push(error);
                self
            }
        }
    }

    /// Caps accepted calls for an explicit tool name, including JSON aliases.
    /// Zero, repeated or undeclared budgets fail compilation.
    pub fn tool_budget_named(mut self, name: impl Into<String>, calls: u32) -> Self {
        let name = name.into();
        if let Some(settings) = self.settings_mut() {
            if calls == 0
                || settings
                    .tool_budgets
                    .iter()
                    .any(|budget| budget.name == name)
            {
                self.definition
                    .errors
                    .push(format!("invalid or repeated tool budget for '{name}'"));
            } else {
                settings
                    .tool_budgets
                    .push(RequestedToolBudget { name, limit: calls });
            }
        }
        self
    }

    /// Finalizes declarative configuration with structured output `O`.
    /// Input is rendered as JSON text, including JSON quoting for strings.
    /// Errors accumulate for compilation. Use configure instead for dynamic settings or rendering.
    ///
    /// ```
    /// use pravah::{Agent, Flow, GraphError, compile};
    /// fn assistant(root: Agent<String>) -> Agent<String> {
    ///     root.model("openai:///gpt-5-mini")
    ///         .instructions("Answer concisely.")
    ///         .build()
    /// }
    /// let workflow = compile(|root: Flow<String>| root.agent(assistant))?;
    /// # Ok::<(), GraphError>(())
    /// ```
    pub fn build<O>(mut self) -> Agent<O>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + DeserializeOwned + JsonSchema + Send + Sync,
    {
        if self.definition.configure.is_some() {
            self.definition
                .errors
                .push("agent build must be terminal and cannot follow configure or build".into());
        } else {
            let settings = self.definition.settings.take().unwrap_or_default();
            let instructions = self.definition.instructions.take().unwrap_or_default();
            self.validate_settings(&settings);
            if self.definition.errors.is_empty() {
                return self.configure_with(settings, instructions, configure_declared::<T>);
            }
        }
        Agent {
            definition: self.definition,
            _marker: PhantomData,
        }
    }

    /// Reports shared declaration errors at Chat's existing fallible construction boundary.
    pub(crate) fn build_checked<O>(self) -> Result<Agent<O>, GraphError>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let agent = self.build();
        if agent.definition.errors.is_empty() {
            Ok(agent)
        } else {
            Err(GraphError::AgentConfigValidation(
                agent.definition.errors.join("; "),
            ))
        }
    }

    /// Borrows the sole construction settings owner, rejecting setters after finalization.
    fn settings_mut(&mut self) -> Option<&mut AgentSettings> {
        if self.definition.configure.is_some() {
            self.definition.errors.push(
                "declarative agent settings must precede build and cannot follow configure".into(),
            );
            return None;
        }
        Some(
            self.definition
                .settings
                .get_or_insert_with(AgentSettings::default),
        )
    }

    /// Validates the complete declaration before serialization or handler registration.
    fn validate_settings(&mut self, settings: &AgentSettings) {
        if settings.model.trim().is_empty() {
            self.definition
                .errors
                .push("model must not be empty".into());
        }
        if settings
            .key
            .as_ref()
            .is_some_and(|key| key.trim().is_empty())
        {
            self.definition
                .errors
                .push("conversation key must not be empty".into());
        }
        if settings.max_output_tokens == Some(0) {
            self.definition
                .errors
                .push("max output tokens must be positive".into());
        }
        if let Err(error) = validate_resources(&settings.resources) {
            self.definition.errors.push(error);
        }
        for budget in &settings.tool_budgets {
            if !self.tool_names().any(|name| name == budget.name) {
                self.definition
                    .errors
                    .push(format!("unknown agent tool '{}'", budget.name));
            }
        }
    }
}

/// Renders one input and combines graph settings through ordinary agent configuration.
async fn configure_declared<I: Serialize>(
    input: I,
    settings: AgentSettings,
    instructions: String,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    let content = serde_json::to_string(&input).map_err(|error| GraphError::ValueConversion {
        target: "agent user message".into(),
        reason: error.to_string(),
    })?;
    let mut config = AgentConfig::new(settings.model, instructions, Message::user(content));
    config.key = settings.key;
    config.provider_config = settings.provider_config;
    config.max_output_tokens = settings.max_output_tokens;
    config.turn_budget = settings.turn_budget;
    config.tool_budgets = settings.tool_budgets;
    config.resources = settings.resources;
    Ok(config)
}
