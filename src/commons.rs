use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::clients::{Message, Role};
use crate::context::Context;
use crate::flows::FlowError;

/// Runtime settings returned by [`Agent::configure`].
pub struct AgentConfig {
    /// Prompt sent before each model turn.
    pub preamble: String,
    /// Model URL used to create the client.
    pub model_url: String,
    /// When true, repeated invocations of this agent share a stable session id,
    /// keeping the full conversation history visible across loop iterations.
    pub keep_alive: bool,
    /// Maximum number of LLM dispatch turns. On the final turn a reminder message
    /// is appended to the outgoing messages to prompt the agent to submit its answer.
    /// Has no effect on agents with no tools.
    pub turn_budget: Option<u32>,
    /// Overrides the default last-turn reminder injected when `turn_budget` is reached.
    /// When `None` a provider-appropriate default is used.
    pub turn_budget_message: Option<String>,
}

impl AgentConfig {
    /// Builds an agent config with no tools.
    pub fn new(preamble: impl Into<String>, model_url: impl Into<String>) -> Self {
        Self {
            preamble: preamble.into(),
            model_url: model_url.into(),
            keep_alive: false,
            turn_budget: None,
            turn_budget_message: None,
        }
    }

    /// Keeps the agent's session alive across repeated invocations in a loop.
    pub fn keep_alive(mut self) -> Self {
        self.keep_alive = true;
        self
    }

    /// Sets the maximum number of LLM dispatch turns for this agent.
    pub fn with_turn_budget(mut self, n: u32) -> Self {
        self.turn_budget = Some(n);
        self
    }

    /// Overrides the last-turn reminder message injected when the budget is reached.
    pub fn with_turn_budget_message(mut self, msg: impl Into<String>) -> Self {
        self.turn_budget_message = Some(msg.into());
        self
    }
}

/// Implemented by every agent input type.
pub trait Agent: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Value the agent must produce to finish.
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Graph node id. Defaults to the schema name.
    fn node_id() -> String {
        Self::schema_name()
    }

    /// Builds the first user message for this agent invocation.
    fn to_message(self, _ctx: &Context) -> Result<Message, FlowError> {
        Message::from_json(Role::User, &self).map_err(FlowError::Serialize)
    }

    /// Returns the runtime settings for this agent.
    fn configure() -> AgentConfig;
}

pub(crate) fn make_agent_message<A: Agent>(
    value: Value,
    ctx: &Context,
) -> Result<Message, FlowError> {
    let input: A = serde_json::from_value(value).map_err(FlowError::Deserialize)?;
    let message = input.to_message(ctx)?;
    if !matches!(message.role, Role::User) {
        return Err(FlowError::Internal {
            handler: "make_agent_message",
            detail: format!(
                "agent '{}' to_message must return a user message",
                A::node_id()
            ),
        });
    }
    Ok(message)
}
