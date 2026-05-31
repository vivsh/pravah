use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::clients::{Message, Role};
use crate::context::Context;
use crate::flows::{FlowError};

/// Controls whether structured output is obtained via a synthetic exit tool call
/// rather than `response_mime_type`.
///
/// Needed for providers that reject function calling and `response_mime_type`
/// simultaneously (Gemini < 3.1, Ollama).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ExitToolMode {
    /// Always use the exit-tool path regardless of model URL.
    Always,
    /// Never use the exit-tool path regardless of model URL.
    Never,
    /// Detect automatically from the model URL. This is the default.
    ///
    /// Currently resolves to `true` for all `ollama://` URLs and for
    /// `gemini://` URLs whose model version is below 3.1.
    #[default]
    Auto,
}

impl ExitToolMode {
    /// Returns `true` if the exit-tool path should be used for `model_url`.
    pub(crate) fn should_use(&self, model_url: &str) -> bool {
        match self {
            ExitToolMode::Always => true,
            ExitToolMode::Never => false,
            ExitToolMode::Auto => {
                if model_url.starts_with("ollama://") {
                    return true;
                }
                if model_url.starts_with("gemini://") {
                    let model = model_url.trim_start_matches("gemini://").trim_start_matches('/');
                    return gemini_needs_exit_tool(model);
                }
                false
            }
        }
    }
}

/// Returns `true` when `model` names a Gemini generation that does not support
/// combining function calling with `response_mime_type`.
///
/// Structured output + function calling is documented as available only from
/// Gemini 3.1 onwards. Versions below 3.1 require the exit-tool path.
/// Returns `false` for unrecognised strings to avoid breaking unknown models.
fn gemini_needs_exit_tool(model: &str) -> bool {
    let model = model.strip_prefix("models/").unwrap_or(model);
    let model = model.strip_prefix("gemini-").unwrap_or(model);
    // The version is the first dash-separated segment and uses dots internally:
    // e.g. "gemini-2.5-pro" → strip "gemini-" → "2.5-pro" → first segment "2.5".
    let version = model.split('-').next().unwrap_or(model);
    let mut parts = version.split('.');
    let major: u32 = match parts.next().and_then(|s| s.parse().ok()) {
        Some(n) => n,
        None => return false,
    };
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (major, minor) < (3, 1)
}

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
    /// Controls whether structured output uses a synthetic exit-tool call instead of
    /// `response_mime_type`. Defaults to [`ExitToolMode::Auto`].
    pub exit_tool: ExitToolMode,
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
            exit_tool: ExitToolMode::Auto,
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

    /// Forces the exit-tool path on for this agent, regardless of model URL.
    pub fn with_exit_tool(mut self) -> Self {
        self.exit_tool = ExitToolMode::Always;
        self
    }

    /// Sets the exit-tool detection mode explicitly.
    pub fn with_exit_tool_mode(mut self, mode: ExitToolMode) -> Self {
        self.exit_tool = mode;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Auto` mode selects exit-tool for all Ollama URLs.
    #[test]
    fn auto_ollama_always_uses_exit_tool() {
        assert!(ExitToolMode::Auto.should_use("ollama://qwen3:8b"));
        assert!(ExitToolMode::Auto.should_use("ollama://localhost:11434/llama3.1"));
    }

    /// `Auto` mode selects exit-tool for Gemini models below version 3.1.
    #[test]
    fn auto_gemini_below_3_1_uses_exit_tool() {
        assert!(ExitToolMode::Auto.should_use("gemini://gemini-2.5-pro"));
        assert!(ExitToolMode::Auto.should_use("gemini://gemini-2.0-flash"));
        assert!(ExitToolMode::Auto.should_use("gemini://gemini-3.0-flash"));
        assert!(ExitToolMode::Auto.should_use("gemini:///gemini-2.5-flash-lite"));
    }

    /// `Auto` mode skips exit-tool for Gemini 3.1 and above.
    #[test]
    fn auto_gemini_3_1_and_above_skips_exit_tool() {
        assert!(!ExitToolMode::Auto.should_use("gemini://gemini-3.1-flash"));
        assert!(!ExitToolMode::Auto.should_use("gemini://gemini-3.5-flash"));
        assert!(!ExitToolMode::Auto.should_use("gemini:///gemini-3.1-pro-preview"));
    }

    /// `Auto` mode does not select exit-tool for OpenAI or Anthropic.
    #[test]
    fn auto_other_providers_skip_exit_tool() {
        assert!(!ExitToolMode::Auto.should_use("openai://gpt-4o"));
        assert!(!ExitToolMode::Auto.should_use("anthropic://claude-opus-4-5"));
    }

    /// `Always` forces exit-tool regardless of URL; `Never` disables it regardless.
    #[test]
    fn always_and_never_override_detection() {
        assert!(ExitToolMode::Always.should_use("openai://gpt-4o"));
        assert!(!ExitToolMode::Never.should_use("ollama://qwen3:8b"));
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

    /// Returns runtime context appended to the system prompt on the first turn.
    ///
    /// Override to inject dynamic facts (current date, user identity, retrieved
    /// memories, etc.) into the system prompt at dispatch time. The returned
    /// string is inserted between the static preamble and the input-schema hint.
    /// Returns `None` by default (no additional context).
    fn environment(_ctx: &Context) -> Option<String> {
        None
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
            detail: format!("agent '{}' to_message must return a user message", A::node_id()),
        });
    }
    Ok(message)
}
