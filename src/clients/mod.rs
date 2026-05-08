use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[cfg(feature = "provider-anthropic")]
mod anthropic;
#[cfg(feature = "provider-gemini")]
mod gemini;
#[cfg(feature = "provider-genai")]
mod genai;
#[cfg(feature = "provider-ollama")]
mod ollama;
#[cfg(feature = "provider-openai")]
mod openai;
pub(super) mod schema;

pub use crate::tools::ToolDefinition;

/// Role of a message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    /// An assistant turn that contains tool calls.
    /// `content` on the enclosing [`Message`] holds the accompanying thought text (may be empty).
    AssistantToolCalls {
        calls: Vec<ToolCall>,
    },
    /// A tool result being fed back to the model.
    /// `call_id` must match the id from the originating [`ToolCall`].
    Tool {
        call_id: String,
    },
}

/// A single turn in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Token usage reported by the provider for this message (set on assistant turns only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: Role::User,
            content: content.into(),
            usage: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: Role::Assistant,
            content: content.into(),
            usage: None,
        }
    }

    pub fn tool_output(call_id: String, content: impl Into<String>) -> Self {
        Message {
            role: Role::Tool { call_id },
            content: content.into(),
            usage: None,
        }
    }

    /// Creates a message by JSON-serializing `value` as the content.
    pub fn from_json(role: Role, value: &impl serde::Serialize) -> Result<Self, serde_json::Error> {
        Ok(Message {
            role,
            content: serde_json::to_string(value)?,
            usage: None,
        })
    }

    pub fn with_usage(self, usage: TokenUsage) -> Self {
        Message {
            usage: Some(usage),
            ..self
        }
    }
}

/// A tool invocation requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlation id — must be echoed back in a [`Role::Tool`] message.
    pub id: String,
    pub name: String,
    pub args: Value,
    /// Opaque bytes returned by Gemini 2.5 thinking models alongside a tool call.
    /// Must be echoed back verbatim on the next turn — do not decode or reconstruct.
    /// `None` for all non-Gemini providers and for Gemini responses without thinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signatures: Option<Vec<String>>,
}

/// Raw response from one LLM call.
#[derive(Debug)]
pub enum ClientOutput {
    /// Model produced structured output (JSON value).
    Output(Value),
    /// Model requested one or more tool calls.
    ToolCalls {
        /// Reasoning/thought text the model emitted alongside the tool calls (if any).
        thought: Option<String>,
        calls: Vec<ToolCall>,
    },
}

/// Provider-normalized response from one LLM call.
#[derive(Debug)]
pub struct ClientResponse {
    pub output: ClientOutput,
    pub usage: Option<TokenUsage>,
    pub provider: Provider,
    pub provider_model: Option<String>,
    pub raw_metadata: Option<Value>,
}

impl ClientResponse {
    pub fn new(provider: Provider, output: ClientOutput) -> Self {
        Self {
            output,
            usage: None,
            provider,
            provider_model: None,
            raw_metadata: None,
        }
    }

    pub fn with_usage(mut self, usage: Option<TokenUsage>) -> Self {
        self.usage = usage;
        self
    }

    pub fn with_provider_model(mut self, provider_model: Option<String>) -> Self {
        self.provider_model = provider_model;
        self
    }

    pub fn with_raw_metadata(mut self, raw_metadata: Option<Value>) -> Self {
        self.raw_metadata = raw_metadata;
        self
    }
}

/// Token counts reported by the provider for a single call.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Tokens consumed by the input (prompt, history, tool definitions).
    pub input: Option<u32>,
    /// Tokens produced by the model in the response.
    pub output: Option<u32>,
}

impl TokenUsage {
    pub fn total(&self) -> Option<u32> {
        match (self.input, self.output) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        }
    }
}

/// Errors produced during a [`Client::execute`] call.
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("failed to serialize input: {0}")]
    Serialize(#[source] serde_json::Error),
    #[error("failed to deserialize output: {source}\nraw response: {raw}")]
    Deserialize {
        #[source]
        source: serde_json::Error,
        raw: String,
    },
    #[error("LLM call failed: {0}")]
    Llm(String),
    #[error("LLM returned an empty response")]
    EmptyResponse,
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("No tool calls found: {0:?}")]
    MissingToolCalls(Option<String>),
    #[error("provider '{provider:?}' does not support capability '{capability}'")]
    UnsupportedCapability {
        provider: Provider,
        capability: String,
    },
    #[error("invalid LLM URL: {0}")]
    InvalidUrl(String),
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// LLM provider backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    Gemini,
    Ollama,
    OpenAi,
    Anthropic,
    Genai,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Gemini => "gemini",
            Provider::Ollama => "ollama",
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Genai => "genai",
        }
    }
}

/// Parsed representation of an LLM URL.
///
/// Format: `scheme://[key@]model-name` for cloud providers,
/// or `ollama://host:port/model-name` for local Ollama.
///
/// Examples:
/// - `gemini://gemini-2.5-flash-lite`          (key from `GEMINI_API_KEY`)
/// - `gemini://mykey@gemini-2.5-flash-lite`    (key inline)
/// - `openai://gpt-4o`                         (key from `OPENAI_API_KEY`)
/// - `anthropic://claude-opus-4-5`             (key from `ANTHROPIC_API_KEY`)
/// - `claude://claude-opus-4-5`                (alias for `anthropic://...`)
/// - `ollama://localhost:11434/qwen3:8b`        (no key, base_url extracted)
#[derive(Debug, Clone)]
pub struct LlmUrl {
    pub provider: Provider,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

impl LlmUrl {
    /// Parses an LLM URL string into a structured [`LlmUrl`].
    pub fn parse(s: &str) -> Result<Self, ClientError> {
        let (scheme, rest) = s.split_once("://").ok_or_else(|| {
            ClientError::InvalidUrl(format!(
                "missing scheme in '{s}'; expected e.g. gemini://model-name"
            ))
        })?;

        let provider = match scheme {
            "gemini" => Provider::Gemini,
            "ollama" => Provider::Ollama,
            "openai" => Provider::OpenAi,
            "anthropic" | "claude" => Provider::Anthropic,
            "genai" => Provider::Genai,
            other => {
                return Err(ClientError::InvalidUrl(format!(
                    "unknown provider '{other}'; expected gemini, ollama, openai, anthropic, claude, or genai"
                )));
            }
        };

        match provider {
            Provider::Ollama => {
                let (authority, model) = rest.split_once('/').ok_or_else(|| {
                    ClientError::InvalidUrl(
                        "ollama URL must have format ollama://host:port/model-name".into(),
                    )
                })?;
                if model.is_empty() {
                    return Err(ClientError::InvalidUrl(
                        "missing model name in ollama URL".into(),
                    ));
                }
                Ok(LlmUrl {
                    provider,
                    model: model.to_owned(),
                    api_key: None,
                    base_url: Some(format!("http://{authority}")),
                })
            }
            Provider::Genai => {
                if rest.is_empty() {
                    return Err(ClientError::InvalidUrl(
                        "genai URL must include a provider/model or model name".into(),
                    ));
                }
                Ok(LlmUrl {
                    provider,
                    model: rest.to_owned(),
                    api_key: None,
                    base_url: None,
                })
            }
            _ => {
                let (api_key, model) = if let Some((key, m)) = rest.split_once('@') {
                    (Some(key.to_owned()), m.to_owned())
                } else {
                    (None, rest.to_owned())
                };
                if model.is_empty() {
                    return Err(ClientError::InvalidUrl(format!(
                        "missing model name in '{s}'"
                    )));
                }
                Ok(LlmUrl {
                    provider,
                    model,
                    api_key,
                    base_url: None,
                })
            }
        }
    }
}

/// Controls whether the model must, may, or may not invoke tools.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// Provider decides (usually auto-selects based on context).
    #[default]
    Auto,
    /// Model must call at least one tool.
    Required,
    /// No tool calls; model produces text/structured output only.
    Disabled,
}

/// Call-time configuration for a [`Client::execute`] invocation.
#[derive(Debug, Clone, Default)]
pub struct ClientOptions {
    /// Optional label for tracing/logging.
    pub name: Option<String>,
    /// System-level preamble sent before the conversation history.
    pub preamble: Option<String>,
    /// Tools the model may invoke. Empty means structured-output mode.
    pub tools: Vec<ToolDefinition>,
    /// Enable chain-of-thought reasoning (provider-specific).
    pub thinking: bool,
    /// Whether tool invocation is forced, optional, or disabled.
    pub tool_choice: ToolChoice,
    /// JSON Schema describing the structure of the user-message input.
    /// Appended to the preamble as a hint for the model.
    pub input_schema: Option<Value>,
    /// JSON Schema describing the expected structured output.
    /// Used by backends to configure provider-native structured-output mode.
    pub output_schema: Option<Value>,
}

impl ClientOptions {
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the JSON Schema describing the structure of the user-message input.
    ///
    /// Backends may append this to the preamble as a hint to help the model
    /// understand the shape of each user turn.
    pub fn with_input_schema(mut self, schema: Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    /// Sets the JSON Schema describing the expected structured output.
    ///
    /// Backends use this to configure provider-native structured-output mode
    /// when no tools are present.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Constructs a client backed by the appropriate provider for the given LLM URL.
    ///
    pub fn create(self, llm_url: &str) -> Result<Box<dyn Client>, ClientError> {
        let url = LlmUrl::parse(llm_url)?;
        match url.provider {
            #[cfg(feature = "provider-gemini")]
            Provider::Gemini => gemini::new_client(&url, self),
            #[cfg(not(feature = "provider-gemini"))]
            Provider::Gemini => provider_feature_disabled(url.provider),

            #[cfg(feature = "provider-openai")]
            Provider::OpenAi => openai::new_client(&url, self),
            #[cfg(not(feature = "provider-openai"))]
            Provider::OpenAi => provider_feature_disabled(url.provider),

            #[cfg(feature = "provider-anthropic")]
            Provider::Anthropic => anthropic::new_client(&url, self),
            #[cfg(not(feature = "provider-anthropic"))]
            Provider::Anthropic => provider_feature_disabled(url.provider),

            #[cfg(feature = "provider-ollama")]
            Provider::Ollama => ollama::new_client(&url, self),
            #[cfg(not(feature = "provider-ollama"))]
            Provider::Ollama => provider_feature_disabled(url.provider),

            #[cfg(feature = "provider-genai")]
            Provider::Genai => genai::create_client(&url, self),
            #[cfg(not(feature = "provider-genai"))]
            Provider::Genai => provider_feature_disabled(url.provider),
        }
    }
}

#[allow(dead_code)]
fn provider_feature_disabled(provider: Provider) -> Result<Box<dyn Client>, ClientError> {
    Err(ClientError::UnsupportedCapability {
        provider,
        capability: "provider feature is disabled".to_string(),
    })
}

#[allow(dead_code)]
pub(super) fn validate_tools(
    provider: Provider,
    tools: &[ToolDefinition],
) -> Result<(), ClientError> {
    let mut seen = std::collections::HashSet::new();
    for tool in tools {
        if tool.name.trim().is_empty() {
            return Err(ClientError::Validation(
                "tool name must not be empty".into(),
            ));
        }
        if !seen.insert(tool.name.as_str()) {
            return Err(ClientError::Validation(format!(
                "duplicate tool name '{}'",
                tool.name
            )));
        }
        if !tool.parameters.is_object() {
            return Err(ClientError::UnsupportedCapability {
                provider,
                capability: format!("tool '{}' has a non-object JSON schema", tool.name),
            });
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn parse_json_output(text: &str) -> Result<Value, ClientError> {
    serde_json::from_str(text).map_err(|e| {
        tracing::error!(model_output = %text, parse_error = %e, "LLM output deserialization failed");
        ClientError::Deserialize {
            source: e,
            raw: text.to_string(),
        }
    })
}

/// Conversation history with optional sliding-window eviction and token accounting.
///
/// Usage is auto-extracted from pushed [`Message`]s that carry a `usage` field.
/// The first `pinned` messages are never evicted (default: 1, to preserve the
/// initial task/seed User message).
#[derive(Debug, Clone, Default)]
pub struct ClientHistory {
    messages: Vec<Message>,
    max_turns: Option<usize>,
    pinned: usize,
    last_usage: Option<TokenUsage>,
    total_input: Option<u32>,
    total_output: Option<u32>,
}

impl ClientHistory {
    pub fn new(max_turns: Option<usize>) -> Self {
        Self {
            max_turns,
            pinned: 1,
            ..Default::default()
        }
    }

    pub fn with_pinned(max_turns: Option<usize>, pinned: usize) -> Self {
        Self {
            max_turns,
            pinned,
            ..Default::default()
        }
    }

    /// Appends `msg`, auto-extracts its token usage, then evicts if over the turn limit.
    pub fn push(&mut self, msg: Message) {
        if let Some(u) = msg.usage {
            self.last_usage = Some(u);
            self.total_input = add_opt(self.total_input, u.input);
            self.total_output = add_opt(self.total_output, u.output);
        }
        self.messages.push(msg);
        self.evict_if_needed();
    }

    pub fn extend(&mut self, msgs: impl IntoIterator<Item = Message>) {
        for msg in msgs {
            self.push(msg);
        }
    }

    pub fn as_slice(&self) -> &[Message] {
        &self.messages
    }

    pub fn last_role(&self) -> Option<&Role> {
        self.messages.last().map(|m| &m.role)
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Number of complete assistant turns after the pinned region.
    pub fn turn_count(&self) -> usize {
        let start = self.pinned.min(self.messages.len());
        self.messages[start..]
            .iter()
            .filter(|m| matches!(m.role, Role::Assistant | Role::AssistantToolCalls { .. }))
            .count()
    }

    pub fn validate(&self) -> Result<(), ClientError> {
        if matches!(self.last_role(), Some(Role::AssistantToolCalls { .. })) {
            return Err(ClientError::Validation(
                "history ends with assistant tool calls without tool results".into(),
            ));
        }
        Ok(())
    }

    pub fn last_usage(&self) -> Option<TokenUsage> {
        self.last_usage
    }

    pub fn total_input(&self) -> Option<u32> {
        self.total_input
    }

    pub fn total_output(&self) -> Option<u32> {
        self.total_output
    }

    /// Cumulative input + output tokens, or `None` if either is missing.
    pub fn total_usage(&self) -> Option<u32> {
        match (self.total_input, self.total_output) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        }
    }

    fn first_turn_end_exclusive(&self) -> Option<usize> {
        let start = self.pinned;
        if start >= self.messages.len() {
            return None;
        }
        match &self.messages[start].role {
            Role::AssistantToolCalls { .. } => {
                let mut end = start + 1;
                while end < self.messages.len()
                    && matches!(self.messages[end].role, Role::Tool { .. })
                {
                    end += 1;
                }
                Some(end)
            }
            Role::Assistant => Some(start + 1),
            _ => None,
        }
    }

    fn evict_if_needed(&mut self) {
        let max = match self.max_turns {
            Some(m) => m,
            None => return,
        };
        while self.turn_count() > max {
            match self.first_turn_end_exclusive() {
                Some(end) => {
                    self.messages.drain(self.pinned..end);
                }
                None => break,
            }
        }
    }
}

fn add_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

/// Provider-agnostic stateless LLM client.
///
/// Options are fixed at construction time and owned by the implementation.
/// Callers push input messages to history before calling `execute`, and
/// push tool-result messages after dispatch.
#[async_trait]
pub trait Client: Send + Sync {
    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError>;
}

/// Creates a [`Client`] from a model URL and call-time options.
///
/// Implement this trait to inject alternative backends (e.g. mocks) into a
/// [`crate::flows`] pipeline. The default implementation delegates to
/// [`ClientOptions::create`].
pub trait ClientFactory: Send + Sync + 'static {
    fn create(
        &self,
        model_url: &str,
        options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError>;
}

/// Default factory — creates real provider clients via [`ClientOptions::create`].
pub struct DefaultClientFactory;

impl ClientFactory for DefaultClientFactory {
    fn create(
        &self,
        model_url: &str,
        options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError> {
        options.create(model_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input: Some(input),
            output: Some(output),
        }
    }

    fn msg_with_usage(u: TokenUsage) -> Message {
        Message {
            role: Role::Assistant,
            content: "hi".into(),
            usage: Some(u),
        }
    }

    fn atc_msg(calls: Vec<ToolCall>) -> Message {
        Message {
            role: Role::AssistantToolCalls { calls },
            content: String::new(),
            usage: None,
        }
    }

    fn tool_msg(call_id: &str) -> Message {
        Message {
            role: Role::Tool {
                call_id: call_id.into(),
            },
            content: "ok".into(),
            usage: None,
        }
    }

    fn dummy_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "f".into(),
            args: serde_json::json!({}),
            thought_signatures: None,
        }
    }

    /// `push` records the most recently seen token usage.
    #[test]
    fn push_records_last_usage() {
        let mut h = ClientHistory::new(None);
        h.push(msg_with_usage(usage(10, 5)));
        let u = h.last_usage().unwrap();
        assert_eq!(u.input, Some(10));
        assert_eq!(u.output, Some(5));
    }

    /// `push` accumulates total input and output tokens across all messages.
    #[test]
    fn push_accumulates_totals() {
        let mut h = ClientHistory::new(None);
        h.push(msg_with_usage(usage(10, 5)));
        h.push(msg_with_usage(usage(20, 8)));
        assert_eq!(h.total_input(), Some(30));
        assert_eq!(h.total_output(), Some(13));
        assert_eq!(h.total_usage(), Some(43));
    }

    /// `turn_count` counts only assistant turns after the pinned region.
    #[test]
    fn turn_count_counts_assistant_turns() {
        let mut h = ClientHistory::new(None);
        h.push(Message::user("seed"));
        h.push(atc_msg(vec![dummy_call("1")]));
        h.push(tool_msg("1"));
        assert_eq!(h.turn_count(), 1);
        h.push(atc_msg(vec![dummy_call("2")]));
        h.push(tool_msg("2"));
        assert_eq!(h.turn_count(), 2);
    }

    /// Sliding window evicts the oldest non-pinned turn when limit is exceeded.
    #[test]
    fn sliding_evicts_oldest_turn() {
        let mut h = ClientHistory::new(Some(1));
        h.push(Message::user("seed"));
        h.push(atc_msg(vec![dummy_call("1")]));
        h.push(tool_msg("1"));
        h.push(atc_msg(vec![dummy_call("2")]));
        h.push(tool_msg("2"));
        assert_eq!(h.turn_count(), 1);
        assert!(matches!(h.as_slice()[0].role, Role::User));
    }

    /// Pinned messages are never evicted even when the window is exceeded.
    #[test]
    fn pinned_messages_survive_eviction() {
        let mut h = ClientHistory::with_pinned(Some(1), 2);
        h.push(Message::user("seed"));
        h.push(Message::user("ctx"));
        h.push(atc_msg(vec![dummy_call("1")]));
        h.push(tool_msg("1"));
        h.push(atc_msg(vec![dummy_call("2")]));
        h.push(tool_msg("2"));
        assert_eq!(h.as_slice().len(), 4);
        assert!(matches!(h.as_slice()[0].role, Role::User));
        assert!(matches!(h.as_slice()[1].role, Role::User));
    }

    /// `validate` returns an error when history ends with unresolved tool calls.
    #[test]
    fn validate_rejects_dangling_tool_calls() {
        let mut h = ClientHistory::new(None);
        h.push(Message::user("seed"));
        h.push(atc_msg(vec![dummy_call("1")]));
        assert!(matches!(h.validate(), Err(ClientError::Validation(_))));
    }

    /// `last_role` returns the role of the most recently pushed message.
    #[test]
    fn last_role_returns_correct_role() {
        let mut h = ClientHistory::new(None);
        assert!(h.last_role().is_none());
        h.push(Message::user("hi"));
        assert!(matches!(h.last_role(), Some(Role::User)));
    }

    /// `total_usage` returns `None` when either input or output is missing.
    #[test]
    fn total_usage_requires_both_values() {
        let mut h = ClientHistory::new(None);
        h.push(Message {
            role: Role::Assistant,
            content: "x".into(),
            usage: Some(TokenUsage {
                input: Some(5),
                output: None,
            }),
        });
        assert_eq!(h.total_input(), Some(5));
        assert_eq!(h.total_output(), None);
        assert_eq!(h.total_usage(), None);
    }

    /// `LlmUrl::parse` correctly parses a gemini URL without an API key.
    #[test]
    fn parse_gemini_url_no_key() {
        let url = LlmUrl::parse("gemini://gemini-2.5-flash-lite").unwrap();
        assert_eq!(url.provider, Provider::Gemini);
        assert_eq!(url.model, "gemini-2.5-flash-lite");
        assert!(url.api_key.is_none());
        assert!(url.base_url.is_none());
    }

    /// `LlmUrl::parse` extracts an inline API key.
    #[test]
    fn parse_gemini_url_with_key() {
        let url = LlmUrl::parse("gemini://mykey@gemini-2.5-flash-lite").unwrap();
        assert_eq!(url.api_key.as_deref(), Some("mykey"));
        assert_eq!(url.model, "gemini-2.5-flash-lite");
    }

    /// `LlmUrl::parse` extracts host and model from an ollama URL.
    #[test]
    fn parse_ollama_url() {
        let url = LlmUrl::parse("ollama://localhost:11434/qwen3:8b").unwrap();
        assert_eq!(url.provider, Provider::Ollama);
        assert_eq!(url.model, "qwen3:8b");
        assert_eq!(url.base_url.as_deref(), Some("http://localhost:11434"));
        assert!(url.api_key.is_none());
    }

    /// `anthropic://` and `claude://` both select the Anthropic provider.
    #[test]
    fn parse_anthropic_aliases() {
        let anthropic = LlmUrl::parse("anthropic://claude-sonnet-4-5").unwrap();
        let claude = LlmUrl::parse("claude://claude-sonnet-4-5").unwrap();
        assert_eq!(anthropic.provider, Provider::Anthropic);
        assert_eq!(claude.provider, Provider::Anthropic);
    }

    /// Provider schemes are authoritative even for model names genai would not infer correctly.
    #[test]
    fn parse_openai_custom_model_stays_openai() {
        let url = LlmUrl::parse("openai://key@ft-custom-agent-model").unwrap();
        assert_eq!(url.provider, Provider::OpenAi);
        assert_eq!(url.api_key.as_deref(), Some("key"));
        assert_eq!(url.model, "ft-custom-agent-model");
    }

    /// Tool schemas must be JSON objects and duplicate names are rejected before provider calls.
    #[test]
    fn validate_tools_rejects_bad_definitions() {
        let non_object = vec![ToolDefinition {
            name: "bad".into(),
            description: "bad".into(),
            parameters: serde_json::json!(true),
        }];
        assert!(matches!(
            validate_tools(Provider::OpenAi, &non_object),
            Err(ClientError::UnsupportedCapability { .. })
        ));

        let duplicate = vec![
            ToolDefinition {
                name: "dup".into(),
                description: "one".into(),
                parameters: serde_json::json!({ "type": "object" }),
            },
            ToolDefinition {
                name: "dup".into(),
                description: "two".into(),
                parameters: serde_json::json!({ "type": "object" }),
            },
        ];
        assert!(matches!(
            validate_tools(Provider::OpenAi, &duplicate),
            Err(ClientError::Validation(_))
        ));
    }

    /// Disabled optional providers fail explicitly instead of silently falling back.
    #[cfg(not(feature = "provider-genai"))]
    #[test]
    fn disabled_genai_provider_returns_capability_error() {
        let err = match ClientOptions::default().create("genai://openai/gpt-4o") {
            Ok(_) => panic!("genai provider should be disabled"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            ClientError::UnsupportedCapability {
                provider: Provider::Genai,
                ..
            }
        ));
    }

    /// `LlmUrl::parse` returns an error for an unknown provider scheme.
    #[test]
    fn parse_unknown_scheme_errors() {
        assert!(matches!(
            LlmUrl::parse("unknown://model"),
            Err(ClientError::InvalidUrl(_))
        ));
    }

    /// `LlmUrl::parse` returns an error when no scheme separator is present.
    #[test]
    fn parse_missing_scheme_errors() {
        assert!(matches!(
            LlmUrl::parse("gemini-2.5-flash-lite"),
            Err(ClientError::InvalidUrl(_))
        ));
    }
}
