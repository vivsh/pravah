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
///
/// Wire-format only — carries no pravah-internal metadata (session, agent).
/// Metadata lives on [`crate::flows::HistoryEntry`].
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

// ── Embedding types ──────────────────────────────────────────────────────────

/// Task hint for embedding optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedTaskType {
    RetrievalDocument,
    RetrievalQuery,
    SemanticSimilarity,
    Classification,
    Clustering,
    QuestionAnswering,
    FactVerification,
    CodeRetrievalQuery,
}

/// Request to generate a text embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedRequest {
    pub input: String,
    pub task_type: Option<EmbedTaskType>,
    /// Document title hint (improves retrieval quality on some models).
    pub title: Option<String>,
    /// Truncate the output vector to this many dimensions.
    pub output_dimensionality: Option<i32>,
    /// Provider-specific options serialized as a JSON object.
    pub provider_config: Option<serde_json::Value>,
}

/// Embedding vector returned by [`Client::embed`].
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    pub values: Vec<f32>,
}

// ── Client trait ──────────────────────────────────────────────────────────────

/// Provider-agnostic stateless LLM client.
///
/// Options are fixed at construction time and owned by the implementation.
/// Callers push input messages to history before calling `execute`, and
/// push tool-result messages after dispatch.
#[async_trait]
pub trait Client: Send + Sync {
    /// The provider backing this client instance.
    fn provider(&self) -> Provider;

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError>;

    /// Generate a text embedding vector for `request.input`.
    ///
    /// Providers that do not support embeddings return
    /// [`ClientError::UnsupportedCapability`] by default.
    async fn embed(&self, _request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        Err(ClientError::UnsupportedCapability {
            provider: self.provider(),
            capability: "embeddings".into(),
        })
    }

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
