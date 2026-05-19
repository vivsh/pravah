use base64::Engine;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::context::Context;

/// A binary or URL attachment that can accompany a message.
/// Attachments are carried through the history layer and translated into
/// provider-specific wire formats by each client adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Attachment {
    /// Inline binary data (e.g. a screenshot).
    /// `data` must be base64-encoded.
    Inline {
        mime_type: String,
        data: String,
    },
    /// File path resolved through the current [`Context`].
    File {
        mime_type: String,
        path: String,
    },
    /// Reference to a publicly accessible URL.
    Url {
        mime_type: String,
        url: String,
    },
}

#[cfg(feature = "provider-anthropic")]
mod anthropic;
#[cfg(feature = "provider-gemini")]
mod gemini;
#[cfg(feature = "provider-ollama")]
mod ollama;
#[cfg(feature = "provider-openai")]
mod openai;
pub(super) mod schema;

pub use crate::tools::ToolDefinition;

/// Role of a message in provider-facing history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Assistant turn that carries tool calls.
    /// The enclosing [`Message`] keeps any accompanying text in `content`.
    AssistantToolCalls {
        calls: Vec<ToolCall>,
    },
    /// Tool result fed back to the model.
    /// `call_id` must match the originating [`ToolCall`].
    Tool {
        call_id: String,
    },
}

/// One history message prepared for provider dispatch.
/// Pravah metadata such as session and agent ids lives on [`crate::flows::HistoryEntry`].
/// File attachments are materialized before the message reaches a provider adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Attachments (images, files) to send alongside the message content.
    /// Serialization is skipped when empty so existing stored history is unaffected.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,
    /// Provider-reported token usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: Role::User,
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: Role::Assistant,
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    pub fn tool_output(call_id: String, content: impl Into<String>) -> Self {
        Message {
            role: Role::Tool { call_id },
            content: content.into(),
            attachments: Vec::new(),
            usage: None,
        }
    }

    /// Builds a message by JSON-encoding `value`.
    pub fn from_json(role: Role, value: &impl serde::Serialize) -> Result<Self, serde_json::Error> {
        Ok(Message {
            role,
            content: serde_json::to_string(value)?,
            attachments: Vec::new(),
            usage: None,
        })
    }

    pub fn with_usage(self, usage: TokenUsage) -> Self {
        Message {
            usage: Some(usage),
            ..self
        }
    }

    pub fn with_attachment(mut self, attachment: Attachment) -> Self {
        self.attachments.push(attachment);
        self
    }

    pub fn with_inline(mut self, mime_type: impl Into<String>, bytes: impl AsRef<[u8]>) -> Self {
        self.attachments.push(Attachment::Inline {
            mime_type: mime_type.into(),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
        });
        self
    }

    pub fn with_file(mut self, mime_type: impl Into<String>, path: impl Into<String>) -> Self {
        self.attachments.push(Attachment::File {
            mime_type: mime_type.into(),
            path: path.into(),
        });
        self
    }

    pub fn with_url(mut self, mime_type: impl Into<String>, url: impl Into<String>) -> Self {
        self.attachments.push(Attachment::Url {
            mime_type: mime_type.into(),
            url: url.into(),
        });
        self
    }
}

async fn materialize_attachment(
    attachment: &Attachment,
    ctx: &Context,
) -> Result<Attachment, ClientError> {
    match attachment {
        Attachment::Inline { mime_type, data } => {
            Ok(Attachment::Inline {
                mime_type: mime_type.clone(),
                data: data.clone(),
            })
        }
        Attachment::Url { mime_type, url } => {
            Ok(Attachment::Url {
                mime_type: mime_type.clone(),
                url: url.clone(),
            })
        }
        Attachment::File { mime_type, path } => {
            let resolved = ctx.resolve(path).map_err(|e| {
                ClientError::Validation(format!("attachment path '{path}' is invalid: {e}"))
            })?;
            let bytes = tokio::fs::read(&resolved).await.map_err(|e| {
                ClientError::Validation(format!(
                    "failed to read attachment file '{}': {e}",
                    resolved.display()
                ))
            })?;
            Ok(Attachment::Inline {
                mime_type: mime_type.clone(),
                data: base64::engine::general_purpose::STANDARD.encode(bytes),
            })
        }
    }
}

pub(crate) async fn materialize_messages(
    messages: &[Message],
    ctx: &Context,
) -> Result<Vec<Message>, ClientError> {
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let mut materialized = message.clone();
        let mut attachments = Vec::with_capacity(materialized.attachments.len());
        for attachment in &materialized.attachments {
            attachments.push(materialize_attachment(attachment, ctx).await?);
        }
        materialized.attachments = attachments;
        out.push(materialized);
    }
    Ok(out)
}

/// Tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlation id that must be echoed back in [`Role::Tool`].
    pub id: String,
    pub name: String,
    pub args: Value,
    /// Provider-specific continuation data from Gemini thinking models.
    /// Echo this back unchanged on the next turn when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signatures: Option<Vec<String>>,
}

/// Output from a single model call.
#[derive(Debug)]
pub enum ClientOutput {
    /// Structured output payload, or plain text wrapped as `Value::String`.
    Output(Value),
    /// Tool calls requested by the model.
    ToolCalls {
        /// Accompanying text emitted with the tool calls.
        thought: Option<String>,
        calls: Vec<ToolCall>,
    },
}

/// Provider-normalized result from one model call.
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

/// Token counts reported for a single call.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input-side tokens.
    pub input: Option<u32>,
    /// Output-side tokens.
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

/// Errors returned by a client call.
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

/// Supported provider backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    Gemini,
    Ollama,
    OpenAi,
    Anthropic,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Gemini => "gemini",
            Provider::Ollama => "ollama",
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug, Default)]
struct TransportQuery {
    api_key: Option<String>,
    base_url: Option<String>,
}

/// Parsed model URL.
/// Cloud providers use `scheme://[key@]model`.
/// Compatible endpoints may add `?base_url=...&api_key_env=...`.
/// Ollama also accepts the legacy `ollama://host:port/model` form.
#[derive(Debug, Clone)]
pub struct LlmUrl {
    pub provider: Provider,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

impl LlmUrl {
    /// Parses a model URL.
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
            other => {
                return Err(ClientError::InvalidUrl(format!(
                    "unknown provider '{other}'; expected gemini, ollama, openai, anthropic, or claude"
                )));
            }
        };

        let (rest, query) = split_query(rest);
        let transport = parse_transport_query(query)?;

        match provider {
            Provider::Ollama => parse_ollama_url(rest, transport),
            _ => parse_cloud_url(provider, rest, transport, s),
        }
    }
}

fn split_query(rest: &str) -> (&str, Option<&str>) {
    match rest.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (rest, None),
    }
}

fn parse_transport_query(query: Option<&str>) -> Result<TransportQuery, ClientError> {
    let Some(query) = query else {
        return Ok(TransportQuery::default());
    };

    let mut transport = TransportQuery::default();
    for pair in query.split('&').filter(|pair| !pair.is_empty()) {
        let (name, value) = pair.split_once('=').ok_or_else(|| {
            ClientError::InvalidUrl(format!("query parameter '{pair}' must be key=value"))
        })?;
        if value.is_empty() {
            return Err(ClientError::InvalidUrl(format!(
                "query parameter '{name}' must not be empty"
            )));
        }
        match name {
            "api_key_env" => set_query_api_key(&mut transport, value)?,
            "base_url" => set_query_base_url(&mut transport, value)?,
            other => {
                return Err(ClientError::InvalidUrl(format!(
                    "unknown query parameter '{other}'; supported params are base_url and api_key_env"
                )));
            }
        }
    }
    Ok(transport)
}

fn set_query_api_key(transport: &mut TransportQuery, env_name: &str) -> Result<(), ClientError> {
    if transport.api_key.is_some() {
        return Err(ClientError::InvalidUrl(
            "api_key_env must only be provided once".into(),
        ));
    }
    let api_key = std::env::var(env_name).map_err(|_| {
        ClientError::InvalidUrl(format!(
            "environment variable '{env_name}' referenced by api_key_env is not set"
        ))
    })?;
    transport.api_key = Some(api_key);
    Ok(())
}

fn set_query_base_url(transport: &mut TransportQuery, base_url: &str) -> Result<(), ClientError> {
    if transport.base_url.is_some() {
        return Err(ClientError::InvalidUrl(
            "base_url must only be provided once".into(),
        ));
    }
    transport.base_url = Some(normalize_base_url(base_url)?);
    Ok(())
}

fn normalize_base_url(base_url: &str) -> Result<String, ClientError> {
    let url = reqwest::Url::parse(base_url).map_err(|e| {
        ClientError::InvalidUrl(format!("invalid base_url '{base_url}': {e}"))
    })?;
    Ok(url.as_str().trim_end_matches('/').to_string())
}

fn parse_cloud_url(
    provider: Provider,
    rest: &str,
    transport: TransportQuery,
    original: &str,
) -> Result<LlmUrl, ClientError> {
    let (inline_api_key, model) = match rest.split_once('@') {
        Some((key, model)) => (Some(key.to_owned()), model.to_owned()),
        None => (None, rest.to_owned()),
    };
    if model.is_empty() {
        return Err(ClientError::InvalidUrl(format!(
            "missing model name in '{original}'"
        )));
    }
    Ok(LlmUrl {
        provider,
        model,
        api_key: inline_api_key.or(transport.api_key),
        base_url: transport.base_url,
    })
}

fn parse_ollama_url(rest: &str, transport: TransportQuery) -> Result<LlmUrl, ClientError> {
    if rest.is_empty() {
        return Err(ClientError::InvalidUrl(
            "missing model name in ollama URL".into(),
        ));
    }

    let (model, base_url) = match rest.split_once('/') {
        Some((authority, model)) => {
            if model.is_empty() {
                return Err(ClientError::InvalidUrl(
                    "missing model name in ollama URL".into(),
                ));
            }
            let base_url = transport
                .base_url
                .unwrap_or_else(|| format!("http://{authority}"));
            (model.to_owned(), base_url)
        }
        None => {
            let base_url = transport.base_url.ok_or_else(|| {
                ClientError::InvalidUrl(
                    "ollama URL must have format ollama://host:port/model-name or provide ?base_url=..."
                        .into(),
                )
            })?;
            (rest.to_owned(), base_url)
        }
    };

    Ok(LlmUrl {
        provider: Provider::Ollama,
        model,
        api_key: transport.api_key,
        base_url: Some(base_url),
    })
}

pub(super) fn required_api_key(url: &LlmUrl, default_env: &str) -> Result<String, ClientError> {
    url.api_key
        .clone()
        .or_else(|| std::env::var(default_env).ok())
        .ok_or_else(|| ClientError::Llm(format!("{default_env} is not set")))
}

pub(super) fn optional_api_key(url: &LlmUrl, default_env: &str) -> Option<String> {
    url.api_key
        .clone()
        .or_else(|| std::env::var(default_env).ok())
}

pub(super) fn configured_base_url(url: &LlmUrl, default_base_url: &str) -> String {
    url.base_url
        .clone()
        .unwrap_or_else(|| default_base_url.to_string())
}

/// Controls whether the model may call tools.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// Let the provider decide.
    #[default]
    Auto,
    /// Require at least one tool call.
    Required,
    /// Disable tool calls.
    Disabled,
}

/// Per-call client settings.
#[derive(Debug, Clone, Default)]
pub struct ClientOptions {
    /// Optional label for tracing.
    pub name: Option<String>,
    /// Preamble sent before history.
    pub preamble: Option<String>,
    /// Tools available to the model.
    pub tools: Vec<ToolDefinition>,
    /// Enables provider-specific reasoning modes.
    pub thinking: bool,
    /// Tool-call policy.
    pub tool_choice: ToolChoice,
    /// JSON Schema for the user payload.
    pub input_schema: Option<Value>,
    /// JSON Schema for structured output.
    pub output_schema: Option<Value>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Reasoning budget. Ignored unless `thinking` is enabled.
    pub thinking_budget: Option<u32>,
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

    /// Sets the input schema.
    pub fn with_input_schema(mut self, schema: Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    pub(crate) fn effective_preamble(&self) -> Option<String> {
        match (&self.preamble, &self.input_schema) {
            (None, None) => None,
            (Some(preamble), None) => Some(preamble.clone()),
            (None, Some(schema)) => Some(Self::input_schema_hint(schema)),
            (Some(preamble), Some(schema)) => Some(format!(
                "{preamble}\n\n{}",
                Self::input_schema_hint(schema)
            )),
        }
    }

    fn input_schema_hint(schema: &Value) -> String {
        format!(
            "The user message is JSON. Interpret it using this JSON Schema: {schema}"
        )
    }

    pub(crate) fn wants_json_output(&self) -> bool {
        self.input_schema.is_some()
    }

    /// Sets the structured-output schema.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the sampling temperature from an `Option`.
    pub fn with_temperature_opt(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the reasoning budget.
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    /// Sets the reasoning budget from an `Option`.
    pub fn with_thinking_budget_opt(mut self, budget: Option<u32>) -> Self {
        self.thinking_budget = budget;
        self
    }

    /// Builds a provider client for the given model URL.
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

pub(super) fn decode_output_text(text: &str, wants_json_output: bool) -> Result<Value, ClientError> {
    if wants_json_output {
        parse_json_output(text)
    } else {
        Ok(Value::String(text.to_owned()))
    }
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

    /// Wraps this factory with `layer`.
    /// The most recently added layer becomes the outermost wrapper.
    fn layer<L>(self, layer: L) -> L::Factory
    where
        Self: Sized,
        L: ClientFactoryLayer<Self>,
    {
        layer.layer(self)
    }
}

/// Decorates one [`ClientFactory`] with another.
pub trait ClientFactoryLayer<F> {
    type Factory: ClientFactory;

    fn layer(self, inner: F) -> Self::Factory;
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
    use async_trait::async_trait;

    struct DummyFactory;

    struct DummyClient;

    struct MarkerLayer;

    struct MarkedFactory<F> {
        inner: F,
        marked: bool,
    }

    #[async_trait]
    impl Client for DummyClient {
        fn provider(&self) -> Provider {
            Provider::OpenAi
        }

        async fn execute(&self, _messages: &[Message]) -> Result<ClientResponse, ClientError> {
            Ok(ClientResponse::new(
                Provider::OpenAi,
                ClientOutput::Output(serde_json::json!({ "ok": true })),
            ))
        }
    }

    impl ClientFactory for DummyFactory {
        fn create(
            &self,
            _model_url: &str,
            _options: ClientOptions,
        ) -> Result<Box<dyn Client>, ClientError> {
            Ok(Box::new(DummyClient))
        }
    }

    impl<F: ClientFactory> ClientFactory for MarkedFactory<F> {
        fn create(
            &self,
            model_url: &str,
            options: ClientOptions,
        ) -> Result<Box<dyn Client>, ClientError> {
            self.inner.create(model_url, options)
        }
    }

    impl<F: ClientFactory> ClientFactoryLayer<F> for MarkerLayer {
        type Factory = MarkedFactory<F>;

        fn layer(self, inner: F) -> Self::Factory {
            MarkedFactory {
                inner,
                marked: true,
            }
        }
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

    /// `LlmUrl::parse` accepts query-param base URLs for protocol-compatible OpenAI endpoints.
    #[test]
    fn parse_openai_query_base_url() {
        let url = LlmUrl::parse(
            "openai://gpt-4o?base_url=https://openrouter.ai/api/v1/",
        )
        .unwrap();
        assert_eq!(url.provider, Provider::OpenAi);
        assert_eq!(url.model, "gpt-4o");
        assert_eq!(url.base_url.as_deref(), Some("https://openrouter.ai/api/v1"));
    }

    /// `LlmUrl::parse` resolves `api_key_env` query params before the client is built.
    #[test]
    fn parse_query_api_key_env() {
        let expected = std::env::var("PATH").expect("PATH should be set during tests");
        let url = LlmUrl::parse(
            "anthropic://claude-haiku-4-5?api_key_env=PATH",
        )
        .unwrap();
        assert_eq!(url.provider, Provider::Anthropic);
        assert_eq!(url.api_key.as_deref(), Some(expected.as_str()));
    }

    /// Query-param Ollama URLs work without the legacy host/model split.
    #[test]
    fn parse_ollama_query_base_url() {
        let url = LlmUrl::parse(
            "ollama://qwen3:8b?base_url=http://localhost:11434",
        )
        .unwrap();
        assert_eq!(url.provider, Provider::Ollama);
        assert_eq!(url.model, "qwen3:8b");
        assert_eq!(url.base_url.as_deref(), Some("http://localhost:11434"));
    }

    /// `anthropic://` and `claude://` both select the Anthropic provider.
    #[test]
    fn parse_anthropic_aliases() {
        let anthropic = LlmUrl::parse("anthropic://claude-sonnet-4-5").unwrap();
        let claude = LlmUrl::parse("claude://claude-sonnet-4-5").unwrap();
        assert_eq!(anthropic.provider, Provider::Anthropic);
        assert_eq!(claude.provider, Provider::Anthropic);
    }

    /// Provider schemes are authoritative even for custom model names.
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

    /// `LlmUrl::parse` returns an error for an unknown provider scheme.
    #[test]
    fn parse_unknown_scheme_errors() {
        assert!(matches!(
            LlmUrl::parse("unknown://model"),
            Err(ClientError::InvalidUrl(_))
        ));
    }

    /// Missing `api_key_env` variables fail early with a clear URL-configuration error.
    #[test]
    fn parse_missing_api_key_env_errors() {
        assert!(matches!(
            LlmUrl::parse("openai://gpt-4o?api_key_env=__PRAVAH_MISSING_ENV__"),
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

    #[test]
    fn effective_preamble_appends_input_schema_hint() {
        let options = ClientOptions::default()
            .with_preamble("You are helpful.")
            .with_input_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string" }
                },
                "required": ["kind"]
            }));

        let preamble = options
            .effective_preamble()
            .expect("effective preamble should be present");
        assert!(preamble.contains("You are helpful."));
        assert!(preamble.contains("The user message is JSON."));
        assert!(preamble.contains("\"required\":[\"kind\"]"));
    }

    #[test]
    fn wants_json_output_requires_input_schema() {
        assert!(!ClientOptions::default().wants_json_output());
        assert!(ClientOptions::default()
            .with_input_schema(serde_json::json!({ "type": "object" }))
            .wants_json_output());
    }

    #[test]
    fn decode_output_text_returns_plain_text_when_json_mode_disabled() {
        assert_eq!(
            decode_output_text("hello", false).unwrap(),
            Value::String("hello".into())
        );
    }

    #[test]
    fn decode_output_text_parses_json_when_json_mode_enabled() {
        assert_eq!(
            decode_output_text(r#"{"ok":true}"#, true).unwrap(),
            serde_json::json!({ "ok": true })
        );
    }

    /// `ClientFactory::layer` wraps a concrete factory with the supplied decorator.
    #[tokio::test]
    async fn layer_wraps_factory() {
        let factory = DummyFactory.layer(MarkerLayer);
        assert!(factory.marked);

        let client = factory
            .create("openai://test-model", ClientOptions::default())
            .expect("layered factory should create a client");
        let response = client
            .execute(&[Message::user("hi")])
            .await
            .expect("layered client should execute");
        assert!(matches!(response.output, ClientOutput::Output(_)));
    }

    /// File attachments are materialized into inline base64 before dispatch.
    #[tokio::test]
    async fn materialize_messages_reads_file_attachments() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        tokio::fs::write(dir.path().join("shot.png"), b"hello")
            .await
            .expect("attachment file should be written");
        let ctx = crate::Context::new(crate::FlowConf {
            working_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        });

        let messages = vec![Message {
            role: Role::User,
            content: "look".into(),
            attachments: vec![Attachment::File {
                mime_type: "image/png".into(),
                path: "shot.png".into(),
            }],
            usage: None,
        }];

        let materialized = materialize_messages(&messages, &ctx)
            .await
            .expect("file attachment should materialize");

        assert!(matches!(
            materialized[0].attachments.as_slice(),
            [Attachment::Inline { mime_type, data }]
                if mime_type == "image/png" && data == "aGVsbG8="
        ));
    }

    /// Non-image file attachments are materialized without being rejected globally.
    #[tokio::test]
    async fn materialize_messages_reads_non_image_file_attachments() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        tokio::fs::write(dir.path().join("note.pdf"), b"hello")
            .await
            .expect("attachment file should be written");
        let ctx = crate::Context::new(crate::FlowConf {
            working_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        });

        let messages = vec![Message {
            role: Role::User,
            content: "look".into(),
            attachments: vec![Attachment::File {
                mime_type: "application/pdf".into(),
                path: "note.pdf".into(),
            }],
            usage: None,
        }];

        let materialized = materialize_messages(&messages, &ctx)
            .await
            .expect("file attachment should materialize");

        assert!(matches!(
            materialized[0].attachments.as_slice(),
            [Attachment::Inline { mime_type, data }]
                if mime_type == "application/pdf" && data == "aGVsbG8="
        ));
    }
}
