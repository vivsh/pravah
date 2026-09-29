//! Rath client handles, provider registration, messages and structured diagnostics.

mod attachments;
mod guidance;
#[cfg(test)]
mod tests;

pub use rath::core::{ErrorBody, ErrorKind, ModelUrl};
pub use rath::embeddings::{EmbedRequest, EmbedResponse, EmbedTaskType, EmbeddingClient};
pub use rath::llm::{
    Attachment, CacheControl, LlmBackend, LlmClient as Client, LlmOptions as ClientOptions,
    LlmOutput as ClientOutput, LlmResponse as ClientResponse, Message, Provider,
    RathError as ClientError, ResponseFormat, Role, ThinkingLevel, TokenUsage, ToolCall,
    ToolChoice, ToolDefinition,
};
pub use rath::registry::{BuiltinProviderFactory, ProviderFactory, ProviderRegistry};

pub(crate) use attachments::{materialize_messages, materialize_owned_messages};
pub(crate) use guidance::{conclusion_message, wrap_system_reminder};
