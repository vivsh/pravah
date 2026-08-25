use base64::Engine;

use crate::context::Context;

pub use rath::core::ModelUrl;
pub use rath::embeddings::{EmbedRequest, EmbedResponse, EmbedTaskType, EmbeddingClient};
pub use rath::llm::{
    Attachment, CacheControl, DefaultLlmClientFactory as DefaultClientFactory, LlmClient as Client,
    LlmClientFactory as ClientFactory, LlmClientFactoryLayer as ClientFactoryLayer,
    LlmOptions as ClientOptions, LlmOutput as ClientOutput, LlmResponse as ClientResponse, Message,
    Provider, RathError as ClientError, ResponseFormat, Role, ThinkingLevel, TokenUsage, ToolCall,
    ToolChoice, ToolDefinition, schema,
};

async fn materialize_attachment(
    attachment: &Attachment,
    ctx: &Context,
) -> Result<Attachment, ClientError> {
    match attachment {
        Attachment::Inline { mime_type, data } => Ok(Attachment::Inline {
            mime_type: mime_type.clone(),
            data: data.clone(),
        }),
        Attachment::Url { mime_type, url } => Ok(Attachment::Url {
            mime_type: mime_type.clone(),
            url: url.clone(),
        }),
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

#[cfg(test)]
pub(crate) fn extract_exit_tool_call(calls: &[ToolCall], name: &str) -> Option<serde_json::Value> {
    calls
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.args.clone())
}
