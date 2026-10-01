//! Operation-local preparation of messages, sharing unchanged encoded values between requests.

use crate::Context;
use crate::clients::{Attachment, Message, materialize_owned_messages};
use crate::graph::{FetchRequest, GraphError, Value};

use super::effect_values::body_field;
use super::effects::{PreparationRequest, encode};

/// Validates the whole replacement before constructing its provider-facing message sequence.
pub(super) async fn messages(
    preparation: &PreparationRequest,
    request: &FetchRequest,
    guidance: Vec<Message>,
    context: &Context,
) -> Result<Value, GraphError> {
    let encoded = body_field(request.body_ref(), "request")?
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| messages.len() == preparation.request.messages().len())
        .ok_or_else(|| GraphError::FetchValidation("invalid preparation messages".into()))?;
    let mut messages = Vec::with_capacity(encoded.len() + guidance.len());
    for (message, encoded) in preparation.request.messages().iter().zip(encoded) {
        messages.push(materialize(message, encoded, context).await?);
    }
    for message in guidance {
        messages.push(encode(message)?);
    }
    Ok(Value::array(messages))
}

/// Shares already buffered messages; files and unsupported future attachments follow normal checks.
async fn materialize(
    message: &Message,
    encoded: &Value,
    ctx: &Context,
) -> Result<Value, GraphError> {
    if message.attachments.iter().all(|attachment| {
        matches!(
            attachment,
            Attachment::Inline { .. } | Attachment::Url { .. }
        )
    }) {
        return Ok(encoded.clone());
    }
    let messages = materialize_owned_messages(vec![message.clone()], ctx)
        .await
        .map_err(|error| {
            GraphError::FetchValidation(format!("message materialization failed: {error}"))
        })?;
    let message = messages
        .into_iter()
        .next()
        .ok_or_else(|| GraphError::Invalid("materialized message disappeared".into()))?;
    encode(message)
}
