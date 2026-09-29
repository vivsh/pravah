//! Operation-local preparation of messages, sharing unchanged encoded values between requests.

use crate::Context;
use crate::clients::{Attachment, Message, materialize_owned_messages};
use crate::graph::{FetchRequest, GraphError, Value};
use crate::history::{
    CompactionResult, ValidatedCompactionResult, prepare_compaction, summary_uuid,
};

use super::effect_values::body_field;
use super::effects::{PreparationRequest, encode};

/// Validates the whole replacement before constructing its provider-facing message sequence.
pub(super) async fn messages(
    preparation: &PreparationRequest,
    request: &FetchRequest,
    decision: CompactionResult,
    guidance: Vec<Message>,
    context: &Context,
) -> Result<Value, GraphError> {
    let replacement = replacement(preparation, decision)?;
    let encoded = body_field(request.body_ref(), "entries")?
        .as_array()
        .filter(|entries| entries.len() == preparation.entries.len())
        .ok_or_else(|| GraphError::FetchValidation("invalid preparation entries".into()))?;
    let mut messages = Vec::new();
    if let Some(summary) = replacement.summary_message() {
        messages.push(encode(summary)?);
    }
    let retained = preparation
        .entries
        .iter()
        .zip(encoded)
        .filter(|(entry, _)| !entry.evicted && entry.session_id == preparation.session)
        .skip(replacement.removed_count());
    for (entry, encoded) in retained {
        let encoded = encoded
            .get("message")
            .ok_or_else(|| GraphError::FetchValidation("missing preparation message".into()))?;
        messages.push(materialize(&entry.message, encoded, context).await?);
    }
    for message in guidance {
        messages.push(encode(message)?);
    }
    Ok(Value::array(messages))
}

/// Uses the existing prefix/group validator and exactly the same deterministic summary identity.
fn replacement(
    preparation: &PreparationRequest,
    decision: CompactionResult,
) -> Result<ValidatedCompactionResult, GraphError> {
    let current = preparation
        .entries
        .iter()
        .filter(|entry| !entry.evicted && entry.session_id == preparation.session)
        .collect::<Vec<_>>();
    let position = preparation
        .entries
        .iter()
        .map(|entry| entry.position.saturating_add(1))
        .max()
        .unwrap_or(0);
    let id = summary_uuid(preparation.execution, &preparation.session, position);
    prepare_compaction(&preparation.session, &current, decision, Some(id)).map_err(|reason| {
        GraphError::HistoryCompactionValidation {
            session_id: preparation.session.clone(),
            reason,
        }
    })
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
