//! Operation-local dispatch context for caller-owned history maintenance.

use super::*;
use crate::graph::{FetchBody, fetch::rath::RathRequest};

/// Reconstructs only the upcoming request context, never retaining a client or configuration copy.
pub(crate) fn dispatch_request(
    payload: &Value,
    checkpoint: &Value,
) -> Result<(String, RathRequest, Vec<Message>, bool), GraphError> {
    let payload = AgentPayloadView::read(payload)?;
    let checkpoint = EdgeAgentCheckpoint::from_value(checkpoint)?;
    let EdgeAgentPhase::Dispatch { conclusion } = checkpoint.phase else {
        return Err(GraphError::Invalid(
            "agent is not at a dispatch boundary".into(),
        ));
    };
    let value = super::request::generation(&payload, &checkpoint, conclusion.is_some())?;
    let request =
        crate::graph::FetchRequest::new("POST", "rath://generate").body(FetchBody::Value(value));
    let mut guidance = Vec::new();
    if let Some(text) = checkpoint.guidance {
        guidance.push(Message::new(
            Role::System,
            format!("<pravah_agent_intervention>\n{text}\n</pravah_agent_intervention>"),
        ));
    }
    Ok((
        checkpoint.session_id,
        RathRequest::from_fetch_request(&request)?,
        guidance,
        matches!(conclusion, Some(ConclusionCause::TurnBudget)),
    ))
}
