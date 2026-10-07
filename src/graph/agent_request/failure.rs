//! Explicit conversion of local diagnostics into portable, serializable failure data.

use crate::clients::{ClientError, ErrorKind};
use crate::graph::{GraphError, Value, ValueError, to_value};

use super::AgentError;

impl AgentError {
    /// Copies explicitly selected diagnostics for durable delivery. This is not the original source.
    /// Retained body bytes may contain private content and are never printed by default.
    pub fn from_execution_error(error: &GraphError) -> Result<Self, ValueError> {
        match error {
            GraphError::AgentClient { operation, source } => {
                let details = Value::object([
                    ("operation", Value::from(operation.to_string())),
                    ("rath", rath_details(source)?),
                ])?;
                Ok(Self::new("rath", source.message()).with_details(details))
            }
            GraphError::AgentOutputLimit { agent, provider } => Ok(Self::new(
                "output_limit",
                "generation output limit reached",
            )
            .with_details(Value::object([
                ("agent", Value::from(agent.clone())),
                ("provider", to_value(provider)?),
            ])?)),
            _ => Ok(Self::new("execution", error.to_string())),
        }
    }
}

/// Preserves every public Rath diagnostic field and its normalized cause chain.
fn rath_details(error: &ClientError) -> Result<Value, ValueError> {
    let body = match error.response_body() {
        Some(body) => Value::object([
            ("complete", Value::from(body.is_complete())),
            ("bytes", to_value(body.bytes())?),
        ])?,
        None => Value::NULL,
    };
    let cause = error
        .source()
        .map(rath_details)
        .transpose()?
        .unwrap_or(Value::NULL);
    Value::object([
        ("kind", Value::from(kind_name(error.kind()))),
        ("provider", to_value(error.provider())?),
        ("operation", to_value(error.operation())?),
        ("message", Value::from(error.message())),
        ("http_status", to_value(error.http_status())?),
        ("provider_code", to_value(error.provider_code())?),
        ("request_id", to_value(error.request_id())?),
        ("retry_after", to_value(error.retry_after())?),
        ("response_body", body),
        ("cause", cause),
    ])
}

fn kind_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Validation => "validation",
        ErrorKind::InvalidUrl => "invalid_url",
        ErrorKind::UnsupportedCapability => "unsupported_capability",
        ErrorKind::Transport => "transport",
        ErrorKind::Timeout => "timeout",
        ErrorKind::Http => "http",
        ErrorKind::Provider => "provider",
        ErrorKind::Serialize => "serialize",
        ErrorKind::Deserialize => "deserialize",
        ErrorKind::InvalidResponse => "invalid_response",
        ErrorKind::OutputLimitReached => "output_limit_reached",
        ErrorKind::TokenCounting => "token_counting",
        _ => "other",
    }
}
