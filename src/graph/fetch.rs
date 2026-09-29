//! Buffered, serializable requests crossing the synchronous execution boundary.
//!
//! Request data is retained faithfully, including credentials supplied by the
//! application. Default diagnostics omit URLs, headers, bodies and error details.

use std::{fmt, sync::Arc};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::value::Value;

mod executor;
mod failure;
pub mod rath;
mod value;

pub use executor::{DynFetchHandler, FetchExecutor};

/// A buffered body; local protocols can pass shared values without a JSON codec.
#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum FetchBody {
    /// Opaque bytes, interpreted by the external executor.
    Bytes(#[schemars(with = "Vec<u8>")] Arc<[u8]>),
    /// Structured data, interpreted by the selected protocol.
    Value(#[schemars(with = "serde_json::Value")] Value),
}

/// One external request. Schemes select execution; they do not imply HTTP transport.
#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchRequest {
    method: String,
    url: String,
    headers: Vec<(String, Vec<u8>)>,
    body: Option<FetchBody>,
}

impl FetchRequest {
    /// Creates a request without headers or a body. The executor validates its protocol.
    pub fn new(method: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            url: url.into(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Appends one header, preserving duplicates, order and uninterpreted bytes.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Replaces the buffered body.
    pub fn body(mut self, body: FetchBody) -> Self {
        self.body = Some(body);
        self
    }

    /// Borrows the supplied method without normalizing it.
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Borrows the full locator, which may contain application-supplied secrets.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Borrows headers in the order supplied, including repeated headers.
    pub fn headers(&self) -> &[(String, Vec<u8>)] {
        &self.headers
    }

    /// Borrows the optional buffered body.
    pub fn body_ref(&self) -> Option<&FetchBody> {
        self.body.as_ref()
    }
}

/// A delivered response. HTTP error statuses remain responses, not execution failures.
#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchResponse {
    status: u16,
    headers: Vec<(String, Vec<u8>)>,
    body: Option<FetchBody>,
}

impl FetchResponse {
    /// Creates a response without headers or a body.
    pub fn new(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: None,
        }
    }

    /// Appends a header without coalescing repeated fields.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Replaces the buffered body.
    pub fn body(mut self, body: FetchBody) -> Self {
        self.body = Some(body);
        self
    }

    /// Returns the status supplied by the executor.
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Borrows headers in their original order.
    pub fn headers(&self) -> &[(String, Vec<u8>)] {
        &self.headers
    }

    /// Borrows the response body.
    pub fn body_ref(&self) -> Option<&FetchBody> {
        self.body.as_ref()
    }
}

/// Portable execution failure, not an arbitrary Rust source chain.
/// Applications explicitly inspect diagnostics; formatting does not reveal them.
#[derive(Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchError {
    code: String,
    message: String,
    #[schemars(with = "Option<serde_json::Value>")]
    details: Option<Value>,
}

impl FetchError {
    /// Creates a portable failure. Code and message are not printed automatically.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    /// Attaches explicitly inspectable, serializable diagnostic data.
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    /// Borrows the application or protocol failure classification.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Borrows the original portable diagnostic message, which can contain private data.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Borrows retained diagnostics, which can include response bodies.
    pub fn details(&self) -> Option<&Value> {
        self.details.as_ref()
    }
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("external execution failed")
    }
}

impl std::error::Error for FetchError {}

/// Pending request identity and immutable contents, emitted by one execution.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fetch {
    id: Uuid,
    request: Arc<FetchRequest>,
}

impl Fetch {
    /// Creates an executor input from its durable identity and request.
    pub fn new(id: Uuid, request: Arc<FetchRequest>) -> Self {
        Self { id, request }
    }

    /// Returns the identity to include when delivering an outcome.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Borrows the exact buffered request.
    pub fn request(&self) -> &FetchRequest {
        &self.request
    }
}

impl fmt::Debug for FetchBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bytes(_) => "Bytes(..)",
            Self::Value(_) => "Value(..)",
        })
    }
}

impl fmt::Debug for FetchRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FetchRequest").finish_non_exhaustive()
    }
}

impl fmt::Debug for FetchResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FetchResponse")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FetchError").finish_non_exhaustive()
    }
}

impl fmt::Debug for Fetch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fetch")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "tests/fetch.rs"]
mod tests;
