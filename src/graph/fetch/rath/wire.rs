//! One closed envelope definition for both owned decoding and allocation-light validation.

use serde::Deserialize;

use crate::clients::Message;

/// Operation-local decoding; the caller selects how option JSON fields are visited.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request<O> {
    pub version: u32,
    pub model: String,
    pub options: O,
    pub messages: Vec<Message>,
}

/// The version check and normalized output always use the same enclosing representation.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Response<R> {
    pub version: u32,
    pub response: R,
}
