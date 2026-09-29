//! Versioned representation of one ordinary Rath generation operation.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de::IgnoredAny};

use crate::clients::{ClientOptions, ClientResponse, Message};
use crate::graph::{FetchBody, FetchRequest, FetchResponse, GraphError, from_value, to_value};

mod options;
mod response;
mod wire;

/// Current version of Rath request and response payloads.
pub const RATH_FETCH_VERSION: u32 = 1;

/// Frozen logical model inputs, not a client, credential resolver or HTTP request.
#[derive(Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RathRequest {
    version: u32,
    model: String,
    #[serde(serialize_with = "options::serialize")]
    options: ClientOptions,
    messages: Vec<Message>,
}

impl RathRequest {
    /// Encodes agent defaults while sharing authored schemas and resolved provider configuration.
    pub(crate) fn agent_value(
        model: super::Value,
        name: &str,
        preamble: String,
        tools: Vec<super::Value>,
        schema: super::Value,
        provider_config: super::Value,
        max_output_tokens: super::Value,
    ) -> Result<super::Value, GraphError> {
        let options = options::agent_value(
            name,
            preamble,
            tools,
            schema,
            provider_config,
            max_output_tokens,
        )?;
        super::value::object([
            ("version", RATH_FETCH_VERSION.into()),
            ("model", model),
            ("options", options),
            ("messages", super::Value::array([])),
        ])
    }

    /// Preserves validated construction inputs when preparation replaces only the message list.
    /// Installation and external execution still validate the complete resulting request.
    pub(crate) fn replace_messages(
        source: &super::Value,
        messages: super::Value,
    ) -> Result<FetchRequest, GraphError> {
        let field = |key| {
            source
                .get(key)
                .cloned()
                .ok_or_else(|| GraphError::FetchValidation("missing Rath request field".into()))
        };
        let version = from_value(field("version")?)
            .map_err(|_| GraphError::FetchValidation("invalid Rath request version".into()))?;
        validate_version(version, "Rath request")?;
        let body = super::value::object([
            ("version", version.into()),
            ("model", field("model")?),
            ("options", field("options")?),
            ("messages", messages),
        ])?;
        Ok(FetchRequest::new("POST", "rath://generate").body(FetchBody::Value(body)))
    }

    pub(crate) fn into_parts(self) -> (String, ClientOptions, Vec<Message>) {
        (self.model, self.options, self.messages)
    }
    /// Captures the caller's construction inputs without applying provider defaults.
    pub fn new(model: impl Into<String>, options: ClientOptions, messages: Vec<Message>) -> Self {
        Self {
            version: RATH_FETCH_VERSION,
            model: model.into(),
            options,
            messages,
        }
    }

    /// Borrows the original model locator, including any caller-supplied parameters.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Borrows the exact construction options, before Rath applies URL precedence.
    pub fn options(&self) -> &ClientOptions {
        &self.options
    }

    /// Borrows the frozen messages, including attachments and continuation signatures.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Encodes this operation directly into a structured Fetch body.
    pub fn into_fetch_request(self) -> Result<FetchRequest, GraphError> {
        self.validate()?;
        Ok(FetchRequest::new("POST", "rath://generate")
            .body(FetchBody::Value(encode(self, "Rath request")?)))
    }

    /// Decodes a generation request, rejecting unsupported envelopes and versions.
    pub fn from_fetch_request(request: &FetchRequest) -> Result<Self, GraphError> {
        let value = request_body(request)?;
        let request: Self = from_value(value.clone())
            .map_err(|_| GraphError::FetchValidation("invalid Rath request payload".into()))?;
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), GraphError> {
        validate_version(self.version, "Rath request")
    }
}

/// Provider-normalized output, preserving Rath metadata without reinterpreting it.
#[derive(Serialize)]
#[serde(deny_unknown_fields)]
pub struct RathResponse {
    version: u32,
    #[serde(serialize_with = "response::serialize")]
    response: ClientResponse,
}

impl RathResponse {
    /// Wraps one normalized Rath response without discarding provider fields.
    pub fn new(response: ClientResponse) -> Self {
        Self {
            version: RATH_FETCH_VERSION,
            response,
        }
    }

    /// Borrows the normalized response; generated content is explicitly accessible.
    pub fn response(&self) -> &ClientResponse {
        &self.response
    }

    /// Returns the normalized response after consuming its versioned envelope.
    pub fn into_response(self) -> ClientResponse {
        self.response
    }

    /// Encodes a successful logical generation operation, not an underlying HTTP status.
    pub fn into_fetch_response(self) -> Result<FetchResponse, GraphError> {
        validate_version(self.version, "Rath response")?;
        Ok(FetchResponse::new(200).body(FetchBody::Value(encode(self, "Rath response")?)))
    }

    /// Validates the logical response envelope and protocol version before use.
    pub fn from_fetch_response(response: &FetchResponse) -> Result<Self, GraphError> {
        let value = response_body(response)?;
        let response: Self = from_value(value.clone())
            .map_err(|_| GraphError::FetchValidation("invalid Rath response payload".into()))?;
        validate_version(response.version, "Rath response")?;
        Ok(response)
    }
}

impl<'de> Deserialize<'de> for RathRequest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = wire::Request::<options::Options<serde_json::Value>>::deserialize(deserializer)?;
        Ok(Self {
            version: wire.version,
            model: wire.model,
            options: wire
                .options
                .into_options()
                .map_err(serde::de::Error::custom)?,
            messages: wire.messages,
        })
    }
}

impl<'de> Deserialize<'de> for RathResponse {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire =
            wire::Response::<response::Response<serde_json::Value>>::deserialize(deserializer)?;
        Ok(Self {
            version: wire.version,
            response: wire.response.into_response(),
        })
    }
}

/// Checks the complete protocol without reconstructing option JSON trees.
/// Rath-owned messages and typed controls still use their ordinary decoders.
pub(crate) fn validate_request(request: &FetchRequest) -> Result<(), GraphError> {
    let wire = wire::Request::<options::Options<IgnoredAny>>::deserialize(request_body(request)?)
        .map_err(|_| GraphError::FetchValidation("invalid Rath request payload".into()))?;
    wire.options
        .controls()
        .map_err(|_| GraphError::FetchValidation("invalid Rath request payload".into()))?;
    validate_version(wire.version, "Rath request")
}

/// Checks delivery and restore without constructing discarded normalized output/metadata values.
/// Serde's tagged output and Rath-owned tool calls may still allocate temporary decoding data.
pub(crate) fn validate_response(response: &FetchResponse) -> Result<(), GraphError> {
    let wire =
        wire::Response::<response::Response<IgnoredAny>>::deserialize(response_body(response)?)
            .map_err(|_| GraphError::FetchValidation("invalid Rath response payload".into()))?;
    validate_version(wire.version, "Rath response")
}

/// Applies the same method, destination, header and body checks to both decoding modes.
fn request_body(request: &FetchRequest) -> Result<&super::Value, GraphError> {
    if request.method() != "POST"
        || request.url() != "rath://generate"
        || !request.headers().is_empty()
    {
        return Err(GraphError::FetchValidation(
            "invalid Rath request envelope".into(),
        ));
    }
    structured_body(request.body_ref())
}

fn response_body(response: &FetchResponse) -> Result<&super::Value, GraphError> {
    if response.status() != 200 || !response.headers().is_empty() {
        return Err(GraphError::FetchValidation(
            "invalid Rath response envelope".into(),
        ));
    }
    structured_body(response.body_ref())
}

fn structured_body(body: Option<&FetchBody>) -> Result<&super::Value, GraphError> {
    match body {
        Some(FetchBody::Value(value)) => Ok(value),
        _ => Err(GraphError::FetchValidation(
            "expected a structured protocol body".into(),
        )),
    }
}

fn validate_version(got: u32, format: &'static str) -> Result<(), GraphError> {
    if got == RATH_FETCH_VERSION {
        return Ok(());
    }
    Err(GraphError::UnsupportedVersion {
        format,
        got,
        expected: RATH_FETCH_VERSION,
    })
}

fn encode(value: impl Serialize, target: &str) -> Result<super::Value, GraphError> {
    to_value(value).map_err(|err| GraphError::ValueConversion {
        target: target.into(),
        reason: err.to_string(),
    })
}

impl fmt::Debug for RathRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RathRequest")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for RathResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RathResponse")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "tests/rath.rs"]
mod tests;
