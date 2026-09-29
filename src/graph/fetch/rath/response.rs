//! Lossless serialization of normalized Rath output and diagnostics metadata.

use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use crate::clients::{ClientOutput, ClientResponse, Provider, TokenUsage, ToolCall};

#[derive(Serialize)]
struct ResponseRef<'a> {
    output: OutputRef<'a>,
    usage: Option<TokenUsage>,
    provider: &'a Provider,
    provider_model: &'a Option<String>,
    raw_metadata: &'a Option<Value>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OutputRef<'a> {
    Output {
        value: &'a Value,
    },
    ToolCalls {
        text: &'a Option<String>,
        calls: &'a [ToolCall],
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Response<J> {
    output: Output<J>,
    usage: Option<TokenUsage>,
    provider: Provider,
    provider_model: Option<String>,
    raw_metadata: Option<J>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Output<J> {
    Output {
        value: J,
    },
    ToolCalls {
        text: Option<String>,
        calls: Vec<ToolCall>,
    },
}

/// Serializes the original response by borrowing every non-scalar field.
pub(super) fn serialize<S: Serializer>(
    response: &ClientResponse,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let output = match &response.output {
        ClientOutput::Output(value) => OutputRef::Output { value },
        ClientOutput::ToolCalls { text, calls } => OutputRef::ToolCalls { text, calls },
    };
    ResponseRef {
        output,
        usage: response.usage,
        provider: &response.provider,
        provider_model: &response.provider_model,
        raw_metadata: &response.raw_metadata,
    }
    .serialize(serializer)
}

impl Response<Value> {
    /// Restores every normalized field without converting tool calls to plain text.
    pub(super) fn into_response(self) -> ClientResponse {
        let output = match self.output {
            Output::Output { value } => ClientOutput::Output(value),
            Output::ToolCalls { text, calls } => ClientOutput::ToolCalls { text, calls },
        };
        ClientResponse::new(self.provider, output)
            .with_usage(self.usage)
            .with_provider_model(self.provider_model)
            .with_raw_metadata(self.raw_metadata)
    }
}
