//! Serde support for Rath's non-Serde option types, without normalizing inputs.

use serde::Deserialize;
use serde_json::Value;

use crate::clients::{
    CacheControl, ClientOptions, ResponseFormat, ThinkingLevel, ToolChoice, ToolDefinition,
};

/// Keeps the agent option shape alongside Rath's ordinary option codec, without recoding schemas.
pub(crate) fn agent_value(
    name: &str,
    preamble: String,
    tools: Vec<crate::graph::Value>,
    schema: crate::graph::Value,
    provider_config: crate::graph::Value,
    max_output_tokens: crate::graph::Value,
) -> Result<crate::graph::Value, crate::graph::GraphError> {
    use super::object;
    use crate::graph::Value as VmValue;
    let tool_choice = if tools.is_empty() { "disabled" } else { "auto" };
    object([
        ("max_output_tokens", max_output_tokens),
        ("name", name.into()),
        ("preamble", preamble.into()),
        ("tools", VmValue::array(tools)),
        ("thinking", VmValue::NULL),
        ("tool_choice", tool_choice.into()),
        ("input_schema", VmValue::NULL),
        (
            "response_format",
            object([("kind", "json_schema".into()), ("schema", schema)])?,
        ),
        ("temperature", VmValue::NULL),
        ("provider_config", provider_config),
        ("cache", VmValue::NULL),
    ])
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Options<J> {
    max_output_tokens: Option<u32>,
    name: Option<String>,
    preamble: Option<String>,
    tools: Vec<(String, String, J)>,
    thinking: Option<String>,
    tool_choice: String,
    input_schema: Option<J>,
    response_format: Format<J>,
    temperature: Option<f32>,
    provider_config: Option<J>,
    cache: Option<String>,
}

#[derive(Deserialize)]
#[serde(
    tag = "kind",
    content = "schema",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Format<J> {
    Text,
    Json,
    JsonSchema(J),
}

/// Decodes construction options without applying provider defaults or URL overrides.
pub(crate) fn deserialize<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ClientOptions, D::Error> {
    Options::<Value>::deserialize(deserializer)?
        .into_options()
        .map_err(serde::de::Error::custom)
}

impl<J> Options<J> {
    /// Shares control decoding with validation; JSON fields never influence these wire enums.
    pub(crate) fn controls(
        &self,
    ) -> Result<(Option<ThinkingLevel>, ToolChoice, Option<CacheControl>), &'static str> {
        let thinking = self
            .thinking
            .as_deref()
            .map(|value| value.parse().map_err(|()| "unsupported thinking level"))
            .transpose()?;
        let tool_choice = match self.tool_choice.as_str() {
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            "disabled" => ToolChoice::Disabled,
            _ => return Err("unsupported tool choice"),
        };
        let cache = self
            .cache
            .as_deref()
            .map(|value| match value {
                "5m" => Ok(CacheControl::Ephemeral5m),
                "1h" => Ok(CacheControl::Ephemeral1h),
                _ => Err("unsupported cache policy"),
            })
            .transpose()?;
        Ok((thinking, tool_choice, cache))
    }
}

impl Options<Value> {
    /// Reconstructs only supplied options; Rath retains validation and URL precedence ownership.
    pub(crate) fn into_options(self) -> Result<ClientOptions, &'static str> {
        let (thinking, tool_choice, cache) = self.controls()?;
        let mut options = ClientOptions::default();
        options.max_output_tokens = self.max_output_tokens;
        options.name = self.name;
        options.preamble = self.preamble;
        options.tools = self
            .tools
            .into_iter()
            .map(|(name, description, parameters)| {
                ToolDefinition::new(name, description, parameters)
            })
            .collect();
        options.thinking = thinking;
        options.tool_choice = tool_choice;
        options.input_schema = self.input_schema;
        options.response_format = match self.response_format {
            Format::Text => ResponseFormat::Text,
            Format::Json => ResponseFormat::Json,
            Format::JsonSchema(schema) => ResponseFormat::JsonSchema { schema },
        };
        options.temperature = self.temperature;
        options.provider_config = self.provider_config;
        options.cache = cache;
        Ok(options)
    }
}
