//! Serde support for Rath's non-Serde option types, without normalizing inputs.

use serde::{Deserialize, Serialize, Serializer, ser::SerializeSeq};
use serde_json::Value;

use crate::clients::{
    CacheControl, ClientOptions, ResponseFormat, ThinkingLevel, ToolChoice, ToolDefinition,
};

/// Keeps the agent option shape alongside Rath's ordinary option codec, without recoding schemas.
pub(super) fn agent_value(
    name: &str,
    preamble: String,
    tools: Vec<crate::graph::Value>,
    schema: crate::graph::Value,
    provider_config: crate::graph::Value,
    max_output_tokens: crate::graph::Value,
) -> Result<crate::graph::Value, crate::graph::GraphError> {
    use super::super::value::object;
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

#[derive(Serialize)]
struct OptionsRef<'a> {
    max_output_tokens: Option<u32>,
    name: &'a Option<String>,
    preamble: &'a Option<String>,
    tools: ToolsRef<'a>,
    thinking: Option<&'static str>,
    tool_choice: &'static str,
    input_schema: &'a Option<Value>,
    response_format: FormatRef<'a>,
    temperature: Option<f32>,
    provider_config: &'a Option<Value>,
    cache: Option<&'static str>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Options<J> {
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

#[derive(Serialize)]
#[serde(tag = "kind", content = "schema", rename_all = "snake_case")]
enum FormatRef<'a> {
    Text,
    Json,
    JsonSchema(&'a Value),
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

struct ToolsRef<'a>(&'a [ToolDefinition]);

impl Serialize for ToolsRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for tool in self.0 {
            sequence.serialize_element(&(&tool.name, &tool.description, &tool.parameters))?;
        }
        sequence.end()
    }
}

/// Serializes borrowed fields directly; option encoding makes no schema or message copies.
pub(super) fn serialize<S: Serializer>(
    options: &ClientOptions,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let thinking = options
        .thinking
        .as_ref()
        .map(thinking_name)
        .transpose()
        .map_err(serde::ser::Error::custom)?;
    let cache = options
        .cache
        .as_ref()
        .map(cache_name)
        .transpose()
        .map_err(serde::ser::Error::custom)?;
    let response_format = match &options.response_format {
        ResponseFormat::Text => FormatRef::Text,
        ResponseFormat::Json => FormatRef::Json,
        ResponseFormat::JsonSchema { schema } => FormatRef::JsonSchema(schema),
    };
    OptionsRef {
        max_output_tokens: options.max_output_tokens,
        name: &options.name,
        preamble: &options.preamble,
        tools: ToolsRef(&options.tools),
        thinking,
        tool_choice: match options.tool_choice {
            ToolChoice::Auto => "auto",
            ToolChoice::Required => "required",
            ToolChoice::Disabled => "disabled",
        },
        input_schema: &options.input_schema,
        response_format,
        temperature: options.temperature,
        provider_config: &options.provider_config,
        cache,
    }
    .serialize(serializer)
}

impl<J> Options<J> {
    /// Shares control decoding with validation; JSON fields never influence these wire enums.
    pub(super) fn controls(
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
    pub(super) fn into_options(self) -> Result<ClientOptions, &'static str> {
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

fn thinking_name(value: &ThinkingLevel) -> Result<&'static str, &'static str> {
    match value {
        ThinkingLevel::Off => Ok("off"),
        ThinkingLevel::Low => Ok("low"),
        ThinkingLevel::Medium => Ok("medium"),
        ThinkingLevel::High => Ok("high"),
        ThinkingLevel::XHigh => Ok("xhigh"),
        _ => Err("unsupported thinking level"),
    }
}

fn cache_name(value: &CacheControl) -> Result<&'static str, &'static str> {
    match value {
        CacheControl::Ephemeral5m => Ok("5m"),
        CacheControl::Ephemeral1h => Ok("1h"),
        _ => Err("unsupported cache policy"),
    }
}
