use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use super::super::tools::ToolDefinition;
use super::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmUrl, Message, Provider,
    Role, TokenUsage, ToolCall, ToolChoice, parse_json_output, validate_tools,
};

struct AnthropicClient {
    http: HttpClient,
    api_key: String,
    model: String,
    options: ClientOptions,
}

pub fn new_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    let api_key = url
        .api_key
        .clone()
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        .ok_or_else(|| ClientError::Llm("ANTHROPIC_API_KEY is not set".into()))?;
    Ok(Box::new(AnthropicClient {
        http: HttpClient::new(),
        api_key,
        model: url.model.clone(),
        options,
    }))
}

#[async_trait]
impl Client for AnthropicClient {
    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        validate_history(messages)?;
        validate_tools(Provider::Anthropic, &self.options.tools)?;

        if self.options.thinking {
            return Err(ClientError::UnsupportedCapability {
                provider: Provider::Anthropic,
                capability: "thinking is not exposed by the Anthropic adapter yet".into(),
            });
        }

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);

        let response: Value = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .error_for_status()
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .json()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?;

        map_response(response, tools_enabled)
    }
}

fn validate_history(messages: &[Message]) -> Result<(), ClientError> {
    if messages.is_empty() {
        return Err(ClientError::Validation("messages must not be empty".into()));
    }
    if matches!(
        messages.last().map(|m| &m.role),
        Some(Role::AssistantToolCalls { .. })
    ) {
        return Err(ClientError::Validation(
            "history ends with assistant tool calls without tool results".into(),
        ));
    }
    Ok(())
}

fn build_payload(
    model: &str,
    options: &ClientOptions,
    messages: &[Message],
    tools_enabled: bool,
) -> Value {
    let mut payload = json!({
        "model": model,
        "max_tokens": 4096,
        "messages": build_messages(messages),
    });

    let mut system = Vec::new();
    if let Some(preamble) = &options.preamble {
        system.push(preamble.clone());
    }
    for msg in messages {
        if matches!(msg.role, Role::System) {
            system.push(msg.content.clone());
        }
    }
    if !system.is_empty() {
        payload["system"] = Value::String(system.join("\n\n"));
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        if options.tool_choice == ToolChoice::Required {
            payload["tool_choice"] = json!({ "type": "any" });
        }
    } else {
        let schema_hint = options
            .output_schema
            .as_ref()
            .map(|schema| format!("\n\nReturn only valid JSON matching this JSON Schema: {schema}"))
            .unwrap_or_else(|| "\n\nReturn only valid JSON.".to_string());
        payload["system"] = Value::String(format!(
            "{}{}",
            payload["system"].as_str().unwrap_or(""),
            schema_hint
        ));
    }

    payload
}

fn build_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        match &msg.role {
            Role::System => {}
            Role::User => out.push(json!({ "role": "user", "content": msg.content })),
            Role::Assistant => out.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                let mut content = Vec::new();
                if !msg.content.is_empty() {
                    content.push(json!({ "type": "text", "text": msg.content }));
                }
                for call in calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.args,
                    }));
                }
                out.push(json!({ "role": "assistant", "content": content }));
            }
            Role::Tool { call_id } => out.push(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": msg.content,
                }]
            })),
        }
    }
    out
}

fn build_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect()
}

fn map_response(response: Value, tools_enabled: bool) -> Result<ClientResponse, ClientError> {
    let usage = response.get("usage").map(usage_from_value);
    let provider_model = response
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let metadata = Some(json!({
        "id": response.get("id").cloned().unwrap_or(Value::Null),
        "stop_reason": response.get("stop_reason").cloned().unwrap_or(Value::Null),
    }));

    let (text, calls) = collect_content(&response);
    if !calls.is_empty() {
        return Ok(ClientResponse::new(
            Provider::Anthropic,
            ClientOutput::ToolCalls {
                thought: text,
                calls,
            },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(metadata));
    }
    if tools_enabled {
        return Err(ClientError::MissingToolCalls(text));
    }
    let text = text.ok_or(ClientError::EmptyResponse)?;
    Ok(ClientResponse::new(
        Provider::Anthropic,
        ClientOutput::Output(parse_json_output(&text)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

fn collect_content(response: &Value) -> (Option<String>, Vec<ToolCall>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    for part in response
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            Some("tool_use") => {
                if let (Some(id), Some(name)) = (
                    part.get("id").and_then(Value::as_str),
                    part.get("name").and_then(Value::as_str),
                ) {
                    calls.push(ToolCall {
                        id: id.to_string(),
                        name: name.to_string(),
                        args: part.get("input").cloned().unwrap_or_else(|| json!({})),
                        thought_signatures: None,
                    });
                }
            }
            _ => {}
        }
    }
    ((!text.is_empty()).then_some(text), calls)
}

fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: value
            .get("input_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        output: value
            .get("output_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn messages_encode_tool_exchange() {
        let msgs = build_messages(&[
            Message::user("hi"),
            Message {
                role: Role::AssistantToolCalls {
                    calls: vec![ToolCall {
                        id: "toolu_1".into(),
                        name: "lookup".into(),
                        args: json!({"q":"x"}),
                        thought_signatures: None,
                    }],
                },
                content: "checking".into(),
                usage: None,
                agent_id: String::new(),
                session_id: String::new(),
            },
            Message::tool_output("toolu_1".into(), r#"{"ok":true}"#),
        ]);
        assert_eq!(msgs[1]["content"][1]["type"], "tool_use");
        assert_eq!(msgs[2]["content"][0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn maps_tool_call_and_usage() {
        let response = json!({
            "id": "msg_1",
            "model": "claude-x",
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "content": [{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]
        });
        let mapped = map_response(response, true).unwrap();
        assert_eq!(mapped.usage.unwrap().total(), Some(10));
        match mapped.output {
            ClientOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].id, "toolu_1"),
            _ => panic!("expected tool call"),
        }
    }
}
