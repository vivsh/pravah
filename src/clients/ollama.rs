use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use super::super::tools::ToolDefinition;
use super::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmUrl, Message, Provider,
    Role, TokenUsage, ToolCall, ToolChoice, parse_json_output, validate_tools,
};

struct OllamaClient {
    http: HttpClient,
    base_url: String,
    model: String,
    options: ClientOptions,
}

pub fn new_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    let base_url = url
        .base_url
        .clone()
        .ok_or_else(|| ClientError::InvalidUrl("ollama URL missing base URL".into()))?;
    Ok(Box::new(OllamaClient {
        http: HttpClient::new(),
        base_url,
        model: url.model.clone(),
        options,
    }))
}

#[async_trait]
impl Client for OllamaClient {
    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        validate_history(messages)?;
        validate_tools(Provider::Ollama, &self.options.tools)?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);
        let endpoint = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );

        let response: Value = self
            .http
            .post(endpoint)
            .bearer_auth("ollama")
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
        "messages": build_messages(messages, options.preamble.as_deref(), model, options.thinking),
        "stream": false,
    });

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        if options.tool_choice == ToolChoice::Required {
            payload["tool_choice"] = Value::String("required".into());
        }
    } else {
        payload["response_format"] = json!({ "type": "json_object" });
    }

    payload
}

fn build_messages(
    history: &[Message],
    preamble: Option<&str>,
    model: &str,
    thinking: bool,
) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(preamble) = preamble {
        out.push(json!({ "role": "system", "content": preamble }));
    }

    let mut first_user = true;
    for msg in history {
        match &msg.role {
            Role::System => out.push(json!({ "role": "system", "content": msg.content })),
            Role::User => {
                let content = if first_user && !thinking && model.starts_with("qwen3") {
                    first_user = false;
                    format!("/no_think\n\n{}", msg.content)
                } else {
                    first_user = false;
                    msg.content.clone()
                };
                out.push(json!({ "role": "user", "content": content }));
            }
            Role::Assistant => out.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                let tool_calls: Vec<Value> = calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.args.to_string(),
                            }
                        })
                    })
                    .collect();
                out.push(json!({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_calls,
                }));
            }
            Role::Tool { call_id } => out.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": msg.content,
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
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
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
    }));
    let message = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .ok_or(ClientError::EmptyResponse)?;

    let calls = collect_tool_calls(message)?;
    if !calls.is_empty() {
        return Ok(ClientResponse::new(
            Provider::Ollama,
            ClientOutput::ToolCalls {
                thought: message
                    .get("content")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                calls,
            },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(metadata));
    }

    if tools_enabled {
        return Err(ClientError::MissingToolCalls(
            message
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_string),
        ));
    }

    let text = message
        .get("content")
        .and_then(Value::as_str)
        .ok_or(ClientError::EmptyResponse)?;
    Ok(ClientResponse::new(
        Provider::Ollama,
        ClientOutput::Output(parse_json_output(text)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

fn collect_tool_calls(message: &Value) -> Result<Vec<ToolCall>, ClientError> {
    let mut calls = Vec::new();
    for item in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ClientError::Validation("Ollama tool call missing id".into()))?;
        let function = item
            .get("function")
            .ok_or_else(|| ClientError::Validation("Ollama tool call missing function".into()))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ClientError::Validation("Ollama tool call missing function name".into())
            })?;
        let raw_args = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let args = serde_json::from_str(raw_args).map_err(|e| ClientError::Deserialize {
            source: e,
            raw: raw_args.to_string(),
        })?;
        calls.push(ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            args,
            thought_signatures: None,
        });
    }
    Ok(calls)
}

fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        output: value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_no_think_is_added_to_first_user_message() {
        let messages = build_messages(&[Message::user("do it")], None, "qwen3:8b", false);
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .starts_with("/no_think")
        );
    }

    #[test]
    fn payload_uses_supplied_model() {
        let payload = build_payload(
            "custom-local",
            &ClientOptions::default(),
            &[Message::user("hi")],
            false,
        );
        assert_eq!(payload["model"], "custom-local");
        assert_eq!(payload["response_format"]["type"], "json_object");
    }
}
