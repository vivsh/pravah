use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use super::super::tools::ToolDefinition;
use super::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, EmbedRequest, EmbedResponse,
    LlmUrl, Message, Provider, Role, TokenUsage, ToolCall, ToolChoice, parse_json_output,
    validate_tools,
};

struct OpenAiClient {
    http: HttpClient,
    api_key: String,
    model: String,
    options: ClientOptions,
}

pub fn new_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    let api_key = url
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| ClientError::Llm("OPENAI_API_KEY is not set".into()))?;
    Ok(Box::new(OpenAiClient {
        http: HttpClient::new(),
        api_key,
        model: url.model.clone(),
        options,
    }))
}

#[async_trait]
impl Client for OpenAiClient {
    fn provider(&self) -> Provider {
        Provider::OpenAi
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        validate_history(messages)?;
        validate_tools(Provider::OpenAi, &self.options.tools)?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);

        let response: Value = self
            .http
            .post("https://api.openai.com/v1/responses")
            .bearer_auth(&self.api_key)
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

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        let payload = json!({
            "model": self.model,
            "input": request.input,
            "encoding_format": "float",
        });
        let response: Value = self
            .http
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .error_for_status()
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .json()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?;
        let values: Vec<f32> = response["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| ClientError::Llm("embedding missing in response".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        Ok(EmbedResponse { values })
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
        "input": build_input(messages),
    });

    if let Some(t) = options.temperature {
        payload["temperature"] = json!(t);
    }

    if let Some(preamble) = &options.preamble {
        payload["instructions"] = Value::String(preamble.clone());
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        payload["tool_choice"] = match options.tool_choice {
            ToolChoice::Required => Value::String("required".to_string()),
            ToolChoice::Auto => Value::String("auto".to_string()),
            ToolChoice::Disabled => Value::String("none".to_string()),
        };
    } else {
        payload["text"] = json!({
            "format": match &options.output_schema {
                Some(schema) => json!({
                    "type": "json_schema",
                    "name": "agent_output",
                    "schema": schema,
                    "strict": true
                }),
                None => json!({ "type": "json_object" }),
            }
        });
    }

    payload
}

fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut input = Vec::new();
    for msg in messages {
        match &msg.role {
            Role::System => input.push(json!({ "role": "system", "content": msg.content })),
            Role::User => input.push(json!({ "role": "user", "content": msg.content })),
            Role::Assistant => input.push(json!({ "role": "assistant", "content": msg.content })),
            Role::AssistantToolCalls { calls } => {
                for call in calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.args.to_string(),
                    }));
                }
            }
            Role::Tool { call_id } => input.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": msg.content,
            })),
        }
    }
    input
}

fn build_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": true,
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
        "status": response.get("status").cloned().unwrap_or(Value::Null),
    }));

    let calls = collect_tool_calls(&response)?;
    if !calls.is_empty() {
        return Ok(ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::ToolCalls {
                thought: collect_text(&response),
                calls,
            },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(metadata));
    }

    if tools_enabled {
        return Err(ClientError::MissingToolCalls(collect_text(&response)));
    }

    let text = collect_text(&response).ok_or(ClientError::EmptyResponse)?;
    Ok(ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::Output(parse_json_output(&text)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

fn collect_tool_calls(response: &Value) -> Result<Vec<ToolCall>, ClientError> {
    let mut calls = Vec::new();
    if let Some(output) = response.get("output").and_then(Value::as_array) {
        for item in output {
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                continue;
            }
            let id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ClientError::Validation("OpenAI function call missing call_id".into())
                })?;
            let name = item.get("name").and_then(Value::as_str).ok_or_else(|| {
                ClientError::Validation("OpenAI function call missing name".into())
            })?;
            let raw_args = item
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
    }
    Ok(calls)
}

fn collect_text(response: &Value) -> Option<String> {
    if let Some(text) = response.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    let mut out = String::new();
    for item in response.get("output").and_then(Value::as_array)? {
        for content in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if matches!(
                content.get("type").and_then(Value::as_str),
                Some("output_text" | "text")
            ) {
                if let Some(text) = content.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: value
            .get("input_tokens")
            .or_else(|| value.get("prompt_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        output: value
            .get("output_tokens")
            .or_else(|| value.get("completion_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn responses_payload_uses_schema_and_required_tools() {
        let options = ClientOptions::default()
            .with_tool_choice(ToolChoice::Required)
            .with_tools(vec![ToolDefinition {
                name: "lookup".into(),
                description: "Lookup a thing.".into(),
                parameters: json!({"type":"object","properties":{}}),
            }]);
        let payload = build_payload("custom-model", &options, &[Message::user("hi")], true);
        assert_eq!(payload["model"], "custom-model");
        assert_eq!(payload["tool_choice"], "required");
        assert_eq!(payload["tools"][0]["name"], "lookup");
    }

    #[test]
    fn maps_response_usage_and_tool_call() {
        let response = json!({
            "id": "resp_1",
            "model": "gpt-x",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "output": [{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}]
        });
        let mapped = map_response(response, true).unwrap();
        assert_eq!(mapped.usage.unwrap().total(), Some(15));
        assert_eq!(mapped.provider_model.as_deref(), Some("gpt-x"));
        match mapped.output {
            ClientOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].id, "call_1"),
            _ => panic!("expected tool calls"),
        }
    }
}
