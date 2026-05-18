use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use super::super::tools::ToolDefinition;
use super::{
    Attachment, Client, ClientError, ClientOptions, ClientOutput, ClientResponse, EmbedRequest,
    EmbedResponse, LlmUrl, Message, Provider, Role, TokenUsage, ToolCall, ToolChoice,
    configured_base_url, decode_output_text, required_api_key, validate_tools,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

struct OpenAiClient {
    http: HttpClient,
    api_key: String,
    base_url: String,
    model: String,
    options: ClientOptions,
}

pub fn new_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    let api_key = required_api_key(url, "OPENAI_API_KEY")?;
    Ok(Box::new(OpenAiClient {
        http: HttpClient::new(),
        api_key,
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
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
        let wants_json_output = !tools_enabled && self.options.wants_json_output();
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);

        let response: Value = self
            .http
            .post(responses_endpoint(&self.base_url))
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

        map_response(response, tools_enabled, wants_json_output)
    }

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        let payload = json!({
            "model": self.model,
            "input": request.input,
            "encoding_format": "float",
        });
        let response: Value = self
            .http
            .post(embeddings_endpoint(&self.base_url))
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

fn responses_endpoint(base_url: &str) -> String {
    format!("{}/responses", base_url.trim_end_matches('/'))
}

fn embeddings_endpoint(base_url: &str) -> String {
    format!("{}/embeddings", base_url.trim_end_matches('/'))
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

    if let Some(preamble) = options.effective_preamble() {
        payload["instructions"] = Value::String(preamble);
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        payload["tool_choice"] = match options.tool_choice {
            ToolChoice::Required => Value::String("required".to_string()),
            ToolChoice::Auto => Value::String("auto".to_string()),
            ToolChoice::Disabled => Value::String("none".to_string()),
        };
    } else if options.wants_json_output() {
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
            Role::User => input.push(build_user_input(msg)),
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
            Role::Tool { call_id } => {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": msg.content,
                }));
                push_tool_attachment_inputs(&mut input, &msg.attachments);
            }
        }
    }
    input
}

fn build_user_input(msg: &Message) -> Value {
    if msg.attachments.is_empty() {
        return json!({ "role": "user", "content": msg.content });
    }

    let mut content = msg
        .attachments
        .iter()
        .filter_map(openai_image_content)
        .collect::<Vec<_>>();
    if !msg.content.is_empty() {
        content.push(json!({ "type": "input_text", "text": msg.content }));
    }
    json!({ "role": "user", "content": content })
}

fn push_tool_attachment_inputs(input: &mut Vec<Value>, attachments: &[Attachment]) {
    for image_url in attachments.iter().filter_map(openai_image_url) {
        input.push(json!({
            "role": "user",
            "content": [{
                "type": "input_image",
                "image_url": image_url,
            }]
        }));
    }
}

fn openai_image_content(att: &Attachment) -> Option<Value> {
    openai_image_url(att).map(|image_url| {
        json!({
            "type": "input_image",
            "image_url": image_url,
        })
    })
}

fn openai_image_url(att: &Attachment) -> Option<String> {
    match att {
        Attachment::Inline { mime_type, data } if mime_type.starts_with("image/") => {
            Some(format!("data:{mime_type};base64,{data}"))
        }
        Attachment::Url { mime_type, url } if mime_type.starts_with("image/") => Some(url.clone()),
        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. } => {
            tracing::warn!(mime_type = %mime_type, "OpenAI attachment support is image-only for this path; dropping attachment");
            None
        }
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before OpenAI serialization; dropping");
            None
        }
    }
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

fn map_response(
    response: Value,
    tools_enabled: bool,
    wants_json_output: bool,
) -> Result<ClientResponse, ClientError> {
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
        ClientOutput::Output(decode_output_text(&text, wants_json_output)?),
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
    fn custom_base_url_builds_openai_responses_endpoint() {
        assert_eq!(
            responses_endpoint("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/responses"
        );
        assert_eq!(
            embeddings_endpoint("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1/embeddings"
        );
    }

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
    fn responses_payload_appends_input_schema_to_instructions() {
        let options = ClientOptions::default()
            .with_preamble("You are helpful.")
            .with_input_schema(json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string" }
                },
                "required": ["kind"]
            }));

        let payload = build_payload("custom-model", &options, &[Message::user("hi")], false);

        let instructions = payload["instructions"]
            .as_str()
            .expect("instructions should be a string");
        assert!(instructions.contains("You are helpful."));
        assert!(instructions.contains("The user message is JSON."));
        assert!(instructions.contains("\"required\":[\"kind\"]"));
    }

    #[test]
    fn payload_without_input_schema_uses_text_mode() {
        let payload = build_payload(
            "custom-model",
            &ClientOptions::default(),
            &[Message::user("hi")],
            false,
        );
        assert!(payload.get("text").is_none());
    }

    #[test]
    fn payload_with_input_schema_and_no_output_schema_uses_json_object_mode() {
        let payload = build_payload(
            "custom-model",
            &ClientOptions::default().with_input_schema(json!({ "type": "object" })),
            &[Message::user("hi")],
            false,
        );
        assert_eq!(payload["text"]["format"]["type"], "json_object");
    }

    /// Non-image attachments are ignored on the OpenAI image-only wire path.
    #[test]
    fn non_image_user_attachments_are_dropped() {
        let payload = build_payload(
            "custom-model",
            &ClientOptions::default(),
            &[Message {
                role: Role::User,
                content: "describe this".into(),
                attachments: vec![Attachment::Inline {
                    mime_type: "application/pdf".into(),
                    data: "aGVsbG8=".into(),
                }],
                usage: None,
            }],
            false,
        );

        assert_eq!(payload["input"][0]["role"], "user");
        assert_eq!(payload["input"][0]["content"].as_array().unwrap().len(), 1);
        assert_eq!(payload["input"][0]["content"][0]["type"], "input_text");
    }

    /// User attachments are encoded as OpenAI `input_image` content items.
    #[test]
    fn user_attachments_use_content_array() {
        let payload = build_payload(
            "custom-model",
            &ClientOptions::default(),
            &[Message {
                role: Role::User,
                content: "describe this".into(),
                attachments: vec![Attachment::Inline {
                    mime_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                }],
                usage: None,
            }],
            false,
        );

        assert_eq!(payload["input"][0]["role"], "user");
        assert_eq!(payload["input"][0]["content"][0]["type"], "input_image");
        assert_eq!(payload["input"][0]["content"][1]["type"], "input_text");
    }

    #[test]
    fn schema_and_tools_openai_prefers_tools_over_structured_output() {
        let options = ClientOptions::default()
            .with_tool_choice(ToolChoice::Required)
            .with_output_schema(json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }))
            .with_tools(vec![ToolDefinition {
                name: "submit".into(),
                description: "Submit the final answer.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    },
                    "required": ["answer"]
                }),
            }]);

        let payload = build_payload("custom-model", &options, &[Message::user("hi")], true);

        assert!(payload.get("text").is_none());
        assert_eq!(payload["tools"][0]["name"], "submit");
        assert_eq!(payload["tool_choice"], "required");
    }

    #[test]
    fn maps_response_usage_and_tool_call() {
        let response = json!({
            "id": "resp_1",
            "model": "gpt-x",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "output": [{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}]
        });
        let mapped = map_response(response, true, false).unwrap();
        assert_eq!(mapped.usage.unwrap().total(), Some(15));
        assert_eq!(mapped.provider_model.as_deref(), Some("gpt-x"));
        match mapped.output {
            ClientOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].id, "call_1"),
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn map_response_without_json_mode_returns_string() {
        let response = json!({
            "id": "resp_1",
            "model": "gpt-x",
            "output_text": "plain text"
        });
        let mapped = map_response(response, false, false).unwrap();
        match mapped.output {
            ClientOutput::Output(Value::String(text)) => assert_eq!(text, "plain text"),
            _ => panic!("expected string output"),
        }
    }
}
