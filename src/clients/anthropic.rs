use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use super::super::tools::ToolDefinition;
use super::{
    Attachment, Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmUrl, Message,
    Provider, Role, ThinkingLevel, TokenUsage, ToolCall, ToolChoice, configured_base_url,
    decode_output_text, required_api_key, validate_tools,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

struct AnthropicClient {
    http: HttpClient,
    api_key: String,
    base_url: String,
    model: String,
    options: ClientOptions,
    url: LlmUrl,
}

pub fn new_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    let api_key = required_api_key(url, "ANTHROPIC_API_KEY")?;
    Ok(Box::new(AnthropicClient {
        http: HttpClient::new(),
        api_key,
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
        model: url.model.clone(),
        options,
        url: url.clone(),
    }))
}

#[async_trait]
impl Client for AnthropicClient {
    fn model_url(&self) -> &LlmUrl {
        &self.url
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        validate_history(messages)?;
        validate_tools(Provider::Anthropic, &self.options.tools)?;

        if matches!(&self.options.thinking, Some(t) if *t != ThinkingLevel::Off) {
            return Err(ClientError::UnsupportedCapability {
                provider: Provider::Anthropic,
                capability: "thinking is not exposed by the Anthropic adapter yet".into(),
            });
        }

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let wants_json_output = self.options.wants_json_output();
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);
        let response = send_messages_request(
            &self.http,
            messages_endpoint(&self.base_url),
            &self.api_key,
            &payload,
        )
        .await?;

        map_response(response, wants_json_output)
    }
}

async fn send_messages_request(
    http: &HttpClient,
    endpoint: String,
    api_key: &str,
    payload: &Value,
) -> Result<Value, ClientError> {
    let response = http
        .post(endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(payload)
        .send()
        .await
        .map_err(|e| ClientError::Llm(e.to_string()))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| ClientError::Llm(e.to_string()))?;
    if !status.is_success() {
        return Err(ClientError::Llm(format_anthropic_http_error(status.as_u16(), &body)));
    }

    serde_json::from_str(&body).map_err(|e| ClientError::Deserialize {
        source: e,
        raw: body,
    })
}

fn messages_endpoint(base_url: &str) -> String {
    format!("{}/messages", base_url.trim_end_matches('/'))
}

fn format_anthropic_http_error(status: u16, body: &str) -> String {
    let response: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => {
            let trimmed = body.trim();
            if trimmed.is_empty() {
                return format!("Anthropic API request failed with HTTP {status}");
            }
            return format!("Anthropic API request failed with HTTP {status}: {trimmed}");
        }
    };

    let error = response.get("error").unwrap_or(&response);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown Anthropic error");
    let error_type = error.get("type").and_then(Value::as_str);
    match error_type {
        Some(error_type) => format!(
            "Anthropic API request failed with HTTP {status} ({error_type}): {message}"
        ),
        None => format!("Anthropic API request failed with HTTP {status}: {message}"),
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

    if let Some(t) = options.temperature {
        payload["temperature"] = json!(t);
    }

    let mut system = Vec::new();
    if let Some(preamble) = options.effective_preamble() {
        system.push(preamble);
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
    }
    if options.wants_json_output() {
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
            Role::User => out.push(build_user_message(msg)),
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
            Role::Tool { call_id } => {
                let mut content = anthropic_image_blocks(&msg.attachments);
                content.push(json!({ "type": "text", "text": msg.content }));
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": content,
                    }]
                }));
            }
        }
    }
    out
}

fn build_user_message(msg: &Message) -> Value {
    if msg.attachments.is_empty() {
        return json!({ "role": "user", "content": msg.content });
    }

    let mut content = anthropic_image_blocks(&msg.attachments);
    if !msg.content.is_empty() {
        content.push(json!({ "type": "text", "text": msg.content }));
    }
    json!({ "role": "user", "content": content })
}

fn anthropic_image_blocks(attachments: &[Attachment]) -> Vec<Value> {
    attachments
        .iter()
        .filter_map(anthropic_image_block)
        .collect()
}

fn anthropic_image_block(att: &Attachment) -> Option<Value> {
    match att {
        Attachment::Inline { mime_type, data } if mime_type.starts_with("image/") => Some(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            }
        })),
        Attachment::Url { mime_type, url } if mime_type.starts_with("image/") => Some(json!({
            "type": "image",
            "source": {
                "type": "url",
                "media_type": mime_type,
                "url": url,
            }
        })),
        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. } => {
            tracing::warn!(mime_type = %mime_type, "Anthropic attachment support is image-only for this path; dropping attachment");
            None
        }
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before Anthropic serialization; dropping");
            None
        }
    }
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

fn map_response(
    response: Value,
    wants_json_output: bool,
) -> Result<ClientResponse, ClientError> {
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
    let text = text.ok_or(ClientError::EmptyResponse)?;
    Ok(ClientResponse::new(
        Provider::Anthropic,
        ClientOutput::Output(decode_output_text(&text, wants_json_output)?),
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
    fn custom_base_url_builds_anthropic_messages_endpoint() {
        assert_eq!(
            messages_endpoint("https://anthropic-proxy.example/v1/"),
            "https://anthropic-proxy.example/v1/messages"
        );
    }

    /// Anthropic HTTP errors keep the API's structured message and type.
    #[test]
    fn formats_structured_http_errors() {
        let msg = format_anthropic_http_error(
            404,
            r#"{"type":"error","error":{"type":"not_found_error","message":"model claude-3-5-haiku-latest not found"}}"#,
        );
        assert!(msg.contains("HTTP 404"));
        assert!(msg.contains("not_found_error"));
        assert!(msg.contains("model claude-3-5-haiku-latest not found"));
    }

    /// Anthropic HTTP errors fall back to the raw body when it is not valid JSON.
    #[test]
    fn formats_unstructured_http_errors() {
        let msg = format_anthropic_http_error(500, "upstream unavailable");
        assert!(msg.contains("HTTP 500"));
        assert!(msg.contains("upstream unavailable"));
    }

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
                content: "checking".into(),                attachments: Vec::new(),                usage: None,
            },
            Message::tool_output("toolu_1".into(), r#"{"ok":true}"#),
        ]);
        assert_eq!(msgs[1]["content"][1]["type"], "tool_use");
        assert_eq!(msgs[2]["content"][0]["tool_use_id"], "toolu_1");
    }

    /// Tool results remain unchanged when a reminder is appended as a later user turn.
    #[test]
    fn messages_keep_tool_result_and_reminder_separate() {
        let msgs = build_messages(&[
            Message {
                role: Role::AssistantToolCalls {
                    calls: vec![ToolCall {
                        id: "toolu_1".into(),
                        name: "lookup".into(),
                        args: json!({"q":"x"}),
                        thought_signatures: None,
                    }],
                },
                content: String::new(),
                attachments: Vec::new(),
                usage: None,
            },
            Message::tool_output("toolu_1".into(), r#"{"ok":true}"#),
            Message::user(
                "<system-reminder><critical>call final_answer</critical></system-reminder>",
            ),
        ]);

        assert_eq!(msgs[1]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[1]["content"][0]["content"][0]["text"], r#"{"ok":true}"#);
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(
            msgs[2]["content"],
            "<system-reminder><critical>call final_answer</critical></system-reminder>"
        );
    }

    #[test]
    fn maps_tool_call_and_usage() {
        let response = json!({
            "id": "msg_1",
            "model": "claude-x",
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "content": [{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]
        });
        let mapped = map_response(response, false).unwrap();
        assert_eq!(mapped.usage.unwrap().total(), Some(10));
        match mapped.output {
            ClientOutput::ToolCalls { calls, .. } => assert_eq!(calls[0].id, "toolu_1"),
            _ => panic!("expected tool call"),
        }
    }

    #[test]
    fn payload_appends_input_schema_to_system_prompt() {
        let options = ClientOptions::default()
            .with_preamble("You are helpful.")
            .with_input_schema(json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string" }
                },
                "required": ["kind"]
            }));

        let payload = build_payload("claude", &options, &[Message::user("hi")], false);

        let system = payload["system"]
            .as_str()
            .expect("system prompt should be a string");
        assert!(system.contains("You are helpful."));
        assert!(system.contains("The user message is JSON."));
        assert!(system.contains("\"required\":[\"kind\"]"));
    }

    #[test]
    fn payload_without_input_schema_keeps_text_mode() {
        let payload = build_payload(
            "claude",
            &ClientOptions::default().with_preamble("You are helpful."),
            &[Message::user("hi")],
            false,
        );
        assert_eq!(payload["system"], "You are helpful.");
    }

    /// Explicit JSON response mode appends a textual JSON-only instruction.
    #[test]
    fn payload_with_json_response_and_no_output_schema_requests_json_textually() {
        let payload = build_payload(
            "claude",
            &ClientOptions::default()
                .with_preamble("You are helpful.")
                .with_response_format(crate::clients::ResponseFormat::Json),
            &[Message::user("hi")],
            false,
        );
        let system = payload["system"].as_str().expect("system prompt should be present");
        assert!(system.contains("Return only valid JSON."));
    }

    /// User attachments are emitted as Anthropic image blocks ahead of text.
    #[test]
    fn user_attachments_use_image_blocks() {
        let msgs = build_messages(&[Message {
            role: Role::User,
            content: "describe this".into(),
            attachments: vec![Attachment::Inline {
                mime_type: "image/png".into(),
                data: "aGVsbG8=".into(),
            }],
            usage: None,
        }]);

        assert_eq!(msgs[0]["content"][0]["type"], "image");
        assert_eq!(msgs[0]["content"][1]["type"], "text");
    }

    #[test]
    fn schema_and_tools_anthropic_sends_both_tools_and_output_hint() {
        let options = ClientOptions::default()
            .with_preamble("You are helpful.")
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

        let payload = build_payload("claude", &options, &[Message::user("hi")], true);

        let system = payload["system"].as_str().expect("system should be a string");
        assert!(system.contains("You are helpful."), "system should contain preamble");
        assert!(system.contains("JSON Schema"), "system should contain JSON schema hint");
        assert_eq!(payload["tool_choice"]["type"], "any");
        assert_eq!(payload["tools"][0]["name"], "submit");
    }

    #[test]
    fn map_response_without_json_mode_returns_string() {
        let response = json!({
            "id": "msg_1",
            "model": "claude-x",
            "content": [{"type":"text","text":"plain text"}]
        });
        let mapped = map_response(response, false).unwrap();
        match mapped.output {
            ClientOutput::Output(Value::String(text)) => assert_eq!(text, "plain text"),
            _ => panic!("expected string output"),
        }
    }
}
