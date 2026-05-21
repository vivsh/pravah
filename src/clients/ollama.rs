use std::borrow::Cow;

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::Serialize;
use serde_json::{Value, json};

use super::super::tools::ToolDefinition;
use super::{
    Attachment, Client, ClientError, ClientOptions, ClientOutput, ClientResponse, EmbedRequest,
    EmbedResponse, LlmUrl, Message, Provider, Role, ThinkingLevel, TokenUsage, ToolCall, ToolChoice,
    configured_base_url, decode_output_text, optional_api_key, parse_json_output,
    validate_tools,
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

struct OllamaClient {
    http: HttpClient,
    api_key: Option<String>,
    base_url: String,
    model: String,
    options: ClientOptions,
}

impl OllamaClient {
    async fn post_json<T: Serialize + ?Sized>(
        &self,
        endpoint: &str,
        payload: &T,
    ) -> Result<Value, ClientError> {
        with_bearer_auth(self.http.post(endpoint), self.api_key.as_deref())
            .json(payload)
            .send()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .error_for_status()
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .json()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))
    }
}

pub fn new_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    Ok(Box::new(OllamaClient {
        http: HttpClient::new(),
        api_key: optional_api_key(url, "OLLAMA_API_KEY"),
        base_url: configured_base_url(url, DEFAULT_BASE_URL),
        model: url.model.clone(),
        options,
    }))
}

#[async_trait]
impl Client for OllamaClient {
    fn provider(&self) -> Provider {
        Provider::Ollama
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        validate_history(messages)?;
        validate_tools(Provider::Ollama, &self.options.tools)?;

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        let wants_json_output = !tools_enabled && self.options.wants_json_output();
        let endpoint = chat_completions_endpoint(&self.base_url);

        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);
        let response = self.post_json(&endpoint, &payload).await?;

        match map_response(response, tools_enabled, wants_json_output) {
            Err(ClientError::MissingToolCalls(Some(ref text))) if tools_enabled => {
                tracing::warn!("model answered without tool call; nudging to retry");
                let tool_names: Vec<&str> =
                    self.options.tools.iter().map(|t| t.name.as_str()).collect();
                let nudge = format!(
                    "Your previous response was plain text, which is not allowed. \
                     You MUST call one of the following tools: {}. \
                     Do not write any prose — issue a tool call now.",
                    tool_names.join(", ")
                );
                let mut nudged = messages.to_vec();
                nudged.push(Message::assistant(text.clone()));
                nudged.push(Message::user(nudge));
                let nudged_payload = build_payload(&self.model, &self.options, &nudged, true);
                let nudged_response = self.post_json(&endpoint, &nudged_payload).await?;
                map_response(nudged_response, true, false)
            }
            other => other,
        }
    }

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        let endpoint = embed_endpoint(&self.base_url);
        let payload = json!({ "model": self.model, "input": request.input });
        let response = self.post_json(&endpoint, &payload).await?;
        let values: Vec<f32> = response["embeddings"][0]
            .as_array()
            .ok_or_else(|| ClientError::Llm("embeddings missing in response".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        Ok(EmbedResponse { values })
    }
}

fn chat_completions_endpoint(base_url: &str) -> String {
    format!("{}/v1/chat/completions", base_url.trim_end_matches('/'))
}

fn embed_endpoint(base_url: &str) -> String {
    format!("{}/api/embed", base_url.trim_end_matches('/'))
}

fn with_bearer_auth(
    request: reqwest::RequestBuilder,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    match api_key {
        Some(api_key) => request.bearer_auth(api_key),
        None => request,
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
    let thinking_enabled = options
        .thinking
        .as_ref()
        .map_or(false, |t| *t != ThinkingLevel::Off);
    let schema_hint = if !tools_enabled && options.wants_json_output() {
        options.output_schema.as_ref()
    } else {
        None
    };

    let mut payload = json!({
        "model": model,
        "messages": build_messages(
            messages,
            options.effective_preamble().as_deref(),
            model,
            thinking_enabled,
            schema_hint,
        ),
        "stream": false,
    });

    if let Some(t) = options.temperature {
        payload["temperature"] = json!(t);
    }

    if tools_enabled {
        payload["tools"] = Value::Array(build_tools(&options.tools));
        if options.tool_choice == ToolChoice::Required {
            payload["tool_choice"] = Value::String("required".into());
        }
    } else if options.wants_json_output() && !thinking_enabled {
        payload["response_format"] = json!({ "type": "json_object" });
    }

    payload
}

fn build_messages(
    history: &[Message],
    preamble: Option<&str>,
    model: &str,
    thinking: bool,
    schema_hint: Option<&Value>,
) -> Vec<Value> {
    let mut out = Vec::with_capacity(history.len() + if preamble.is_some() || schema_hint.is_some() { 1 } else { 0 });
    if let Some(system) = combined_system_message(preamble, schema_hint) {
        out.push(json!({ "role": "system", "content": system }));
    }

    let mut first_user = true;
    for msg in history {
        match &msg.role {
            Role::System => out.push(json!({ "role": "system", "content": msg.content })),
            Role::User => out.push(build_user_message(msg, &mut first_user, model, thinking)),
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
            Role::Tool { call_id } => {
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": msg.content,
                }));
                push_tool_attachment_messages(&mut out, &msg.attachments);
            }
        }
    }
    out
}

fn combined_system_message(preamble: Option<&str>, schema_hint: Option<&Value>) -> Option<String> {
    match (preamble, schema_hint) {
        (Some(preamble), Some(schema)) => Some(format!(
            "{preamble}\n\nRespond with JSON that matches the following schema:\n{schema}"
        )),
        (Some(preamble), None) => Some(preamble.to_owned()),
        (None, Some(schema)) => Some(format!(
            "Respond with JSON that matches the following schema:\n{schema}"
        )),
        (None, None) => None,
    }
}

fn build_user_message(message: &Message, first_user: &mut bool, model: &str, thinking: bool) -> Value {
    let content = user_content(message, first_user, model, thinking);
    if message.attachments.is_empty() {
        return json!({ "role": "user", "content": content });
    }

    let mut parts = Vec::with_capacity(
        message.attachments.len() + if content.is_empty() { 0 } else { 1 },
    );
    parts.extend(message.attachments.iter().filter_map(ollama_image_part));
    if !content.is_empty() {
        parts.push(json!({ "type": "text", "text": content }));
    }
    json!({ "role": "user", "content": parts })
}

fn push_tool_attachment_messages(out: &mut Vec<Value>, attachments: &[Attachment]) {
    for part in attachments.iter().filter_map(ollama_image_part) {
        out.push(json!({
            "role": "user",
            "content": [part]
        }));
    }
}

fn user_content(message: &Message, first_user: &mut bool, model: &str, thinking: bool) -> String {
    if *first_user && !thinking && model.starts_with("qwen3") {
        *first_user = false;
        format!("/no_think\n\n{}", message.content)
    } else {
        *first_user = false;
        message.content.clone()
    }
}

fn ollama_image_part(att: &Attachment) -> Option<Value> {
    let url = match att {
        Attachment::Inline { mime_type, data } if mime_type.starts_with("image/") => {
            Some(format!("data:{mime_type};base64,{data}"))
        }
        Attachment::Url { mime_type, url } if mime_type.starts_with("image/") => Some(url.clone()),
        Attachment::Inline { mime_type, .. } | Attachment::Url { mime_type, .. } => {
            tracing::warn!(mime_type = %mime_type, "Ollama attachment support is image-only for this path; dropping attachment");
            None
        }
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before Ollama serialization; dropping");
            None
        }
    }?;
    Some(json!({
        "type": "image_url",
        "image_url": {
            "url": url,
        }
    }))
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
    let text = strip_thinking(text);
    Ok(ClientResponse::new(
        Provider::Ollama,
        ClientOutput::Output(decode_ollama_output(text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(metadata))
}

fn decode_ollama_output(text: &str, wants_json_output: bool) -> Result<Value, ClientError> {
    if !wants_json_output {
        return decode_output_text(text, false);
    }

    let sanitized = sanitize_json_markdown(text);
    parse_json_output(sanitized.as_ref()).map(strip_markdown_json_keys)
}

fn strip_thinking(text: &str) -> &str {
    // qwen3 models emit <think>...</think> before the final answer when thinking is enabled.
    // Strip that block so callers receive only the answer/JSON.
    if let Some(end) = text.find("</think>") {
        text[end + "</think>".len()..].trim_start()
    } else {
        text
    }
}

fn sanitize_json_markdown(text: &str) -> Cow<'_, str> {
    let unfenced = strip_json_code_fence(text).unwrap_or(text);
    let repaired = repair_markdown_wrapped_keys(unfenced);

    if unfenced == text {
        repaired
    } else {
        match repaired {
            Cow::Borrowed(_) => Cow::Owned(unfenced.to_owned()),
            Cow::Owned(value) => Cow::Owned(value),
        }
    }
}

fn strip_json_code_fence(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return None;
    }

    let body_start = trimmed.find('\n')? + 1;
    let body = &trimmed[body_start..];
    let body_end = body.rfind("\n```")?;
    Some(body[..body_end].trim())
}

fn repair_markdown_wrapped_keys(text: &str) -> Cow<'_, str> {
    let mut repaired = String::with_capacity(text.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut changed = false;

    while index < text.len() {
        let rest = &text[index..];
        let Some(ch) = rest.chars().next() else {
            break;
        };

        if in_string {
            repaired.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            index += ch.len_utf8();
            continue;
        }

        if ch == '"' {
            in_string = true;
            repaired.push(ch);
            index += ch.len_utf8();
            continue;
        }

        if let Some((close_end, colon_index, key)) = markdown_wrapped_key(rest) {
            push_json_key(&mut repaired, key);
            repaired.push_str(&rest[close_end..colon_index]);
            index += colon_index;
            changed = true;
            continue;
        }

        repaired.push(ch);
        index += ch.len_utf8();
    }

    if changed {
        Cow::Owned(repaired)
    } else {
        Cow::Borrowed(text)
    }
}

fn markdown_wrapped_key(text: &str) -> Option<(usize, usize, &str)> {
    let marker = if text.starts_with("**") {
        "**"
    } else if text.starts_with("__") {
        "__"
    } else {
        return None;
    };

    let inner = &text[marker.len()..];
    let close_start = marker.len() + inner.find(marker)?;
    let key = &text[marker.len()..close_start];
    if key.is_empty() || key.contains('\n') {
        return None;
    }

    let close_end = close_start + marker.len();
    for (offset, ch) in text[close_end..].char_indices() {
        if ch.is_whitespace() {
            continue;
        }
        return (ch == ':').then_some((close_end, close_end + offset, key));
    }
    None
}

fn push_json_key(output: &mut String, key: &str) {
    output.push('"');
    for ch in key.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            _ => output.push(ch),
        }
    }
    output.push('"');
}

fn strip_markdown_json_keys(value: Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(strip_markdown_json_keys).collect()),
        Value::Object(entries) => Value::Object(
            entries
                .into_iter()
                .map(|(key, value)| (strip_markdown_key(&key).to_owned(), strip_markdown_json_keys(value)))
                .collect(),
        ),
        other => other,
    }
}

fn strip_markdown_key(key: &str) -> &str {
    if key.len() > 4 {
        for marker in ["**", "__"] {
            if key.starts_with(marker) && key.ends_with(marker) {
                return &key[marker.len()..key.len() - marker.len()];
            }
        }
    }
    key
}

fn collect_tool_calls(message: &Value) -> Result<Vec<ToolCall>, ClientError> {
    if let Some(items) = message.get("tool_calls").and_then(Value::as_array) {
        return parse_json_tool_calls(items);
    }
    if let Some(content) = message.get("content").and_then(Value::as_str) {
        return parse_content_tool_calls(content);
    }
    Ok(Vec::new())
}

fn parse_json_tool_calls(items: &[Value]) -> Result<Vec<ToolCall>, ClientError> {
    let mut calls = Vec::with_capacity(items.len());
    for item in items {
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

/// Parses tool calls from content text when the model emits them in the
/// `<function=NAME><parameter=KEY>VALUE</parameter></function>` format
/// instead of the standard `tool_calls` JSON field.
fn parse_content_tool_calls(content: &str) -> Result<Vec<ToolCall>, ClientError> {
    let mut calls = Vec::new();
    let mut remaining = content;
    while let Some(tag_start) = remaining.find("<function=") {
        let after_tag = &remaining[tag_start + "<function=".len()..];
        let Some(name_end) = after_tag.find('>') else {
            break;
        };
        let name = after_tag[..name_end].trim();
        let body = &after_tag[name_end + 1..];
        let body_end = body.find("</function>").unwrap_or(body.len());
        let args = parse_function_params(&body[..body_end]);
        calls.push(ToolCall {
            id: uuid::Uuid::now_v7().to_string(),
            name: name.to_string(),
            args,
            thought_signatures: None,
        });
        let consumed = tag_start + "<function=".len() + name_end + 1 + body_end;
        let skip = consumed + "</function>".len();
        remaining = if skip < remaining.len() { &remaining[skip..] } else { "" };
    }
    Ok(calls)
}

fn parse_function_params(text: &str) -> Value {
    let mut map = serde_json::Map::new();
    let mut remaining = text;
    while let Some(tag_start) = remaining.find("<parameter=") {
        let after_tag = &remaining[tag_start + "<parameter=".len()..];
        let Some(key_end) = after_tag.find('>') else {
            break;
        };
        let key = after_tag[..key_end].trim();
        let value_text = &after_tag[key_end + 1..];
        let end = value_text.find("</parameter>").unwrap_or(value_text.len());
        let raw = value_text[..end].trim();
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        map.insert(key.to_string(), value);
        let consumed = tag_start + "<parameter=".len() + key_end + 1 + end;
        let skip = consumed + "</parameter>".len();
        remaining = if skip < remaining.len() { &remaining[skip..] } else { "" };
    }
    Value::Object(map)
}

fn usage_from_value(value: &Value) -> TokenUsage {
    TokenUsage {
        input: token_count(value, "prompt_tokens"),
        output: token_count(value, "completion_tokens"),
    }
}

fn token_count(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn custom_base_url_builds_ollama_endpoints() {
        assert_eq!(
            chat_completions_endpoint("https://ollama-proxy.example/"),
            "https://ollama-proxy.example/v1/chat/completions"
        );
        assert_eq!(
            embed_endpoint("https://ollama-proxy.example"),
            "https://ollama-proxy.example/api/embed"
        );
    }

    #[test]
    fn qwen_no_think_is_added_to_first_user_message() {
        let messages = build_messages(&[Message::user("do it")], None, "qwen3:8b", false, None);
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .starts_with("/no_think")
        );
    }

    /// User attachments are emitted as OpenAI-compatible image_url parts.
    #[test]
    fn user_attachments_use_content_parts() {
        let messages = build_messages(
            &[Message {
                role: Role::User,
                content: "describe this".into(),
                attachments: vec![Attachment::Inline {
                    mime_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                }],
                usage: None,
            }],
            None,
            "qwen3-vl:8b",
            false,
            None,
        );
        assert_eq!(messages[0]["content"][0]["type"], "image_url");
        assert_eq!(messages[0]["content"][1]["type"], "text");
    }

    /// Tool-result attachments are replayed as synthetic user image turns.
    #[test]
    fn tool_attachments_become_synthetic_user_images() {
        let messages = build_messages(
            &[Message {
                role: Role::Tool {
                    call_id: "call-1".into(),
                },
                content: "done".into(),
                attachments: vec![Attachment::Inline {
                    mime_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                }],
                usage: None,
            }],
            None,
            "qwen3-vl:8b",
            false,
            None,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "image_url");
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
        assert!(payload.get("response_format").is_none());
    }

    /// Structured-output mode includes the provided output schema.
    #[test]
    fn payload_uses_output_schema_when_present() {
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        });
        let payload = build_payload(
            "custom-local",
            &ClientOptions::default()
                .with_input_schema(json!({ "type": "object" }))
                .with_output_schema(schema.clone()),
            &[Message::user("hi")],
            false,
        );

        assert_eq!(payload["response_format"]["type"], "json_object");
        // schema is injected as a system message, not in response_format
        let system_msgs: Vec<_> = payload["messages"]
            .as_array().unwrap()
            .iter()
            .filter(|m| m["role"] == "system")
            .collect();
        assert!(system_msgs.iter().any(|m| m["content"].as_str().unwrap_or("").contains("answer")));
    }

    #[test]
    fn payload_with_input_schema_and_no_output_schema_uses_json_object_mode() {
        let payload = build_payload(
            "custom-local",
            &ClientOptions::default().with_input_schema(json!({ "type": "object" })),
            &[Message::user("hi")],
            false,
        );
        assert_eq!(payload["response_format"]["type"], "json_object");
    }

    #[test]
    fn payload_prepends_input_schema_to_system_message() {
        let payload = build_payload(
            "custom-local",
            &ClientOptions::default()
                .with_preamble("You are helpful.")
                .with_input_schema(json!({
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string" }
                    },
                    "required": ["kind"]
                })),
            &[Message::user("hi")],
            false,
        );

        let system = payload["messages"][0]["content"]
            .as_str()
            .expect("system message should be a string");
        assert!(system.contains("You are helpful."));
        assert!(system.contains("The user message is JSON."));
        assert!(system.contains("\"required\":[\"kind\"]"));
    }

    #[test]
    fn schema_and_tools_ollama_prefers_tools_over_response_format() {
        let payload = build_payload(
            "custom-local",
            &ClientOptions::default()
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
                }]),
            &[Message::user("hi")],
            true,
        );

        assert!(payload.get("response_format").is_none());
        assert_eq!(payload["tool_choice"], "required");
        assert_eq!(payload["tools"][0]["function"]["name"], "submit");
    }

    #[test]
    fn map_response_without_json_mode_returns_string() {
        let response = json!({
            "model": "local-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "plain text"
                }
            }]
        });
        let mapped = map_response(response, false, false).unwrap();
        match mapped.output {
            ClientOutput::Output(Value::String(text)) => assert_eq!(text, "plain text"),
            _ => panic!("expected string output"),
        }
    }

    /// Markdown wrappers around bare JSON keys are repaired before deserialization.
    #[test]
    fn map_response_repairs_markdown_wrapped_keys_in_json_mode() {
        let response = json!({
            "model": "local-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": r#"{"inferences":[],**progress**:75,"queries":[]}"#
                }
            }]
        });

        let mapped = map_response(response, false, true).unwrap();
        match mapped.output {
            ClientOutput::Output(Value::Object(output)) => {
                assert_eq!(output.get("progress"), Some(&json!(75)));
            }
            _ => panic!("expected JSON object output"),
        }
    }

    /// Fenced JSON and markdown-decorated keys are normalized before parsing.
    #[test]
    fn sanitize_json_markdown_strips_fences_and_bold_keys() {
        let sanitized = sanitize_json_markdown("```json\n{\"a\":1, **progress**: 75}\n```");
        assert_eq!(sanitized.as_ref(), "{\"a\":1, \"progress\": 75}");
    }

    /// Models that emit tool calls as XML-style text in the content field are parsed correctly.
    #[test]
    fn parse_content_tool_calls_extracts_function_and_params() {
        let content = "<function=file_search>\n<parameter=query>\nforgot password\n</parameter>\n<parameter=globs>\n[\"**/*auth*.js\"]\n</parameter>\n</function>";
        let calls = parse_content_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file_search");
        assert_eq!(calls[0].args["query"], "forgot password");
        assert_eq!(calls[0].args["globs"][0], "**/*auth*.js");
    }

    /// Multiple tool calls in content are all extracted.
    #[test]
    fn parse_content_tool_calls_handles_multiple_functions() {
        let content = "<function=search><parameter=q>hello</parameter></function><function=fetch><parameter=url>http://x</parameter></function>";
        let calls = parse_content_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "fetch");
    }

    /// Content with no function tags yields an empty vec (not an error).
    #[test]
    fn parse_content_tool_calls_returns_empty_on_plain_text() {
        let calls = parse_content_tool_calls("Just a normal response.").unwrap();
        assert!(calls.is_empty());
    }

    /// collect_tool_calls falls back to content parsing when tool_calls field is absent.
    #[test]
    fn collect_tool_calls_falls_back_to_content_when_no_tool_calls_field() {
        let message = json!({
            "role": "assistant",
            "content": "<function=my_tool><parameter=x>42</parameter></function>"
        });
        let calls = collect_tool_calls(&message).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "my_tool");
        assert_eq!(calls[0].args["x"], 42);
    }
}
