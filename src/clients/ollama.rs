use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde_json::{Value, json};

use super::super::tools::ToolDefinition;
use super::{
    Attachment, Client, ClientError, ClientOptions, ClientOutput, ClientResponse, EmbedRequest,
    EmbedResponse, LlmUrl, Message, Provider, Role, TokenUsage, ToolCall, ToolChoice,
    configured_base_url, decode_output_text, optional_api_key, validate_tools,
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

struct OllamaClient {
    http: HttpClient,
    api_key: Option<String>,
    base_url: String,
    model: String,
    options: ClientOptions,
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
        let payload = build_payload(&self.model, &self.options, messages, tools_enabled);
        let endpoint = chat_completions_endpoint(&self.base_url);

        let response: Value = with_bearer_auth(self.http.post(endpoint), self.api_key.as_deref())
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
        let endpoint = embed_endpoint(&self.base_url);
        let payload = json!({ "model": self.model, "input": request.input });
        let response: Value = with_bearer_auth(self.http.post(endpoint), self.api_key.as_deref())
            .json(&payload)
            .send()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .error_for_status()
            .map_err(|e| ClientError::Llm(e.to_string()))?
            .json()
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?;
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
    let mut payload = json!({
        "model": model,
        "messages": build_messages(
            messages,
            options.effective_preamble().as_deref(),
            model,
            options.thinking,
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
    } else if options.wants_json_output() {
        payload["response_format"] = match &options.output_schema {
            Some(schema) => json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_output",
                    "schema": schema,
                }
            }),
            None => json!({ "type": "json_object" }),
        };
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

fn build_user_message(message: &Message, first_user: &mut bool, model: &str, thinking: bool) -> Value {
    let content = user_content(message, first_user, model, thinking);
    if message.attachments.is_empty() {
        return json!({ "role": "user", "content": content });
    }

    let mut parts = message
        .attachments
        .iter()
        .filter_map(ollama_image_part)
        .collect::<Vec<_>>();
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
    Ok(ClientResponse::new(
        Provider::Ollama,
        ClientOutput::Output(decode_output_text(text, wants_json_output)?),
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
        let messages = build_messages(&[Message::user("do it")], None, "qwen3:8b", false);
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

        assert_eq!(payload["response_format"]["type"], "json_schema");
        assert_eq!(payload["response_format"]["json_schema"]["name"], "agent_output");
        assert_eq!(payload["response_format"]["json_schema"]["schema"], schema);
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
}
