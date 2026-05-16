use async_trait::async_trait;
use genai::Client as GenaiHttpClient;
use genai::ServiceTarget;
use genai::chat::{
    ChatMessage, ChatOptions, ChatRequest, ChatResponse, ChatResponseFormat, JsonSpec,
    Tool as GenaiTool, ToolCall as GenaiToolCall, ToolResponse,
};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use serde_json::Value;

use genai::embed::EmbedOptions;
use super::super::tools::ToolDefinition;
use super::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, EmbedRequest, EmbedResponse,
    LlmUrl, Message, Provider, Role, TokenUsage, ToolCall, ToolChoice, parse_json_output,
    validate_tools,
};

fn build_http_client(url: &LlmUrl) -> GenaiHttpClient {
    let auth_data: Option<AuthData> = if let Some(key) = url.api_key.clone() {
        Some(AuthData::from_single(key))
    } else {
        let env_var = match url.provider {
            Provider::OpenAi => Some("OPENAI_API_KEY"),
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            Provider::Gemini => Some("GEMINI_API_KEY"),
            Provider::Ollama => None,
            Provider::Genai => None,
        };
        env_var
            .and_then(|var| std::env::var(var).ok())
            .map(AuthData::from_single)
    };

    let mut builder = GenaiHttpClient::builder();

    if let Some(auth) = auth_data {
        builder = builder.with_auth_resolver(AuthResolver::from_resolver_fn(move |_| {
            Ok(Some(auth.clone()))
        }));
    }

    if let Some(base_url) = url.base_url.clone() {
        builder = builder.with_service_target_resolver(ServiceTargetResolver::from_resolver_fn(
            move |mut target: ServiceTarget| {
                target.endpoint = Endpoint::from_owned(base_url.clone());
                Ok(target)
            },
        ));
    }

    builder.build()
}

struct GenaiClient {
    client: GenaiHttpClient,
    model_name: String,
    options: ClientOptions,
}

/// Builds provider messages from history.
///
/// For Qwen3 (model name starts with "qwen3") when `thinking` is `false`,
/// prepends `/no_think` to the first user message so the model does not emit
/// verbose chain-of-thought prose.
fn build_genai_messages(
    history: &[Message],
    preamble: Option<&str>,
    model_name: &str,
    thinking: bool,
) -> Vec<ChatMessage> {
    let mut msgs = Vec::with_capacity(history.len() + 1);

    if let Some(p) = preamble {
        msgs.push(ChatMessage::system(p));
    }

    let mut first_user = true;
    for m in history {
        let msg = match &m.role {
            Role::System => ChatMessage::system(&m.content),
            Role::User => {
                let content = if first_user && !thinking && model_name.starts_with("qwen3") {
                    first_user = false;
                    format!("/no_think\n\n{}", m.content)
                } else {
                    first_user = false;
                    m.content.clone()
                };
                ChatMessage::user(&content)
            }
            Role::Assistant => ChatMessage::assistant(&m.content),
            Role::AssistantToolCalls { calls } => {
                let genai_calls: Vec<GenaiToolCall> = calls
                    .iter()
                    .map(|c| GenaiToolCall {
                        call_id: c.id.clone(),
                        fn_name: c.name.clone(),
                        fn_arguments: c.args.clone(),
                        thought_signatures: c.thought_signatures.clone(),
                    })
                    .collect();
                ChatMessage::from(genai_calls)
            }
            Role::Tool { call_id } => ChatMessage::from(ToolResponse::new(call_id, &m.content)),
        };
        msgs.push(msg);
    }
    msgs
}

fn build_request(messages: Vec<ChatMessage>, tools: &[ToolDefinition]) -> ChatRequest {
    let mut req = ChatRequest::new(messages);
    if !tools.is_empty() {
        let genai_tools: Vec<GenaiTool> = tools
            .iter()
            .map(|t| {
                GenaiTool::new(&t.name)
                    .with_description(&t.description)
                    .with_schema(t.parameters.clone())
            })
            .collect();
        req = req.with_tools(genai_tools);
    }
    req
}

fn build_chat_options(schema: Value) -> ChatOptions {
    let spec = JsonSpec::new("agent_output", schema);
    ChatOptions::default()
        .with_capture_usage(true)
        .with_response_format(ChatResponseFormat::JsonSpec(spec))
}

/// Maps the raw API response to a [`ClientOutput`].
fn map_response(
    response: ChatResponse,
    tools_enabled: bool,
) -> Result<ClientResponse, ClientError> {
    let usage = TokenUsage {
        input: response.usage.prompt_tokens.map(|v| v as u32),
        output: response.usage.completion_tokens.map(|v| v as u32),
    };
    let usage = (usage.input.is_some() || usage.output.is_some()).then_some(usage);
    let provider_model = Some(response.provider_model_iden.model_name.to_string());
    let tool_calls = response.tool_calls();
    if !tool_calls.is_empty() {
        let thought = response.first_text().map(str::to_owned);
        let calls: Vec<ToolCall> = tool_calls
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.call_id.clone(),
                name: tc.fn_name.clone(),
                args: tc.fn_arguments.clone(),
                thought_signatures: tc.thought_signatures.clone(),
            })
            .collect();
        return Ok(ClientResponse::new(
            Provider::Genai,
            ClientOutput::ToolCalls { thought, calls },
        )
        .with_usage(usage)
        .with_provider_model(provider_model));
    }
    if tools_enabled {
        let content = response.into_first_text();
        tracing::warn!(model_output = ?content, "LLM response contained no tool calls");
        return Err(ClientError::MissingToolCalls(content));
    }
    let text = response
        .into_first_text()
        .ok_or(ClientError::EmptyResponse)?;
    Ok(ClientResponse::new(
        Provider::Genai,
        ClientOutput::Output(parse_json_output(&text)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model))
}

#[async_trait]
impl Client for GenaiClient {
    fn provider(&self) -> Provider {
        Provider::Genai
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
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

        let tools_enabled =
            !self.options.tools.is_empty() && self.options.tool_choice != ToolChoice::Disabled;
        validate_tools(Provider::Genai, &self.options.tools)?;
        if self.options.tool_choice == ToolChoice::Required {
            return Err(ClientError::UnsupportedCapability {
                provider: Provider::Genai,
                capability: "required tool choice is not exposed by the genai adapter".into(),
            });
        }
        let chat_messages = build_genai_messages(
            messages,
            self.options.preamble.as_deref(),
            &self.model_name,
            self.options.thinking,
        );
        let request = build_request(chat_messages, &self.options.tools);

        let chat_options = if !tools_enabled {
            let schema = self
                .options
                .output_schema
                .clone()
                .unwrap_or_else(|| Value::Object(Default::default()));
            build_chat_options(schema)
        } else {
            ChatOptions::default().with_capture_usage(true)
        };

        let response = self
            .client
            .exec_chat(&self.model_name, request, Some(&chat_options))
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?;

        map_response(response, tools_enabled)
    }

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        let response = self
            .client
            .embed(&self.model_name, &request.input, None as Option<&EmbedOptions>)
            .await
            .map_err(|e| ClientError::Llm(e.to_string()))?;
        let values = response
            .first_vector()
            .cloned()
            .unwrap_or_default();
        Ok(EmbedResponse { values })
    }
}

/// Creates a `genai`-backed client returning a type-erased [`Box<dyn Client>`].
pub fn create_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    Ok(Box::new(GenaiClient {
        client: build_http_client(url),
        model_name: url.model.clone(),
        options,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            args,
            thought_signatures: None,
        }
    }

    /// With a single user message in history, exactly one chat message is produced.
    #[test]
    fn build_messages_user_only() {
        let history = vec![Message::user(r#"{"text":"hi"}"#)];
        let msgs = build_genai_messages(&history, None, "qwen3:8b", false);
        assert_eq!(msgs.len(), 1);
    }

    /// A preamble becomes the first system message.
    #[test]
    fn build_messages_preamble_prepended() {
        let history = vec![Message::user(r#"{"text":"hi"}"#)];
        let msgs = build_genai_messages(&history, Some("You are helpful."), "qwen3:8b", false);
        assert_eq!(msgs.len(), 2);
        let debug = format!("{msgs:?}");
        assert!(debug.contains("You are helpful."));
    }

    /// History turns appear between the preamble and the user message.
    #[test]
    fn build_messages_history_in_order() {
        let history = vec![
            Message::user("prev question"),
            Message::assistant("prev answer"),
            Message::user("next question"),
        ];
        let msgs = build_genai_messages(&history, Some("sys"), "qwen3:8b", false);
        assert_eq!(msgs.len(), 4);
        let debug = format!("{msgs:?}");
        assert!(debug.contains("prev question"));
        assert!(debug.contains("prev answer"));
    }

    /// A Tool-role history entry is mapped with its call_id into the message list.
    #[test]
    fn build_messages_tool_role_included() {
        let history = vec![Message {
            role: Role::Tool {
                call_id: "call-42".into(),
            },
            content: r#"{"temp":22}"#.into(),
            usage: None,
        }];
        let msgs = build_genai_messages(&history, None, "qwen3:8b", false);
        assert_eq!(msgs.len(), 1);
        let debug = format!("{msgs:?}");
        assert!(debug.contains("call-42"));
    }

    /// Full tool exchange: messages match history length exactly.
    #[test]
    fn build_messages_continue_after_tool_result() {
        let history = vec![
            Message::user(r#"{"goal":"ship feature","known_context":[]}"#),
            Message {
                role: Role::AssistantToolCalls {
                    calls: vec![make_call("call-1", "record_fact", json!({ "fact": "x" }))],
                },
                content: "reasoning".into(),
                usage: None,
            },
            Message {
                role: Role::Tool {
                    call_id: "call-1".into(),
                },
                content: "recorded".into(),
                usage: None,
            },
        ];
        let msgs = build_genai_messages(&history, None, "qwen3:8b", false);
        assert_eq!(msgs.len(), 3);
        let debug = format!("{msgs:?}");
        assert!(debug.contains("call-1"));
    }

    /// Qwen3 prepends `/no_think` to the first user message when thinking is off.
    #[test]
    fn build_messages_no_think_prefix_for_qwen3() {
        let history = vec![Message::user("do the thing")];
        let msgs = build_genai_messages(&history, None, "qwen3:8b", false);
        let debug = format!("{msgs:?}");
        assert!(debug.contains("/no_think"));
    }

    /// Non-qwen3 models do not prepend `/no_think`.
    #[test]
    fn build_messages_no_prefix_for_gemini() {
        let history = vec![Message::user("do the thing")];
        let msgs = build_genai_messages(&history, None, "gemini-2.5-flash-lite", false);
        let debug = format!("{msgs:?}");
        assert!(!debug.contains("/no_think"));
    }
}
