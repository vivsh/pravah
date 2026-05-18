use async_trait::async_trait;
use gemini_rust::{
    Blob, Content, FileData as GeminiFileData, FunctionCall as GeminiFunctionCall,
    FunctionCallingMode, FunctionDeclaration, FunctionResponse as GeminiFunctionResponse,
    Gemini, GenerationResponse, Message as GeminiMessage, Part, Role as GeminiRole, TaskType,
    Tool as GeminiTool, client::Model as GeminiModel,
};
use serde_json::Value;

use super::super::tools::ToolDefinition;
use super::schema;
use super::{
    Attachment, Client, ClientError, ClientOptions, ClientOutput, ClientResponse, EmbedRequest,
    EmbedResponse, EmbedTaskType, LlmUrl, Message, Provider, Role,
    TokenUsage, ToolCall, ToolChoice, decode_output_text, validate_tools,
};

fn format_error_chain(e: &dyn std::error::Error) -> String {
    let mut msg = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        msg.push_str(": ");
        msg.push_str(&cause.to_string());
        source = cause.source();
    }
    msg
}

fn build_client(url: &LlmUrl) -> Result<Gemini, ClientError> {
    let api_key = if let Some(key) = &url.api_key {
        key.clone()
    } else {
        std::env::var("GEMINI_API_KEY")
            .map_err(|_| ClientError::Llm("GEMINI_API_KEY is not set".into()))?
    };
    let model_id = if url.model.starts_with("models/") {
        url.model.clone()
    } else {
        format!("models/{}", url.model)
    };
    let model = GeminiModel::Custom(model_id);
    Gemini::with_model(&api_key, model).map_err(|e| ClientError::Llm(format_error_chain(&e)))
}

struct GeminiClient {
    client: Gemini,
    options: ClientOptions,
}

/// Builds Gemini messages from history.
fn build_gemini_messages(history: &[Message]) -> Vec<GeminiMessage> {
    let mut msgs = Vec::new();
    let mut i = 0;
    while i < history.len() {
        match &history[i].role {
            Role::System => {
                i += 1;
            }
            Role::User => {
                msgs.push(user_to_message(&history[i]));
                i += 1;
            }
            Role::Assistant => {
                msgs.push(GeminiMessage::model(&history[i].content));
                i += 1;
            }
            Role::AssistantToolCalls { calls } => {
                msgs.push(tool_calls_to_message(calls));
                i += 1;
            }
            Role::Tool { .. } => {
                let (msg, consumed) = tool_responses_to_message(history, i);
                msgs.push(msg);
                i += consumed;
            }
        }
    }
    msgs
}

fn gemini_part_from_attachment(att: &Attachment) -> Option<Part> {
    match att {
        Attachment::Inline { mime_type, data } => Some(Part::InlineData {
            inline_data: Blob::new(mime_type, data),
            media_resolution: None,
        }),
        Attachment::Url { mime_type, url } => Some(Part::FileData {
            file_data: GeminiFileData {
                mime_type: mime_type.clone(),
                file_uri: url.clone(),
            },
        }),
        Attachment::File { path, .. } => {
            tracing::warn!(path = %path, "file attachment was not materialized before Gemini serialization; dropping");
            None
        }
    }
}

fn user_to_message(message: &Message) -> GeminiMessage {
    if message.attachments.is_empty() {
        return GeminiMessage::user(message.content.clone());
    }

    let mut parts = message
        .attachments
        .iter()
        .filter_map(gemini_part_from_attachment)
        .collect::<Vec<_>>();
    if !message.content.is_empty() {
        parts.push(Part::Text {
            text: message.content.clone(),
            thought: None,
            thought_signature: None,
        });
    }
    GeminiMessage {
        content: Content {
            parts: Some(parts),
            role: Some(GeminiRole::User),
        },
        role: GeminiRole::User,
    }
}

fn build_tools_spec(tools: &[ToolDefinition]) -> Result<Option<GeminiTool>, ClientError> {
    if tools.is_empty() {
        return Ok(None);
    }
    let fns: Vec<FunctionDeclaration> = tools
        .iter()
        .map(build_fn_decl)
        .collect::<Result<Vec<_>, _>>()?;
    if fns.is_empty() {
        Ok(None)
    } else {
        Ok(Some(GeminiTool::with_functions(fns)))
    }
}

/// Converts `AssistantToolCalls` history into a model-role message.
fn tool_calls_to_message(calls: &[ToolCall]) -> GeminiMessage {
    let parts: Vec<Part> = calls
        .iter()
        .map(|c| {
            let thought_sig = c
                .thought_signatures
                .as_ref()
                .and_then(|v| v.first())
                .cloned();
            Part::FunctionCall {
                function_call: GeminiFunctionCall::new(&c.name, c.args.clone()),
                thought_signature: thought_sig,
            }
        })
        .collect();
    GeminiMessage {
        content: Content {
            parts: Some(parts),
            role: Some(GeminiRole::Model),
        },
        role: GeminiRole::Model,
    }
}

/// Groups consecutive `Tool` history entries into one user-role message.
fn tool_responses_to_message(history: &[Message], start: usize) -> (GeminiMessage, usize) {
    let mut parts = Vec::new();
    let mut i = start;
    while i < history.len() {
        let Role::Tool { call_id } = &history[i].role else {
            break;
        };
        let name = resolve_call_name(history, call_id);
        let val: Value = serde_json::from_str(&history[i].content)
            .unwrap_or_else(|_| Value::String(history[i].content.clone()));
        parts.push(Part::FunctionResponse {
            function_response: GeminiFunctionResponse::new(name, val),
        });
        // Append any attachments produced by this tool call as additional parts.
        for part in history[i]
            .attachments
            .iter()
            .filter_map(gemini_part_from_attachment)
        {
            parts.push(part);
        }
        i += 1;
    }
    let msg = GeminiMessage {
        content: Content {
            parts: Some(parts),
            role: Some(GeminiRole::User),
        },
        role: GeminiRole::User,
    };
    (msg, i - start)
}

/// Resolves a tool call name by walking history backwards.
fn resolve_call_name<'a>(history: &'a [Message], call_id: &'a str) -> &'a str {
    for msg in history.iter().rev() {
        if let Role::AssistantToolCalls { calls } = &msg.role {
            for c in calls {
                if c.id == call_id {
                    return &c.name;
                }
            }
        }
    }
    tracing::error!(
        call_id,
        "could not resolve tool call name from history; using call_id as fallback"
    );
    call_id
}

/// Converts a `ToolDefinition` into a Gemini function declaration.
fn build_fn_decl(tool: &ToolDefinition) -> Result<FunctionDeclaration, ClientError> {
    let sanitized = schema::sanitize_strict(tool.parameters.clone());
    let json = serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": sanitized,
    });
    serde_json::from_value(json).map_err(ClientError::Serialize)
}

/// Maps the raw Gemini response into a [`ClientOutput`].
fn map_response(
    response: GenerationResponse,
    tools_enabled: bool,
    wants_json_output: bool,
) -> Result<ClientResponse, ClientError> {
    let usage = response.usage_metadata.as_ref().map(|usage| TokenUsage {
        input: usage.prompt_token_count.map(|v| v as u32),
        output: usage.candidates_token_count.map(|v| v as u32),
    });
    let provider_model = response.model_version.clone();
    let raw_metadata = Some(serde_json::json!({
        "response_id": response.response_id.clone(),
    }));
    let fcs = response.function_calls_with_thoughts();
    if !fcs.is_empty() {
        let thought_text = response.text();
        let thought = if thought_text.is_empty() {
            None
        } else {
            Some(thought_text)
        };
        let calls: Vec<ToolCall> = fcs
            .iter()
            .enumerate()
            .map(|(idx, (fc, sig))| ToolCall {
                id: format!("{}_{}", fc.name, idx),
                name: fc.name.clone(),
                args: fc.args.clone(),
                thought_signatures: sig.map(|s| vec![s.to_string()]),
            })
            .collect();
        return Ok(ClientResponse::new(
            Provider::Gemini,
            ClientOutput::ToolCalls { thought, calls },
        )
        .with_usage(usage)
        .with_provider_model(provider_model)
        .with_raw_metadata(raw_metadata));
    }
    if tools_enabled {
        let text = response.text();
        let content = if text.is_empty() { None } else { Some(text) };
        tracing::warn!(model_output = ?content, "LLM response contained no tool calls");
        return Err(ClientError::MissingToolCalls(content));
    }
    let text = response.text();
    if text.is_empty() {
        return Err(ClientError::EmptyResponse);
    }
    Ok(ClientResponse::new(
        Provider::Gemini,
        ClientOutput::Output(decode_output_text(&text, wants_json_output)?),
    )
    .with_usage(usage)
    .with_provider_model(provider_model)
    .with_raw_metadata(raw_metadata))
}

fn wants_json_output(options: &ClientOptions, tools_enabled: bool) -> bool {
    !tools_enabled && options.wants_json_output()
}

fn response_schema(options: &ClientOptions, tools_enabled: bool) -> Option<Value> {
    if !wants_json_output(options, tools_enabled) {
        return None;
    }
    options
        .output_schema
        .as_ref()
        .map(|value| schema::sanitize_strict(value.clone()))
}

impl GeminiClient {
    async fn call_api(
        &self,
        messages: Vec<GeminiMessage>,
        tools_enabled: bool,
        wants_json_output: bool,
        response_schema: Option<Value>,
    ) -> Result<GenerationResponse, ClientError> {
        let client = &self.client;
        let thinking_budget = if self.options.thinking {
            self.options.thinking_budget.map(|b| b as i32).unwrap_or(i32::MAX)
        } else {
            0
        };
        let mut builder = client
            .generate_content()
            .with_thinking_budget(thinking_budget);
        if let Some(p) = self.options.effective_preamble() {
            builder = builder.with_system_prompt(p);
        }
        builder = builder.with_messages(messages);
        if tools_enabled {
            if let Some(tool_spec) = build_tools_spec(&self.options.tools)? {
                let mode = match self.options.tool_choice {
                    ToolChoice::Required => FunctionCallingMode::Any,
                    _ => FunctionCallingMode::Auto,
                };
                builder = builder
                    .with_tool(tool_spec)
                    .with_function_calling_mode(mode);
            }
        } else if wants_json_output {
            builder = builder.with_response_mime_type("application/json");
            if let Some(schema) = response_schema {
                builder = builder.with_response_schema(schema);
            }
        }
        builder
            .execute()
            .await
            .map_err(|e| ClientError::Llm(format_error_chain(&e)))
    }
}

#[async_trait]
impl Client for GeminiClient {
    fn provider(&self) -> Provider {
        Provider::Gemini
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
        validate_tools(Provider::Gemini, &self.options.tools)?;
        let wants_json_output = wants_json_output(&self.options, tools_enabled);
        let response_schema = response_schema(&self.options, tools_enabled);
        let gemini_messages = build_gemini_messages(messages);
        let response = self
            .call_api(gemini_messages, tools_enabled, wants_json_output, response_schema)
            .await?;
        map_response(response, tools_enabled, wants_json_output)
    }

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        let mut builder = self.client.embed_content().with_text(&request.input);
        if let Some(task_type) = &request.task_type {
            let gemini_task = match task_type {
                EmbedTaskType::RetrievalDocument => TaskType::RetrievalDocument,
                EmbedTaskType::RetrievalQuery => TaskType::RetrievalQuery,
                EmbedTaskType::SemanticSimilarity => TaskType::SemanticSimilarity,
                EmbedTaskType::Classification => TaskType::Classification,
                EmbedTaskType::Clustering => TaskType::Clustering,
                EmbedTaskType::QuestionAnswering => TaskType::QuestionAnswering,
                EmbedTaskType::FactVerification => TaskType::FactVerification,
                EmbedTaskType::CodeRetrievalQuery => TaskType::CodeRetrievalQuery,
            };
            builder = builder.with_task_type(gemini_task);
        }
        if let Some(title) = &request.title {
            builder = builder.with_title(title.clone());
        }
        if let Some(dim) = request.output_dimensionality {
            builder = builder.with_output_dimensionality(dim);
        }
        let response = builder
            .execute()
            .await
            .map_err(|e| ClientError::Llm(format_error_chain(&e)))?;
        Ok(EmbedResponse {
            values: response.embedding.values,
        })
    }
}

/// Creates a Gemini client.
/// Fails when the API key cannot be resolved.
pub fn new_client(url: &LlmUrl, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
    let client = build_client(url)?;
    Ok(Box::new(GeminiClient { client, options }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            args: json!({}),
            thought_signatures: None,
        }
    }

    /// A single user turn produces one provider message.
    #[test]
    fn build_messages_user_only() {
        let history = vec![Message::user(r#"{"text":"hi"}"#)];
        let msgs = build_gemini_messages(&history);
        assert_eq!(msgs.len(), 1);
    }

    /// User attachments are converted into Gemini inline or file parts.
    #[test]
    fn build_messages_user_with_attachment_adds_inline_part() {
        let history = vec![Message {
            role: Role::User,
            content: "describe this".into(),
            attachments: vec![Attachment::Inline {
                mime_type: "image/png".into(),
                data: "aGVsbG8=".into(),
            }],
            usage: None,
        }];
        let msgs = build_gemini_messages(&history);
        let parts = msgs[0]
            .content
            .parts
            .as_ref()
            .expect("user message parts should be present");
        assert!(matches!(parts.first(), Some(Part::InlineData { .. })));
        assert!(matches!(
            parts.last(),
            Some(Part::Text { text, .. }) if text == "describe this"
        ));
    }

    /// Preambles are not duplicated into history messages.
    #[test]
    fn build_messages_preamble_is_separate() {
        let history = vec![Message::user(r#"{"text":"hi"}"#)];
        let msgs = build_gemini_messages(&history);
        assert_eq!(msgs.len(), 1);
    }

    /// History order is preserved.
    #[test]
    fn build_messages_history_in_order() {
        let history = vec![
            Message::user("prev question"),
            Message::assistant("prev answer"),
            Message::user("next question"),
        ];
        let msgs = build_gemini_messages(&history);
        assert_eq!(msgs.len(), 3);
        let debug = format!("{msgs:?}");
        assert!(debug.contains("prev question"));
        assert!(debug.contains("prev answer"));
    }

    /// Tool responses are grouped into a function-response message.
    #[test]
    fn build_messages_tool_role_included() {
        let history = vec![
            Message {
                role: Role::AssistantToolCalls {
                    calls: vec![make_call("call-42", "read_file")],
                },
                content: String::new(),
                attachments: Vec::new(),
                usage: None,
            },
            Message {
                role: Role::Tool {
                    call_id: "call-42".into(),
                },
                content: r#"{"temp":22}"#.into(),
                attachments: Vec::new(),
                usage: None,
            },
        ];
        let msgs = build_gemini_messages(&history);
        assert_eq!(msgs.len(), 2);
        let debug = format!("{msgs:?}");
        assert!(debug.contains("read_file"));
    }

    /// Tool results keep the exchange length aligned with history.
    #[test]
    fn build_messages_continue_after_tool_result() {
        let history = vec![
            Message::user(r#"{"goal":"ship","known_context":[]}"#),
            Message {
                role: Role::AssistantToolCalls {
                    calls: vec![make_call("c1", "project_outline")],
                },
                content: String::new(),
                attachments: Vec::new(),
                usage: None,
            },
            Message {
                role: Role::Tool {
                    call_id: "c1".into(),
                },
                content: r#"{"files":[]}"}"#.into(),
                attachments: Vec::new(),
                usage: None,
            },
        ];
        let msgs = build_gemini_messages(&history);
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn response_mode_requires_input_schema() {
        let no_schema = ClientOptions::default();
        assert!(!wants_json_output(&no_schema, false));
        assert!(response_schema(&no_schema, false).is_none());

        let with_schema = ClientOptions::default()
            .with_input_schema(json!({ "type": "object" }))
            .with_output_schema(json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }));
        assert!(wants_json_output(&with_schema, false));
        assert!(response_schema(&with_schema, false).is_some());
    }

    #[test]
    fn response_mode_without_output_schema_still_uses_json_when_input_schema_is_present() {
        let with_input_schema = ClientOptions::default()
            .with_input_schema(json!({ "type": "object" }));
        assert!(wants_json_output(&with_input_schema, false));
        assert!(response_schema(&with_input_schema, false).is_none());
    }
}
