use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::flow::{
    default_turn_budget_message, maybe_inject_turn_budget_message, wrap_for_provider,
};
use super::nodes::{AgentInfo, ToolInfo};
use super::{Flow, FlowBuilder, FlowError};
use crate::clients::{
    Attachment, Client, ClientError, ClientFactory, ClientOptions, ClientOutput,
    ClientResponse, Message, Provider, Role, ToolCall,
};
use crate::commons::{Agent, AgentConfig};
use crate::context::Context;
use crate::tools::ToolOutput;

#[derive(Clone)]
enum ResponseMode {
    Output(Value),
    ToolCall { name: String, args: Value },
}

#[derive(Clone)]
struct CapturingFactory {
    options: Arc<Mutex<Vec<ClientOptions>>>,
    mode: ResponseMode,
}

struct CapturingClient {
    mode: ResponseMode,
}

impl CapturingFactory {
    fn new(mode: ResponseMode) -> Self {
        Self {
            options: Arc::new(Mutex::new(Vec::new())),
            mode,
        }
    }

    fn captured(&self) -> Vec<ClientOptions> {
        self.options
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[async_trait]
impl Client for CapturingClient {
    fn provider(&self) -> Provider {
        Provider::OpenAi
    }

    async fn execute(&self, _messages: &[Message]) -> Result<ClientResponse, ClientError> {
        match &self.mode {
            ResponseMode::Output(value) => Ok(ClientResponse::new(
                Provider::OpenAi,
                ClientOutput::Output(value.clone()),
            )),
            ResponseMode::ToolCall { name, args } => Ok(ClientResponse::new(
                Provider::OpenAi,
                ClientOutput::ToolCalls {
                    thought: None,
                    calls: vec![ToolCall {
                        id: "call-1".into(),
                        name: name.clone(),
                        args: args.clone(),
                        thought_signatures: None,
                    }],
                },
            )),
        }
    }
}

impl ClientFactory for CapturingFactory {
    fn create(
        &self,
        _model_url: &str,
        options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError> {
        self.options
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(options);
        Ok(Box::new(CapturingClient {
            mode: self.mode.clone(),
        }))
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct PlainAgentInput {
    topic: String,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct PlainAgentOutput {
    answer: String,
}

impl Agent for PlainAgentInput {
    type Output = PlainAgentOutput;

    fn build() -> AgentConfig {
        AgentConfig::new("Answer briefly.", "openai://test-model")
    }
}

impl Flow for PlainAgentInput {
    type Output = PlainAgentOutput;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder.agent::<PlainAgentInput>()
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct MessageAgentInput {
    topic: String,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct MessageAgentOutput {
    answer: String,
}

impl Agent for MessageAgentInput {
    type Output = MessageAgentOutput;

    fn to_message(self, _ctx: &Context) -> Result<Message, FlowError> {
        let mut message = Message::user(format!("Inspect this screenshot about {}", self.topic));
        message.attachments.push(Attachment::Inline {
            mime_type: "image/png".into(),
            data: "aGVsbG8=".into(),
        });
        Ok(message)
    }

    fn build() -> AgentConfig {
        AgentConfig::new("Answer briefly.", "openai://test-model")
    }
}

impl Flow for MessageAgentInput {
    type Output = MessageAgentOutput;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder.agent::<MessageAgentInput>()
    }
}

/// Tool input for agent-with-tool tests.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "lookup")]
struct LookupInput {
    query: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LookupOutput {
    result: String,
}

impl ToolOutput for LookupOutput {}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct ToolAgentInput {
    topic: String,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct ToolAgentOutput {
    answer: String,
}

impl Agent for ToolAgentInput {
    type Output = ToolAgentOutput;

    fn build() -> AgentConfig {
        AgentConfig::new("Use tools before answering.", "openai://test-model")
    }
}

impl Flow for ToolAgentInput {
    type Output = ToolAgentOutput;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<ToolAgentInput>()
            .tool::<ToolAgentInput, LookupInput, LookupOutput>()
            .work(|input: LookupInput, _ctx: Context| async move {
                Ok(LookupOutput {
                    result: input.query,
                })
            })
    }
}

/// Agents without tools stay in structured-output mode.
#[tokio::test]
async fn schema_and_tools_dispatch_without_tools_uses_structured_output() {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" })));
    let mut runtime = crate::flows::runtime::FlowRuntime::new(PlainAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_factory(factory.clone());

    let _ = runtime
        .next(Context::default())
        .await
        .expect("init step should run");
    let _ = runtime
        .next(Context::default())
        .await
        .expect("dispatch step should run");

    let captured = factory.captured();
    assert_eq!(captured.len(), 1);
    let options = &captured[0];
    assert!(options.tools.is_empty());
    assert_eq!(options.tool_choice, crate::clients::ToolChoice::Disabled);
    let expected = serde_json::to_value(schemars::schema_for!(PlainAgentOutput))
        .expect("output schema should serialize");
    assert_eq!(options.output_schema.as_ref(), Some(&expected));
}

/// Agent entry uses `to_message` to populate the first user turn.
#[tokio::test]
async fn agent_entry_uses_custom_to_message() {
    let mut runtime = crate::flows::runtime::FlowRuntime::new(MessageAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_factory(CapturingFactory::new(ResponseMode::Output(
        json!({ "answer": "done" }),
    )));

    let _ = runtime
        .next(Context::default())
        .await
        .expect("entry step should run");

    let entry = runtime
        .inspector()
        .history()
        .entries()
        .iter()
        .find(|entry| entry.message.content == "Inspect this screenshot about rust")
        .expect("custom user message should be recorded");

    assert!(matches!(entry.message.role, Role::User));
    assert!(matches!(
        entry.message.attachments.as_slice(),
        [Attachment::Inline { mime_type, data }]
            if mime_type == "image/png" && data == "aGVsbG8="
    ));
}

/// Agents with tools include the tool definition in options.
#[tokio::test]
async fn schema_and_tools_dispatch_with_tools_includes_lookup() {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" })));
    let mut runtime = crate::flows::runtime::FlowRuntime::new(ToolAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_factory(factory.clone());

    let _ = runtime
        .next(Context::default())
        .await
        .expect("init step should run");
    let _ = runtime
        .next(Context::default())
        .await
        .expect("dispatch step should run");

    let captured = factory.captured();
    assert_eq!(captured.len(), 1);
    let options = &captured[0];
    assert_eq!(options.tool_choice, crate::clients::ToolChoice::Required);
    assert_eq!(
        options.tools.len(),
        2,
        "should have lookup tool and synthetic exit tool"
    );

    let lookup = options
        .tools
        .iter()
        .find(|t| t.name == "lookup")
        .expect("lookup tool should be present");
    assert!(lookup.parameters.is_object());
    let exit_tool = options
        .tools
        .iter()
        .find(|t| t.name == "tool_agent_output")
        .expect("synthetic exit tool should be present");
    assert!(exit_tool.parameters.is_object());
}

/// `default_turn_budget_message` uses XML format for Anthropic/Gemini and plain
/// imperative text for Ollama/OpenAI model URLs.
#[test]
fn default_turn_budget_message_is_provider_specific() {
    let anthropic_msg = default_turn_budget_message("anthropic://claude-opus-4", Some("submit"));
    let gemini_msg = default_turn_budget_message("gemini:///gemini-2.5-pro", Some("submit"));
    let ollama_msg = default_turn_budget_message("ollama://qwen3-coder:30b", Some("submit"));
    let openai_msg = default_turn_budget_message("openai://gpt-4o", Some("submit"));

    assert!(anthropic_msg.contains("<system-reminder>"), "anthropic should use XML");
    assert!(gemini_msg.contains("<system-reminder>"), "gemini should use XML");
    assert!(!ollama_msg.contains('<'), "ollama should use plain text");
    assert!(!openai_msg.contains('<'), "openai should use plain text");

    assert!(anthropic_msg.contains("submit"), "tool name must appear");
    assert!(ollama_msg.contains("submit"), "tool name must appear");
}

/// Custom `turn_budget_message` is wrapped in XML for Anthropic/Gemini and
/// passed through verbatim for other providers.
#[test]
fn wrap_for_provider_wraps_xml_providers_only() {
    let raw = "you must stop now";
    assert!(
        wrap_for_provider("anthropic://claude-opus-4", raw).contains("<system-reminder>"),
        "anthropic should be wrapped"
    );
    assert!(
        wrap_for_provider("gemini://gemini-2.5-pro", raw).contains("<system-reminder>"),
        "gemini should be wrapped"
    );
    assert_eq!(
        wrap_for_provider("openai://gpt-4o", raw),
        raw,
        "openai should be unchanged"
    );
    assert_eq!(
        wrap_for_provider("ollama://qwen3:8b", raw),
        raw,
        "ollama should be unchanged"
    );
}

/// `maybe_inject_turn_budget_message` appends the budget reminder as the last
/// message when completed turns + 1 equals the budget, and leaves msgs unchanged
/// otherwise.
#[test]
fn maybe_inject_injects_on_final_turn_only() {
    use crate::flows::history::FlowHistory;
    use crate::flows::interner::Interner;
    use std::collections::HashMap;

    let session_id = "s1";

    let mut interner = Interner::new();
    let agent_id = interner.intern("test_agent");
    let exit_id = interner.intern("FinalAnswer");
    let entry_id = interner.intern("test_agent::final_answer");

    let tool_def = crate::tools::ToolDefinition {
        name: "final_answer".into(),
        description: "submit".into(),
        parameters: serde_json::json!({"type": "object"}),
    };
    let mut tool_lookup = HashMap::new();
    tool_lookup.insert("final_answer".to_string(), (entry_id, exit_id));

    let node = AgentInfo {
        id: agent_id,
        tools: vec![ToolInfo {
            definition: tool_def,
            exit_id,
            to_message: Box::new(|v| Ok(Message::tool_output("x".into(), v.to_string()))),
        }],
        make_message: |_, _| Ok(Message::user("hi")),
        preamble: "".into(),
        input_schema: serde_json::json!({}),
        model: "ollama://qwen3:8b".into(),
        exit: exit_id,
        output_schema: serde_json::json!({}),
        tool_lookup,
        keep_alive: false,
        turn_budget: Some(2),
        turn_budget_message: None,
    };

    let mut history = FlowHistory::new();
    history.push(session_id, "test_agent", Message::assistant("thinking..."));

    let mut msgs_first: Vec<Message> = vec![Message::user("start")];
    maybe_inject_turn_budget_message(&node, "test_agent", session_id, &history, &mut msgs_first);
    assert_eq!(msgs_first.len(), 1, "reminder is embedded, not a new message");
    assert!(
        msgs_first[0].content.starts_with("start"),
        "original content must be preserved"
    );
    assert!(
        msgs_first[0].content.contains("final_answer"),
        "reminder should name the exit tool"
    );

    let history_empty = FlowHistory::new();
    let mut msgs_early: Vec<Message> = vec![Message::user("start")];
    maybe_inject_turn_budget_message(
        &node,
        "test_agent",
        session_id,
        &history_empty,
        &mut msgs_early,
    );
    assert_eq!(msgs_early.len(), 1, "no injection when turns remain");
    assert_eq!(
        msgs_early[0].content,
        "start",
        "content must be unmodified when no injection"
    );
}