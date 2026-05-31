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
use crate::commons::{Agent, AgentConfig, ExitToolMode};
use crate::context::Context;
use crate::tools::ToolOutput;

#[derive(Clone)]
enum ResponseMode {
    Output(Value),
    ToolCalls(Vec<ToolCall>),
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
            ResponseMode::ToolCalls(calls) => Ok(ClientResponse::new(
                Provider::OpenAi,
                ClientOutput::ToolCalls {
                    thought: None,
                    calls: calls.clone(),
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

    fn configure() -> AgentConfig {
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

    fn configure() -> AgentConfig {
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

    fn configure() -> AgentConfig {
        AgentConfig::new("Use tools before answering.", "openai://test-model")
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct ExitToolAgentInput {
    query: String,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct ExitToolAgentOutput {
    result: String,
}

impl Agent for ExitToolAgentInput {
    type Output = ExitToolAgentOutput;

    fn configure() -> AgentConfig {
        AgentConfig::new("Answer concisely.", "gemini://gemini-2.5-pro")
            .with_exit_tool_mode(ExitToolMode::Auto)
    }
}

impl Flow for ExitToolAgentInput {
    type Output = ExitToolAgentOutput;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder.agent::<ExitToolAgentInput>()
    }
}

impl Flow for ToolAgentInput {
    type Output = ToolAgentOutput;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<ToolAgentInput>()
            .tool_with::<ToolAgentInput, LookupInput, LookupOutput>()
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
    assert_eq!(options.tool_choice, crate::clients::ToolChoice::Auto);
    assert_eq!(
        options.tools.len(),
        1,
        "should have lookup tool"
    );

    let lookup = options
        .tools
        .iter()
        .find(|t| t.name == "lookup")
        .expect("lookup tool should be present");
    assert!(lookup.parameters.is_object());
}

/// `default_turn_budget_message` uses XML format for Anthropic/Gemini and plain
/// imperative text for Ollama/OpenAI model URLs.
#[test]
fn default_turn_budget_message_is_provider_specific() {
    let anthropic_msg = default_turn_budget_message("anthropic://claude-opus-4", None);
    let gemini_msg = default_turn_budget_message("gemini:///gemini-2.5-pro", None);
    let ollama_msg = default_turn_budget_message("ollama://qwen3-coder:30b", None);
    let openai_msg = default_turn_budget_message("openai://gpt-4o", None);

    assert!(anthropic_msg.contains("<system-reminder>"), "anthropic should use XML");
    assert!(gemini_msg.contains("<system-reminder>"), "gemini should use XML");
    assert!(!ollama_msg.contains('<'), "ollama should use plain text");
    assert!(!openai_msg.contains('<'), "openai should use plain text");

    assert!(!anthropic_msg.contains("<tool>"), "should not name a specific tool");
    assert!(anthropic_msg.contains("output format"), "should defer to the output format constraint");
    assert!(!ollama_msg.contains("call the `"), "should not name a specific tool");
    assert!(ollama_msg.contains("output format"), "should defer to the output format constraint");

    let exit_msg_gemini = default_turn_budget_message("gemini://gemini-2.5-pro", Some("MyOutput"));
    let exit_msg_openai = default_turn_budget_message("openai://gpt-4o", Some("MyOutput"));
    assert!(exit_msg_gemini.contains("MyOutput"), "exit tool reminder should name the tool");
    assert!(exit_msg_gemini.contains("<system-reminder>"), "gemini exit reminder should use XML");
    assert!(exit_msg_openai.contains("MyOutput"), "exit tool reminder should name the tool");
    assert!(!exit_msg_openai.contains('<'), "openai exit reminder should use plain text");
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

/// `maybe_inject_turn_budget_message` appends a separate reminder message on the
/// final allowed turn and leaves earlier messages unchanged.
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
        make_environment: |_| None,
        input_schema: serde_json::json!({}),
        model: "ollama://qwen3:8b".into(),
        exit: exit_id,
        output_schema: serde_json::json!({}),
        tool_lookup,
        keep_alive: false,
        turn_budget: Some(2),
        turn_budget_message: None,
        exit_tool_name: None,
    };

    let mut history = FlowHistory::new();
    history.push(session_id, "test_agent", Message::assistant("thinking..."));

    let mut msgs_first: Vec<Message> = vec![Message::user("start")];
    maybe_inject_turn_budget_message(&node, "test_agent", session_id, &history, &mut msgs_first);
    assert_eq!(msgs_first.len(), 2, "reminder should be a new message");
    assert_eq!(msgs_first[0].content, "start", "original content must be preserved");
    assert!(matches!(msgs_first[1].role, Role::User), "reminder should be a user message");
    assert!(
        msgs_first[1].content.contains("FINAL TURN") || msgs_first[1].content.contains("TURN LIMIT"),
        "reminder should signal final turn"
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

/// `maybe_inject_turn_budget_message` never rewrites tool payloads when the
/// prior outbound message is a tool result.
#[test]
fn maybe_inject_preserves_tool_payloads() {
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
        make_environment: |_| None,
        input_schema: serde_json::json!({}),
        model: "gemini://gemini-2.5-pro".into(),
        exit: exit_id,
        output_schema: serde_json::json!({}),
        tool_lookup,
        keep_alive: false,
        turn_budget: Some(2),
        turn_budget_message: None,
        exit_tool_name: None,
    };

    let mut history = FlowHistory::new();
    history.push(
        session_id,
        "test_agent",
        Message {
            role: Role::AssistantToolCalls {
                calls: vec![ToolCall {
                    id: "call-1".into(),
                    name: "lookup".into(),
                    args: serde_json::json!({"query": "rust"}),
                    thought_signatures: None,
                }],
            },
            content: String::new(),
            attachments: Vec::new(),
            usage: None,
        },
    );
    history.push(
        session_id,
        "test_agent",
        Message::tool_output("call-1".into(), r#"{"result":"ok"}"#),
    );

    let mut session_msgs = history.for_session(session_id);
    maybe_inject_turn_budget_message(&node, "test_agent", session_id, &history, &mut session_msgs);

    assert_eq!(session_msgs.len(), 3, "reminder should be appended after the tool result");
    assert!(matches!(session_msgs[1].role, Role::Tool { .. }), "tool message must stay a tool result");
    assert_eq!(session_msgs[1].content, r#"{"result":"ok"}"#, "tool payload must be unchanged");
    assert!(matches!(session_msgs[2].role, Role::User), "reminder should be appended as a user message");
    assert!(
        session_msgs[2].content.contains("<system-reminder>"),
        "gemini reminders should keep XML wrapping"
    );
}

/// An agent with `exit_tool` configured sends a synthetic submit tool in the
/// tool list, forces `Required` tool choice, and omits structured-text output schema.
#[tokio::test]
async fn exit_tool_injects_submit_tool_in_options() {
    let exit_args = json!({ "result": "done" });
    let factory = CapturingFactory::new(ResponseMode::ToolCalls(vec![ToolCall {
        id: "c1".into(),
        name: "ExitToolAgentOutput".into(),
        args: exit_args.clone(),
        thought_signatures: None,
    }]));
    let mut runtime = crate::flows::runtime::FlowRuntime::new(ExitToolAgentInput {
        query: "test".into(),
    })
    .expect("runtime should build")
    .with_factory(factory.clone());

    runtime.next(Context::default()).await.expect("init step");
    runtime.next(Context::default()).await.expect("dispatch step");

    let captured = factory.captured();
    assert_eq!(captured.len(), 1);
    let options = &captured[0];
    assert_eq!(options.tool_choice, crate::clients::ToolChoice::Required, "exit-tool agent must use Required");
    assert!(options.output_schema.is_none(), "exit-tool path must not request structured text output");
    let submit_tool = options.tools.iter().find(|t| t.name == "ExitToolAgentOutput");
    assert!(submit_tool.is_some(), "synthetic submit tool should be present in tool list");
}

/// `maybe_inject_turn_budget_message` fires for exit-tool agents that have no
/// real tools registered, and the reminder names the exit tool.
#[test]
fn maybe_inject_fires_for_exit_tool_agent_without_real_tools() {
    use crate::flows::history::FlowHistory;
    use crate::flows::interner::Interner;
    use std::collections::HashMap;

    let session_id = "s1";
    let mut interner = Interner::new();
    let agent_id = interner.intern("agent");
    let exit_id = interner.intern("ExitOutput");

    let node = AgentInfo {
        id: agent_id,
        tools: vec![],
        make_message: |_, _| Ok(Message::user("hi")),
        preamble: "".into(),
        make_environment: |_| None,
        input_schema: serde_json::json!({}),
        model: "openai://test".into(),
        exit: exit_id,
        output_schema: serde_json::json!({}),
        tool_lookup: HashMap::new(),
        keep_alive: false,
        turn_budget: Some(1),
        turn_budget_message: None,
        exit_tool_name: Some("ExitOutput".into()),
    };

    let history = FlowHistory::new();
    let mut msgs: Vec<Message> = vec![Message::user("start")];
    maybe_inject_turn_budget_message(&node, "agent", session_id, &history, &mut msgs);
    assert_eq!(msgs.len(), 2, "reminder should be injected for exit-tool agent with no real tools");
    assert!(msgs[1].content.contains("ExitOutput"), "reminder should name the exit tool");
}