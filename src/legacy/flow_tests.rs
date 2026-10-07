use crate::clients::Provider;
use std::sync::{Arc, Mutex};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::flow::maybe_inject_turn_budget_message;
use super::nodes::{AgentInfo, ToolInfo};
use super::{Flow, FlowError, FlowStep, Node};
use crate::clients::{
    Attachment, Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmBackend,
    Message, ModelUrl, ProviderFactory, Role, ToolCall,
};
use crate::commons::{Agent, AgentConfig};
use crate::context::Context;
use crate::legacy::FlowRuntime;
use crate::legacy::ToolOutput;
use crate::testing::{ScriptedFactory, mock_tool_call};

#[derive(Debug)]
struct TestHistoryError(&'static str);

impl std::fmt::Display for TestHistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for TestHistoryError {}

struct FailingStore;

impl crate::legacy::store::HistoryStore for FailingStore {
    type Error = TestHistoryError;

    async fn load(
        &self,
        _key: &str,
    ) -> Result<Vec<crate::legacy::history::HistoryEntry>, Self::Error> {
        Err(TestHistoryError("loading not supported by this fixture"))
    }

    async fn record(
        &self,
        _entry: &crate::legacy::history::HistoryEntry,
    ) -> Result<(), Self::Error> {
        Err(TestHistoryError("record failed"))
    }
}

struct InvalidCompactor;

impl crate::legacy::compactor::HistoryCompactor for InvalidCompactor {
    async fn compact(
        &self,
        _session_id: &str,
        _entries: &[&crate::legacy::history::HistoryEntry],
    ) -> crate::legacy::compactor::CompactionResult {
        crate::legacy::compactor::CompactionResult {
            evict_indices: vec![usize::MAX],
            summary: None,
        }
    }
}

#[derive(Clone)]
enum ResponseMode {
    Output(Value),
}

#[derive(Clone)]
struct CapturingFactory {
    options: Arc<Mutex<Vec<ClientOptions>>>,
    mode: ResponseMode,
}

struct CapturingClient {
    mode: ResponseMode,
    url: ModelUrl,
    client_options: ClientOptions,
}

impl CapturingClient {
    fn for_url(mode: ResponseMode, model_url: &str) -> Result<Self, ClientError> {
        let url = ModelUrl::parse(model_url)?;
        Ok(Self {
            mode,
            url,
            client_options: ClientOptions::default(),
        })
    }

    fn with_options(mut self, opts: ClientOptions) -> Self {
        self.client_options = opts;
        self
    }
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

impl LlmBackend for CapturingClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &ClientOptions {
        &self.client_options
    }

    async fn execute(&self, _messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let response = match &self.mode {
            ResponseMode::Output(value) => {
                ClientResponse::new(self.provider(), ClientOutput::Output(value.clone()))
            }
        };
        Ok(response)
    }
}

impl ProviderFactory for CapturingFactory {
    async fn llm(
        &self,
        model_url: &ModelUrl,
        options: ClientOptions,
    ) -> Result<Client, ClientError> {
        self.options
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(options.clone());
        let url = model_url.clone();
        Ok(Client::from_backend(CapturingClient {
            mode: self.mode.clone(),
            url,
            client_options: options,
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
        AgentConfig::new("Answer briefly.", "test:///test-model")
    }
}

impl Flow for PlainAgentInput {
    type Output = PlainAgentOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct ProviderConfigAgentInput {
    topic: String,
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct ProviderConfigAgentOutput {
    answer: String,
}

impl Agent for ProviderConfigAgentInput {
    type Output = ProviderConfigAgentOutput;

    fn configure() -> AgentConfig {
        AgentConfig::new("Answer with provider settings.", "test:///gemini-2.5-flash")
            .with_provider_config(json!({
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }))
    }
}

impl Flow for ProviderConfigAgentInput {
    type Output = ProviderConfigAgentOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
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
        AgentConfig::new("Answer briefly.", "test:///test-model")
    }
}

impl Flow for MessageAgentInput {
    type Output = MessageAgentOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
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
        AgentConfig::new("Use tools before answering.", "test:///test-model")
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
        AgentConfig::new("Answer concisely.", "test:///gemini-2.5-pro")
    }
}

impl Flow for ExitToolAgentInput {
    type Output = ExitToolAgentOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
    }
}

impl Flow for ToolAgentInput {
    type Output = ToolAgentOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(|toolbox| {
            toolbox.tool_handler(|input: LookupInput, _ctx: Context| async move {
                Ok(LookupOutput {
                    result: input.query,
                })
            })
        })
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "explorer_agent")]
struct ExplorerInput {
    question: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct ExplorerOutput {
    answer: String,
}

impl ToolOutput for ExplorerOutput {}

impl Agent for ExplorerInput {
    type Output = ExplorerOutput;

    fn configure() -> AgentConfig {
        AgentConfig::new("Explore and answer.", "test:///child")
    }
}

impl Flow for ExplorerInput {
    type Output = ExplorerOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct ParentToolFlowInput {
    request: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct ParentToolFlowOutput {
    final_answer: String,
}

impl Agent for ParentToolFlowInput {
    type Output = ParentToolFlowOutput;

    fn configure() -> AgentConfig {
        AgentConfig::new("Use the explorer tool.", "test:///parent")
    }
}

impl Flow for ParentToolFlowInput {
    type Output = ParentToolFlowOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(|toolbox| toolbox.tool_flow::<ExplorerInput>())
    }
}

async fn run_test_flow_to_done<I: Flow>(
    mut runtime: FlowRuntime<I>,
) -> Result<I::Output, FlowError> {
    for _ in 0..80 {
        match runtime.next(Context::default()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(output) => return Ok(output),
            FlowStep::Suspend(_) => {
                return Err(FlowError::Internal {
                    handler: "run_test_flow_to_done",
                    detail: "unexpected suspension".into(),
                });
            }
        }
    }
    Err(FlowError::Internal {
        handler: "run_test_flow_to_done",
        detail: "flow did not finish within 80 steps".into(),
    })
}

async fn run_test_flow_to_err<I: Flow>(runtime: FlowRuntime<I>) -> FlowError {
    match run_test_flow_to_done(runtime).await {
        Ok(_) => panic!("flow should fail"),
        Err(err) => err,
    }
}

#[tokio::test]
async fn tool_flow_recovers_double_encoded_output_string() -> Result<(), crate::GraphError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "c1",
            "explorer_agent",
            json!({ "question": "where is it?" }),
        )])
        .then_output(Value::String(r#"{"answer":"found"}"#.into()))
        .then_output(json!({ "final_answer": "done" }));
    let spy = factory.clone();
    let runtime = FlowRuntime::new(ParentToolFlowInput {
        request: "look".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory)?);

    let output = run_test_flow_to_done(runtime)
        .await
        .expect("double-encoded child output should be recovered");

    assert_eq!(output.final_answer, "done");
    let calls = spy.calls();
    assert_eq!(calls.len(), 3);
    let parent_after_tool = &calls[2].1;
    assert!(parent_after_tool.iter().any(|message| {
        matches!(&message.role, Role::Tool { call_id } if call_id == "c1")
            && message.content == r#"{"answer":"found"}"#
    }));
    Ok(())
}

#[tokio::test]
async fn tool_flow_rejects_malformed_double_encoded_output_string() -> Result<(), crate::GraphError>
{
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "c1",
            "explorer_agent",
            json!({ "question": "where is it?" }),
        )])
        .then_output(Value::String(r#"{"missing":"answer"}"#.into()));
    let runtime = FlowRuntime::new(ParentToolFlowInput {
        request: "look".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory)?);

    let err = run_test_flow_to_err(runtime).await;

    match err {
        FlowError::ToolOutput {
            tool,
            expected,
            reason,
            raw,
        } => {
            assert_eq!(tool, "explorer_agent");
            assert_eq!(expected, "ExplorerOutput");
            assert!(reason.contains("missing field"));
            assert!(raw.contains("missing"));
        }
        other => panic!("expected ToolOutput error, got {other}"),
    }
    Ok(())
}

/// Agents without tools stay in structured-output mode.
#[tokio::test]
async fn schema_and_tools_dispatch_without_tools_uses_structured_output()
-> Result<(), crate::GraphError> {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" })));
    let mut runtime = crate::legacy::runtime::FlowRuntime::new(PlainAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory.clone())?);

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
    assert!(
        matches!(&options.response_format, crate::clients::ResponseFormat::JsonSchema { schema } if schema == &expected)
    );
    Ok(())
}

/// `run_until` reports max-turn exhaustion through `RunOutcome`, matching other limits.
#[tokio::test]
async fn run_until_max_turns_returns_limit_outcome() {
    use crate::legacy::runtime::{FlowRuntime, LimitKind, RunLimits, RunOutcome};

    let mut runtime = FlowRuntime::new(PlainAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build");
    let outcome = runtime
        .run_until(Context::default(), RunLimits::new().max_turns(0))
        .await
        .expect("max_turns should be a normal run outcome");

    assert!(matches!(
        outcome,
        RunOutcome::LimitExceeded(LimitKind::MaxTurns)
    ));
}

/// Store record failures from `next` are surfaced as flow history errors.
#[tokio::test]
async fn next_propagates_history_store_errors() {
    use crate::legacy::runtime::FlowRuntime;

    let mut runtime = FlowRuntime::new(PlainAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_store(FailingStore);
    let err = match runtime.next(Context::default()).await {
        Ok(_) => panic!("record failure should propagate"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        FlowError::History {
            operation: "record",
            ..
        }
    ));
}

#[tokio::test]
async fn agent_provider_config_reaches_client_options() -> Result<(), crate::GraphError> {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" })));
    let mut runtime = FlowRuntime::new(ProviderConfigAgentInput {
        topic: "safety".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory.clone())?);

    assert!(matches!(
        runtime.next(Context::default()).await.unwrap(),
        FlowStep::Continue
    ));
    assert!(matches!(
        runtime.next(Context::default()).await.unwrap(),
        FlowStep::Done(_)
    ));

    let options = factory.captured();
    let provider_config = options
        .first()
        .and_then(|opts| opts.provider_config.as_ref())
        .expect("provider config should reach client options");
    assert_eq!(
        provider_config["safety_settings"][0]["threshold"],
        "BLOCK_NONE"
    );
    Ok(())
}

/// Invalid compaction decisions from `next` are surfaced as flow history errors.
#[tokio::test]
async fn next_propagates_compaction_errors() {
    use crate::legacy::runtime::FlowRuntime;

    let mut runtime = FlowRuntime::new(PlainAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_compactor(InvalidCompactor);
    let err = match runtime.next(Context::default()).await {
        Ok(_) => panic!("invalid compaction index should propagate"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        FlowError::History {
            operation: "compaction",
            ..
        }
    ));
}

/// Agent entry uses `to_message` to populate the first user turn.
#[tokio::test]
async fn agent_entry_uses_custom_to_message() -> Result<(), crate::GraphError> {
    let mut runtime = crate::legacy::runtime::FlowRuntime::new(MessageAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(CapturingFactory::new(
        ResponseMode::Output(json!({ "answer": "done" })),
    ))?);

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
    Ok(())
}

/// Agents with tools include the tool definition in options.
#[tokio::test]
async fn schema_and_tools_dispatch_with_tools_includes_lookup() -> Result<(), crate::GraphError> {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "done" })));
    let mut runtime = crate::legacy::runtime::FlowRuntime::new(ToolAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory.clone())?);

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
    assert_eq!(options.tools.len(), 1, "should have lookup tool");

    let lookup = options
        .tools
        .iter()
        .find(|t| t.name == "lookup")
        .expect("lookup tool should be present");
    assert!(lookup.parameters.is_object());
    Ok(())
}

/// Conclusion guidance preserves provider formatting without naming Rath's internal tool.
#[test]
fn default_turn_budget_message_is_provider_specific() {
    for provider in [
        Provider::Anthropic,
        Provider::Gemini,
        Provider::Ollama,
        Provider::OpenAi,
    ] {
        let message = crate::clients::conclusion_message(&provider);
        assert_eq!(
            message.contains("<system-reminder>"),
            matches!(provider, Provider::Anthropic | Provider::Gemini)
        );
        assert!(message.contains("output format"));
        assert!(!message.contains("__rath_final_output"));
    }
}

/// Custom reminders use provider formatting without changing their text.
#[test]
fn wrap_for_provider_wraps_xml_providers_only() {
    for provider in [
        Provider::Anthropic,
        Provider::Gemini,
        Provider::Ollama,
        Provider::OpenAi,
    ] {
        let message = crate::clients::wrap_system_reminder(&provider, "stop now");
        assert!(message.contains("stop now"));
        assert_eq!(
            message.contains("<system-reminder>"),
            matches!(provider, Provider::Anthropic | Provider::Gemini)
        );
    }
}

/// `maybe_inject_turn_budget_message` appends a separate reminder message on the
/// final allowed turn and leaves earlier messages unchanged.
#[test]
fn maybe_inject_injects_on_final_turn_only() -> Result<(), ClientError> {
    use crate::legacy::history::MessageHistory;
    use crate::legacy::interner::Interner;
    use std::collections::HashMap;

    let session_id = "s1";

    let mut interner = Interner::new();
    let agent_id = interner.intern("test_agent");
    let exit_id = interner.intern("FinalAnswer");
    let entry_id = interner.intern("test_agent::final_answer");

    let tool_def = crate::tools::ToolDefinition::new(
        "final_answer".into(),
        "submit".into(),
        serde_json::json!({"type": "object"}),
    );
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
        model: "test:///qwen3:8b".into(),
        exit: exit_id,
        output_schema: serde_json::json!({}),
        tool_lookup,
        keep_alive: false,
        turn_budget: Some(2),
        turn_budget_message: None,
        provider_config: None,
    };

    let mut history = MessageHistory::new();
    history.push(session_id, "test_agent", Message::assistant("thinking..."));

    let opts = ClientOptions::default()
        .with_tools(node.tools.iter().map(|t| t.definition.clone()).collect());
    let client = Client::from_backend(
        CapturingClient::for_url(ResponseMode::Output(json!({})), "ollama:///qwen3:8b")?
            .with_options(opts),
    );
    let mut msgs_first: Vec<Message> = vec![Message::user("start")];
    maybe_inject_turn_budget_message(
        &client,
        &node,
        "test_agent",
        session_id,
        &history,
        &mut msgs_first,
        0,
    );
    assert_eq!(msgs_first.len(), 2, "reminder should be a new message");
    assert_eq!(
        msgs_first[0].content, "start",
        "original content must be preserved"
    );
    assert!(
        matches!(msgs_first[1].role, Role::User),
        "reminder should be a user message"
    );
    assert!(
        msgs_first[1].content.contains("FINAL TURN")
            || msgs_first[1].content.contains("TURN LIMIT"),
        "reminder should signal final turn"
    );

    let history_empty = MessageHistory::new();
    let mut msgs_early: Vec<Message> = vec![Message::user("start")];
    maybe_inject_turn_budget_message(
        &client,
        &node,
        "test_agent",
        session_id,
        &history_empty,
        &mut msgs_early,
        0,
    );
    assert_eq!(msgs_early.len(), 1, "no injection when turns remain");
    assert_eq!(
        msgs_early[0].content, "start",
        "content must be unmodified when no injection"
    );
    Ok(())
}

/// `maybe_inject_turn_budget_message` never rewrites tool payloads when the
/// prior outbound message is a tool result.
#[test]
fn maybe_inject_preserves_tool_payloads() -> Result<(), ClientError> {
    use crate::legacy::history::MessageHistory;
    use crate::legacy::interner::Interner;
    use std::collections::HashMap;

    let session_id = "s1";

    let mut interner = Interner::new();
    let agent_id = interner.intern("test_agent");
    let exit_id = interner.intern("FinalAnswer");
    let entry_id = interner.intern("test_agent::final_answer");

    let tool_def = crate::tools::ToolDefinition::new(
        "final_answer".into(),
        "submit".into(),
        serde_json::json!({"type": "object"}),
    );
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
        model: "test:///gemini-2.5-pro".into(),
        exit: exit_id,
        output_schema: serde_json::json!({}),
        tool_lookup,
        keep_alive: false,
        turn_budget: Some(2),
        turn_budget_message: None,
        provider_config: None,
    };

    let mut history = MessageHistory::new();
    history.push(
        session_id,
        "test_agent",
        Message::new(
            Role::AssistantToolCalls {
                calls: vec![ToolCall::new(
                    "call-1".into(),
                    "lookup".into(),
                    serde_json::json!({"query": "rust"}),
                )],
            },
            String::new(),
        ),
    );
    history.push(
        session_id,
        "test_agent",
        Message::tool_output("call-1".into(), r#"{"result":"ok"}"#),
    );

    let opts = ClientOptions::default()
        .with_tools(node.tools.iter().map(|t| t.definition.clone()).collect());
    let client = Client::from_backend(
        CapturingClient::for_url(ResponseMode::Output(json!({})), "gemini:///gemini-2.5-pro")?
            .with_options(opts),
    );
    let mut session_msgs = history.for_session(session_id);
    maybe_inject_turn_budget_message(
        &client,
        &node,
        "test_agent",
        session_id,
        &history,
        &mut session_msgs,
        0,
    );

    assert_eq!(
        session_msgs.len(),
        3,
        "reminder should be appended after the tool result"
    );
    assert!(
        matches!(session_msgs[1].role, Role::Tool { .. }),
        "tool message must stay a tool result"
    );
    assert_eq!(
        session_msgs[1].content, r#"{"result":"ok"}"#,
        "tool payload must be unchanged"
    );
    assert!(
        matches!(session_msgs[2].role, Role::User),
        "reminder should be appended as a user message"
    );
    assert!(
        session_msgs[2].content.contains("<system-reminder>"),
        "gemini reminders should keep XML wrapping"
    );
    Ok(())
}

/// Provider-normalized output completes legacy agents without exposing synthetic tool identities.
#[tokio::test]
async fn normalized_output_uses_schema_without_submit_tool() -> Result<(), crate::GraphError> {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({"result":"done"})));
    let mut runtime = crate::legacy::runtime::FlowRuntime::new(ExitToolAgentInput {
        query: "test".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory.clone())?);
    runtime.next(Context::default()).await.expect("init step");
    runtime
        .next(Context::default())
        .await
        .expect("dispatch step");
    let captured = factory.captured();
    assert_eq!(captured.len(), 1);
    assert!(matches!(
        captured[0].response_format,
        crate::clients::ResponseFormat::JsonSchema { .. }
    ));
    assert_ne!(
        captured[0].tool_choice,
        crate::clients::ToolChoice::Required
    );
    Ok(())
}

// ── last_step_was_effect tests ─────────────────────────────────────────────────

/// The first `next()` for an agent flow initialises agent state (None arm) — no
/// LLM call, so the flag should be `false`.
#[tokio::test]
async fn last_step_was_effect_false_on_state_init() -> Result<(), crate::GraphError> {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "hi" })));
    let mut runtime = crate::legacy::runtime::FlowRuntime::new(PlainAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory)?);

    // first step: agent None arm — push initial user message, no LLM call
    runtime
        .next(Context::default())
        .await
        .expect("state-init step");
    assert!(
        !runtime.last_step_was_effect(),
        "state-init is not an effect"
    );
    Ok(())
}

/// The second `next()` fires the LLM dispatch — this IS an effect.
#[tokio::test]
async fn last_step_was_effect_true_after_dispatch() -> Result<(), crate::GraphError> {
    let factory = CapturingFactory::new(ResponseMode::Output(json!({ "answer": "hi" })));
    let mut runtime = crate::legacy::runtime::FlowRuntime::new(PlainAgentInput {
        topic: "rust".into(),
    })
    .expect("runtime should build")
    .with_providers(crate::testing::providers(factory)?);

    runtime
        .next(Context::default())
        .await
        .expect("state-init step");
    runtime
        .next(Context::default())
        .await
        .expect("dispatch step");
    assert!(
        runtime.last_step_was_effect(),
        "LLM dispatch must set the effect flag"
    );
    Ok(())
}

/// A `work`-node step runs a user closure and must set the flag.
#[tokio::test]
async fn last_step_was_effect_true_after_work() {
    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct WorkInput {
        value: i64,
    }

    #[derive(Clone, Serialize, Deserialize, JsonSchema)]
    struct WorkOutput {
        doubled: i64,
    }

    impl Flow for WorkInput {
        type Output = WorkOutput;

        fn build(root: Node<Self>) -> Node<Self::Output> {
            root.work(|input: WorkInput, _ctx: Context| async move {
                Ok(WorkOutput {
                    doubled: input.value * 2,
                })
            })
        }
    }

    let mut runtime = crate::legacy::runtime::FlowRuntime::new(WorkInput { value: 3 })
        .expect("runtime should build");

    // single step runs the work closure directly
    runtime.next(Context::default()).await.expect("work step");
    assert!(
        runtime.last_step_was_effect(),
        "work step must set the effect flag"
    );
}

// ── Each node tests ───────────────────────────────────────────────────────────

/// Simple item type for each-node tests: wraps a single integer.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
struct EachItem {
    value: i64,
}

/// Output produced by the per-item sub-flow.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct EachItemOutput {
    doubled: i64,
}

impl Flow for EachItem {
    type Output = EachItemOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.work(|item: EachItem, _ctx: Context| async move {
            Ok(EachItemOutput {
                doubled: item.value * 2,
            })
        })
    }
}

/// A flow that fans out over `Vec<EachItem>` and collects `Vec<EachItemOutput>`.
/// The flow input IS the vec; each node is the entry.
impl Flow for Vec<EachItem> {
    type Output = EachFlowOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.each()
            .work(|items: Vec<EachItemOutput>, _ctx: Context| async move {
                Ok(EachFlowOutput { results: items })
            })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct EachFlowOutput {
    results: Vec<EachItemOutput>,
}

/// `each` node runs the sub-flow once per item and collects all results into `Vec<F::Output>`.
#[tokio::test]
async fn each_node_runs_sub_flow_for_each_item() {
    use crate::legacy::runtime::{FlowRuntime, RunLimits};

    let mut runtime = FlowRuntime::new(vec![
        EachItem { value: 1 },
        EachItem { value: 2 },
        EachItem { value: 3 },
    ])
    .expect("runtime should build");

    let outcome = runtime
        .run_until(Context::default(), RunLimits::default())
        .await
        .expect("run should succeed");

    let result = match outcome {
        crate::legacy::runtime::RunOutcome::Done(v) => v,
        other => panic!("expected Done, got {other:?}"),
    };

    let doubled: Vec<i64> = result.results.iter().map(|r| r.doubled).collect();
    assert_eq!(doubled, vec![2, 4, 6]);
}

/// `each` node with an empty input vec immediately writes an empty `Vec<F::Output>`.
#[tokio::test]
async fn each_node_with_empty_vec_returns_empty_result() {
    use crate::legacy::runtime::{FlowRuntime, RunLimits};

    let mut runtime = FlowRuntime::new(vec![] as Vec<EachItem>).expect("runtime should build");

    let outcome = runtime
        .run_until(Context::default(), RunLimits::default())
        .await
        .expect("run should succeed");

    let result = match outcome {
        crate::legacy::runtime::RunOutcome::Done(v) => v,
        other => panic!("expected Done, got {other:?}"),
    };

    assert!(result.results.is_empty());
}
