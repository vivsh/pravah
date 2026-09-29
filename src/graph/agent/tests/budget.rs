use crate::clients::ErrorKind;
use crate::graph::FetchExecutor;
use crate::graph::tests::host;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmBackend, Message,
    ModelUrl, Provider, ProviderFactory, Role, ToolCall,
};
use crate::deps::Deps;
use crate::graph::{Chat, CompiledFlow, Flow, Runtime, Snapshot, Step, compile};
use crate::tools::ToolError;
use crate::{Context, FlowConf};

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct BudgetRequest {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
struct BudgetAnswer {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct LookupRequest {
    query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct UnknownRequest {
    value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct LookupResult {
    text: String,
}

fn lookup_tools(tools: Toolset) -> Toolset {
    tools.tool(lookup)
}

async fn lookup(input: LookupRequest, _ctx: Context) -> Result<LookupResult, ToolError> {
    if input.query == "recoverable" {
        return Err(ToolError::Validation("try another query".into()));
    }
    Ok(LookupResult {
        text: input.query.to_uppercase(),
    })
}

/// Configures the combined turn-and-tool budget fixture.
async fn configure_budgeted(
    input: BudgetRequest,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    let model = "test:///test-model";
    Ok(AgentConfig::new(
        model,
        "Use lookup and return a structured answer.",
        Message::user(input.text),
    )
    .turn_budget(1)
    .tool_budget::<LookupRequest>(1))
}

/// Configures a turn ceiling without limiting any individual tool.
async fn configure_turn_only(
    input: BudgetRequest,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "Use lookup and return a structured answer.",
        Message::user(input.text),
    )
    .turn_budget(1))
}

/// Configures a tool ceiling while leaving model turns unrestricted.
async fn configure_tool_only(
    input: BudgetRequest,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "Use lookup and return a structured answer.",
        Message::user(input.text),
    )
    .tool_budget::<LookupRequest>(1))
}

/// Configures a two-turn fixture used to reject one model proposal.
async fn configure_rejection(
    input: BudgetRequest,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "Use lookup and return a structured answer.",
        Message::user(input.text),
    )
    .turn_budget(2)
    .tool_budget::<LookupRequest>(1))
}

async fn configure_keep_alive(
    input: BudgetRequest,
    ctx: Context,
) -> Result<AgentConfig, GraphError> {
    configure_budgeted(input, ctx)
        .await
        .map(AgentConfig::keep_alive)
}

/// Builds intentionally invalid settings to verify accumulated errors.
async fn configure_invalid(input: BudgetRequest, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "Invalid budget fixture.",
        Message::user(input.text),
    )
    .turn_budget(0)
    .turn_budget(2)
    .tool_budget::<LookupRequest>(0)
    .tool_budget::<LookupRequest>(1)
    .tool_budget::<LookupRequest>(2)
    .tool_budget::<UnknownRequest>(1))
}

/// Excludes a declared budgeted tool during invocation configuration.
async fn configure_filtered(
    input: BudgetRequest,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "Return a structured answer.",
        Message::user(input.text),
    )
    .tool_filter(ToolFilter::new(|_| false))
    .tool_budget::<LookupRequest>(1))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BudgetObservation {
    point: AgentInterventionPoint,
    turns_remaining: Option<u32>,
    calls_remaining: Option<u32>,
    active_tools: Vec<String>,
}

#[derive(Default)]
struct BudgetTrace(Mutex<Vec<BudgetObservation>>);

/// Records budget observations and attempts to re-enable every tool.
async fn observe_budgets(
    loop_: AgentLoop<BudgetRequest>,
    ctx: Context,
) -> Result<AgentDecision, GraphError> {
    let observation = BudgetObservation {
        point: loop_.point(),
        turns_remaining: loop_.turns_remaining(),
        calls_remaining: loop_.calls_remaining("lookup_request"),
        active_tools: loop_
            .active_tools()
            .iter()
            .map(|tool| tool.name().to_owned())
            .collect(),
    };
    ctx.require::<BudgetTrace>()
        .map_err(|error| GraphError::AgentControl {
            agent: loop_.agent_id().to_owned(),
            reason: error.to_string(),
        })?
        .0
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(observation);
    if loop_.point() == AgentInterventionPoint::AfterTools {
        Ok(AgentDecision::redirect()
            .guidance("Use the evidence already collected")
            .tools(ToolFilter::all()))
    } else {
        Ok(AgentDecision::continue_())
    }
}

async fn reject_first_proposal(
    loop_: AgentLoop<BudgetRequest>,
    _ctx: Context,
) -> Result<AgentDecision, GraphError> {
    if loop_.point() == AgentInterventionPoint::BeforeTools && loop_.control_state().is_none() {
        Ok(AgentDecision::redirect()
            .guidance("Try the proposal once more")
            .with_state(Value::from(true)))
    } else {
        Ok(AgentDecision::continue_())
    }
}

fn controlled_budget_agent(root: Agent<BudgetRequest>) -> Agent<BudgetAnswer> {
    root.tools(lookup_tools)
        .control(observe_budgets)
        .configure(configure_budgeted)
}

fn turn_only_agent(root: Agent<BudgetRequest>) -> Agent<BudgetAnswer> {
    root.tools(lookup_tools).configure(configure_turn_only)
}

fn tool_only_agent(root: Agent<BudgetRequest>) -> Agent<BudgetAnswer> {
    root.tools(lookup_tools).configure(configure_tool_only)
}

fn filtered_budget_agent(root: Agent<BudgetRequest>) -> Agent<BudgetAnswer> {
    root.tools(lookup_tools).configure(configure_filtered)
}

fn invalid_budget_agent(root: Agent<BudgetRequest>) -> Agent<BudgetAnswer> {
    root.tools(lookup_tools).configure(configure_invalid)
}

fn rejection_budget_agent(root: Agent<BudgetRequest>) -> Agent<BudgetAnswer> {
    root.tools(lookup_tools)
        .control(reject_first_proposal)
        .configure(configure_rejection)
}

fn keep_alive_budget_agent(root: Agent<BudgetRequest>) -> Agent<BudgetAnswer> {
    root.tools(lookup_tools).configure(configure_keep_alive)
}

fn controlled_budget_flow(root: Flow<BudgetRequest>) -> Flow<BudgetAnswer> {
    root.agent(controlled_budget_agent)
}

fn turn_only_flow(root: Flow<BudgetRequest>) -> Flow<BudgetAnswer> {
    root.agent(turn_only_agent)
}

fn tool_only_flow(root: Flow<BudgetRequest>) -> Flow<BudgetAnswer> {
    root.agent(tool_only_agent)
}

fn filtered_budget_flow(root: Flow<BudgetRequest>) -> Flow<BudgetAnswer> {
    root.agent(filtered_budget_agent)
}

fn invalid_budget_flow(root: Flow<BudgetRequest>) -> Flow<BudgetAnswer> {
    root.agent(invalid_budget_agent)
}

fn rejection_budget_flow(root: Flow<BudgetRequest>) -> Flow<BudgetAnswer> {
    root.agent(rejection_budget_agent)
}

struct RecordedState {
    responses: VecDeque<Result<ClientResponse, ClientError>>,
    options: Vec<ClientOptions>,
    messages: Vec<Vec<Message>>,
}

#[derive(Clone)]
struct RecordedFactory(Arc<Mutex<RecordedState>>);

impl RecordedFactory {
    /// Creates a recording client factory with deterministic responses.
    fn new(responses: impl IntoIterator<Item = ClientResponse>) -> Self {
        Self(Arc::new(Mutex::new(RecordedState {
            responses: responses.into_iter().map(Ok).collect(),
            options: Vec::new(),
            messages: Vec::new(),
        })))
    }

    fn options(&self) -> Vec<ClientOptions> {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .options
            .clone()
    }

    fn messages(&self) -> Vec<Vec<Message>> {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .messages
            .clone()
    }
}

struct RecordedClient {
    state: Arc<Mutex<RecordedState>>,
    model: ModelUrl,
    options: ClientOptions,
}

#[derive(Debug)]
struct RecordFailure;

impl std::fmt::Display for RecordFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("injected record failure")
    }
}

impl std::error::Error for RecordFailure {}

#[derive(Clone)]
struct FailOnceStore {
    calls: Arc<AtomicUsize>,
    fail_at: usize,
}

impl crate::history::HistoryStore for FailOnceStore {
    type Error = RecordFailure;

    async fn record(&self, _entry: &crate::history::HistoryEntry) -> Result<(), Self::Error> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call == self.fail_at {
            Err(RecordFailure)
        } else {
            Ok(())
        }
    }
}

impl LlmBackend for RecordedClient {
    fn model_url(&self) -> &ModelUrl {
        &self.model
    }

    fn options(&self) -> &ClientOptions {
        &self.options
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.options.push(self.options.clone());
        state.messages.push(messages.to_vec());
        match state.responses.pop_front() {
            Some(response) => response,
            None => Err(ClientError::new(
                ErrorKind::Provider,
                "budget test response queue exhausted",
            )),
        }
    }
}

impl ProviderFactory for RecordedFactory {
    /// Records provider options and creates a deterministic response client.
    async fn llm(
        &self,
        model_url: &ModelUrl,
        options: ClientOptions,
    ) -> Result<Client, ClientError> {
        Ok(Client::from_backend(RecordedClient {
            state: Arc::clone(&self.0),
            model: model_url.clone(),
            options,
        }))
    }
}

fn tool_call(id: &str, query: &str) -> ToolCall {
    ToolCall::new(
        id.into(),
        "lookup_request".into(),
        json!({ "query": query }),
    )
}

fn tool_response(calls: Vec<ToolCall>) -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::ToolCalls { text: None, calls },
    )
}

fn output_response(text: &str) -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::Output(json!({ "text": text })),
    )
}

fn test_context(
    factory: RecordedFactory,
    trace: Option<Arc<BudgetTrace>>,
) -> Result<Context, crate::GraphError> {
    let mut deps = Deps::default();
    if let Some(trace) = trace {
        deps.insert(trace);
    }
    Ok(Context::new(FlowConf::default())
        .with_deps(deps)
        .with_providers(crate::testing::providers(factory)?))
}

/// Drives a compiled test flow until its typed output is available.
async fn run_to_output(
    flow: &CompiledFlow<BudgetRequest, BudgetAnswer>,
    runtime: &mut Runtime,
    executor: &FetchExecutor,
) -> Result<BudgetAnswer, GraphError> {
    loop {
        match host::step(runtime, executor).await? {
            Step::Continue => {}
            Step::Done(value) => return flow.decode_output(value),
            Step::Fetch(_) => panic!("host must deliver fetch"),
            Step::Suspend(_) => {
                return Err(GraphError::Invalid(
                    "budget test agent suspended unexpectedly".into(),
                ));
            }
        }
    }
}

/// Verifies exact admission, controller observations, and hard conclusion ceilings.
#[tokio::test]
async fn budgets_share_tool_visibility_and_conclusion_control() -> Result<(), crate::GraphError> {
    let flow = compile(controlled_budget_flow).expect("budgeted flow should compile");
    let responses = [
        tool_response(vec![
            tool_call("first", "recoverable"),
            tool_call("second", "two"),
            tool_call("third", "three"),
        ]),
        output_response("done"),
    ];
    let factory = RecordedFactory::new(responses);
    let trace = Arc::new(BudgetTrace::default());
    let ctx = test_context(factory.clone(), Some(Arc::clone(&trace)))?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();
    let output = run_to_output(&flow, &mut runtime, &executor).await.unwrap();

    assert_eq!(
        output,
        BudgetAnswer {
            text: "done".into()
        }
    );
    assert_budgeted_requests(&factory);
    assert_budget_observations(&trace);
    Ok(())
}

/// Checks tool admission, conclusion visibility, and guidance ordering.
fn assert_budgeted_requests(factory: &RecordedFactory) {
    let options = factory.options();
    assert_eq!(options.len(), 2);
    assert_eq!(options[0].tools.len(), 1);
    assert!(options[1].tools.is_empty());
    let messages = factory.messages();
    let unavailable = messages[1]
        .iter()
        .filter(|message| message.content.contains("tool unavailable for this turn"))
        .count();
    assert_eq!(unavailable, 2);
    assert!(messages[1].iter().any(|message| {
        matches!(message.role, Role::User) && message.content.contains("FINAL TURN")
    }));
    let guidance = messages[1]
        .iter()
        .position(|message| {
            message
                .content
                .contains("Use the evidence already collected")
        })
        .unwrap();
    let reminder = messages[1]
        .iter()
        .position(|message| message.content.contains("FINAL TURN"))
        .unwrap();
    assert!(guidance < reminder);
}

/// Checks the controller's remaining-turn and remaining-call observations.
fn assert_budget_observations(trace: &BudgetTrace) {
    let observations = trace.0.lock().unwrap_or_else(|error| error.into_inner());
    let before_tools = observations
        .iter()
        .find(|item| item.point == AgentInterventionPoint::BeforeTools)
        .expect("BeforeTools should be observed");
    assert_eq!(before_tools.turns_remaining, Some(0));
    assert_eq!(before_tools.calls_remaining, Some(1));
    let after_tools = observations
        .iter()
        .find(|item| item.point == AgentInterventionPoint::AfterTools)
        .expect("AfterTools should be observed");
    assert_eq!(after_tools.calls_remaining, Some(0));
    assert!(after_tools.active_tools.is_empty());
}

/// Verifies an agent-only budget concludes while retaining domain tools on its ordinary turn.
#[tokio::test]
async fn agent_only_budget_forces_one_tool_disabled_conclusion() -> Result<(), crate::GraphError> {
    let flow = compile(turn_only_flow).expect("turn-budgeted flow should compile");
    let factory = RecordedFactory::new([
        tool_response(vec![tool_call("first", "one")]),
        output_response("turn ceiling"),
    ]);
    let ctx = test_context(factory.clone(), None)?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();

    let output = run_to_output(&flow, &mut runtime, &executor).await.unwrap();
    assert_eq!(output.text, "turn ceiling");
    let options = factory.options();
    assert_eq!(options[0].tools.len(), 1);
    assert!(options[1].tools.is_empty());
    Ok(())
}

/// Verifies a tool-only budget removes the tool without forcing conclusion.
#[tokio::test]
async fn tool_only_budget_leaves_model_turns_unrestricted() -> Result<(), crate::GraphError> {
    let flow = compile(tool_only_flow).expect("tool-budgeted flow should compile");
    let factory = RecordedFactory::new([
        tool_response(vec![tool_call("first", "one")]),
        output_response("tool ceiling"),
    ]);
    let ctx = test_context(factory.clone(), None)?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();

    let output = run_to_output(&flow, &mut runtime, &executor).await.unwrap();
    assert_eq!(output.text, "tool ceiling");
    let options = factory.options();
    assert_eq!(options[0].tools.len(), 1);
    assert!(options[1].tools.is_empty());
    assert!(
        factory.messages()[1]
            .iter()
            .all(|message| !message.content.contains("FINAL TURN"))
    );
    Ok(())
}

/// Verifies natural output on the last ordinary turn needs no forced request.
#[tokio::test]
async fn last_budgeted_turn_may_complete_naturally() -> Result<(), crate::GraphError> {
    let flow = compile(controlled_budget_flow).expect("budgeted flow should compile");
    let factory = RecordedFactory::new([output_response("natural")]);
    let trace = Arc::new(BudgetTrace::default());
    let ctx = test_context(factory.clone(), Some(trace))?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();

    let output = run_to_output(&flow, &mut runtime, &executor).await.unwrap();
    assert_eq!(output.text, "natural");
    assert_eq!(factory.options().len(), 1);
    assert!(
        factory.messages()[0]
            .iter()
            .all(|message| !message.content.contains("FINAL TURN"))
    );
    Ok(())
}

/// Verifies conclusion retains schema-constrained output without domain tools.
#[tokio::test]
async fn budget_conclusion_preserves_normalized_structured_output() -> Result<(), crate::GraphError>
{
    let flow = compile(controlled_budget_flow).expect("budgeted flow should compile");
    let factory = RecordedFactory::new([
        tool_response(vec![tool_call("first", "one")]),
        output_response("exit done"),
    ]);
    let trace = Arc::new(BudgetTrace::default());
    let ctx = test_context(factory.clone(), Some(trace))?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(
            BudgetRequest {
                text: "exit".into(),
            },
            uuid::Uuid::nil(),
        )
        .unwrap();

    let output = run_to_output(&flow, &mut runtime, &executor).await.unwrap();
    assert_eq!(output.text, "exit done");
    assert!(factory.options()[1].tools.is_empty());
    assert!(matches!(
        factory.options()[1].response_format,
        crate::clients::ResponseFormat::JsonSchema { .. }
    ));
    assert!(
        factory.messages()[1]
            .iter()
            .any(|message| message.content.contains("FINAL TURN"))
    );
    Ok(())
}

/// Verifies rejected proposals consume neither history nor tool capacity.
#[tokio::test]
async fn rejected_proposal_does_not_consume_tool_budget() -> Result<(), crate::GraphError> {
    let flow = compile(rejection_budget_flow).expect("budgeted flow should compile");
    let factory = RecordedFactory::new([
        tool_response(vec![tool_call("rejected", "first")]),
        tool_response(vec![tool_call("accepted", "second")]),
        output_response("done"),
    ]);
    let ctx = test_context(factory.clone(), None)?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();

    let output = run_to_output(&flow, &mut runtime, &executor).await.unwrap();
    assert_eq!(output.text, "done");
    let options = factory.options();
    assert_eq!(options[0].tools.len(), 1);
    assert_eq!(options[1].tools.len(), 1);
    assert!(options[2].tools.is_empty());
    assert_only_accepted_call_reaches_history(&factory.messages()[2]);
    Ok(())
}

/// Verifies keep-alive history does not carry exhausted budgets into a new invocation.
#[tokio::test]
async fn keep_alive_agent_resets_budgets_for_each_invocation() -> Result<(), crate::GraphError> {
    let factory = RecordedFactory::new([
        tool_response(vec![tool_call("first", "one")]),
        output_response("first answer"),
        tool_response(vec![tool_call("second", "two")]),
        output_response("second answer"),
    ]);
    let ctx = test_context(factory.clone(), None)?;
    let mut chat = Chat::new(keep_alive_budget_agent, ctx).expect("chat initializes");

    let first = chat
        .send(BudgetRequest {
            text: "first".into(),
        })
        .await
        .unwrap();
    let second = chat
        .send(BudgetRequest {
            text: "second".into(),
        })
        .await
        .unwrap();
    assert_eq!(first.output.text, "first answer");
    assert_eq!(second.output.text, "second answer");
    let options = factory.options();
    assert_eq!(options[0].tools.len(), 1);
    assert!(options[1].tools.is_empty());
    assert_eq!(options[2].tools.len(), 1);
    assert!(options[3].tools.is_empty());
    Ok(())
}

/// Checks that a discarded proposal never enters persisted history.
fn assert_only_accepted_call_reaches_history(messages: &[Message]) {
    let accepted = messages.iter().any(|message| match &message.role {
        Role::AssistantToolCalls { calls } => calls.iter().any(|call| call.id == "accepted"),
        _ => false,
    });
    let rejected = messages.iter().any(|message| match &message.role {
        Role::AssistantToolCalls { calls } => calls.iter().any(|call| call.id == "rejected"),
        _ => false,
    });
    assert!(accepted);
    assert!(!rejected);
}

/// Verifies filtered declared budgets are harmless while invalid budgets accumulate.
#[tokio::test]
async fn activation_validates_only_effective_budget_errors() -> Result<(), crate::GraphError> {
    let filtered = compile(filtered_budget_flow).expect("filtered flow should compile");
    let factory = RecordedFactory::new([output_response("filtered")]);
    let ctx = test_context(factory.clone(), None)?;
    let executor =
        FetchExecutor::new(ctx.clone()).with_registry(Arc::new(filtered.registry().clone()));
    let mut runtime = filtered
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();
    let output = run_to_output(&filtered, &mut runtime, &executor)
        .await
        .unwrap();
    assert_eq!(output.text, "filtered");
    assert!(factory.options()[0].tools.is_empty());

    assert_invalid_budget_configuration().await;
    Ok(())
}

/// Checks that activation reports every accumulated budget error at once.
async fn assert_invalid_budget_configuration() {
    let invalid = compile(invalid_budget_flow).expect("invalid config is runtime data");
    let executor =
        FetchExecutor::new(Context::default()).with_registry(Arc::new(invalid.registry().clone()));
    let mut runtime = invalid
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();
    let error = host::step(&mut runtime, &executor)
        .await
        .expect_err("invalid budget configuration should fail");
    let message = error.to_string();
    assert!(message.contains("positive"));
    assert!(message.contains("only be declared once"));
    assert!(message.contains("unknown agent tool 'unknown_request'"));
    assert!(runtime.snapshot().unwrap().history().entries().is_empty());
}

/// Verifies budget counters and forced conclusion survive JSON and CBOR snapshots.
#[tokio::test]
async fn budget_state_restores_without_reconfiguration() -> Result<(), crate::GraphError> {
    let flow = compile(controlled_budget_flow).expect("budgeted flow should compile");
    let factory = RecordedFactory::new([
        tool_response(vec![tool_call("first", "one")]),
        output_response("restored"),
    ]);
    let trace = Arc::new(BudgetTrace::default());
    let ctx = test_context(factory.clone(), Some(Arc::clone(&trace)))?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();
    advance_through_after_tools(&mut runtime, &trace, &executor).await;

    let snapshot = runtime.snapshot().unwrap();
    let json = serde_json::to_vec(&snapshot).unwrap();
    let from_json: Snapshot = serde_json::from_slice(&json).unwrap();
    let mut cbor = Vec::new();
    ciborium::into_writer(&from_json, &mut cbor).unwrap();
    let restored_snapshot = ciborium::from_reader(cbor.as_slice()).unwrap();
    let mut restored = flow.restore(restored_snapshot).unwrap();
    let output = run_to_output(&flow, &mut restored, &executor)
        .await
        .unwrap();

    assert_eq!(output.text, "restored");
    assert!(factory.options()[1].tools.is_empty());
    Ok(())
}

/// Verifies malformed remaining counts are rejected before runtime expansion.
#[tokio::test]
async fn restore_rejects_malformed_budget_state() -> Result<(), crate::GraphError> {
    let flow = compile(controlled_budget_flow).expect("budgeted flow should compile");
    let factory = RecordedFactory::new([output_response("unused")]);
    let trace = Arc::new(BudgetTrace::default());
    let ctx = test_context(factory, Some(trace))?;
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();
    host::step(&mut runtime, &executor).await.unwrap();
    let mut snapshot = runtime.snapshot().unwrap();
    let frame = snapshot.state.frame_mut(0).unwrap();
    let checkpoint = &mut Arc::make_mut(&mut frame.checkpoints)[0].value;
    let mut encoded = serde_json::to_value(&*checkpoint).unwrap();
    encoded["budget"]["tools"][0]["remaining"] = json!(2);
    *checkpoint = to_value(encoded).unwrap();

    let error = match flow.restore(snapshot) {
        Ok(_) => panic!("remaining calls above the limit must fail"),
        Err(error) => error,
    };
    assert!(matches!(error, GraphError::SnapshotValidation(_)));
    Ok(())
}

/// Verifies a failed history batch does not consume a tool budget twice.
#[tokio::test]
async fn history_failure_leaves_budget_admission_retryable() -> Result<(), crate::GraphError> {
    let flow = compile(controlled_budget_flow).expect("budgeted flow should compile");
    let factory = RecordedFactory::new([
        tool_response(vec![tool_call("first", "one"), tool_call("second", "two")]),
        output_response("retried"),
    ]);
    let trace = Arc::new(BudgetTrace::default());
    let ctx = test_context(factory.clone(), Some(trace))?;
    let store = FailOnceStore {
        calls: Arc::new(AtomicUsize::new(0)),
        fail_at: 2,
    };
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(BudgetRequest { text: "run".into() }, uuid::Uuid::nil())
        .unwrap();
    let executor = executor.with_store(store);

    await_history_failure(&mut runtime, &executor).await;
    let output = run_to_output(&flow, &mut runtime, &executor).await.unwrap();
    assert_eq!(output.text, "retried");
    let unavailable = factory.messages()[1]
        .iter()
        .filter(|message| message.content.contains("tool unavailable for this turn"))
        .count();
    assert_eq!(unavailable, 1);
    Ok(())
}

/// Advances until the injected persistence error reaches the caller.
async fn await_history_failure(runtime: &mut Runtime, executor: &FetchExecutor) {
    loop {
        match host::step(runtime, executor).await {
            Ok(Step::Continue) => {}
            Err(GraphError::HistoryPersistence(_)) => return,
            Ok(other) => panic!("expected retryable history failure, got {other:?}"),
            Err(error) => panic!("expected history failure, got {error}"),
        }
    }
}

/// Advances until the controller has observed the complete accepted tool batch.
async fn advance_through_after_tools(
    runtime: &mut Runtime,
    trace: &BudgetTrace,
    executor: &FetchExecutor,
) {
    loop {
        host::step(runtime, executor).await.unwrap();
        let reached = trace
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .any(|item| item.point == AgentInterventionPoint::AfterTools);
        if reached {
            return;
        }
    }
}
