//! Integration tests for agent tool-call dispatch.

use pravah::clients::{ClientError, Role};
use pravah::flows::{
    Agent, AgentConfig, AgentError, Flow, FlowError, FlowRuntime, FlowStep, Node, PhaseKind,
    Toolbox,
};
use pravah::testing::{CapturingHistoryStore, ScriptedFactory, mock_tool_call};
use pravah::tools::{ToolError, ToolOutput};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

fn ctx() -> Context {
    Context::new(FlowConf::default())
}

async fn run_to_done<I: Flow>(mut rt: FlowRuntime<I>) -> Result<I::Output, FlowError> {
    for _ in 0..60 {
        match rt.next(ctx()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(v) => return Ok(v),
            FlowStep::Suspend(_) => {
                return Err(FlowError::Internal {
                    handler: "run_to_done",
                    detail: "unexpected suspension".into(),
                });
            }
        }
    }
    Err(FlowError::Internal {
        handler: "run_to_done",
        detail: "flow did not finish within 60 steps".into(),
    })
}

async fn run_to_err<I: Flow>(mut rt: FlowRuntime<I>) -> FlowError {
    for _ in 0..60 {
        match rt.next(ctx()).await {
            Err(e) => return e,
            Ok(FlowStep::Continue) => {}
            Ok(FlowStep::Done(_)) => panic!("expected error but flow completed"),
            Ok(FlowStep::Suspend(_)) => panic!("expected error but flow suspended"),
        }
    }
    panic!("flow did not error within 60 steps")
}

// ---------------------------------------------------------------------------
// Tool types (struct definitions only; execution via work handlers)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "echo")]
struct EchoInput {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EchoOutput {
    echoed: String,
}
impl ToolOutput for EchoOutput {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "reverse")]
struct ReverseInput {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReverseOutput {
    reversed: String,
}
impl ToolOutput for ReverseOutput {}

// ---------------------------------------------------------------------------
// Work handlers
// ---------------------------------------------------------------------------

async fn echo_handler(input: EchoInput, _ctx: Context) -> Result<EchoOutput, ToolError> {
    Ok(EchoOutput { echoed: input.text })
}
async fn reverse_handler(input: ReverseInput, _ctx: Context) -> Result<ReverseOutput, ToolError> {
    Ok(ReverseOutput {
        reversed: input.text.chars().rev().collect(),
    })
}

fn with_echo<A: Agent>(toolbox: Toolbox<A>) -> Toolbox<A> {
    toolbox.tool_handler(
        |input: EchoInput, ctx: Context| async move { echo_handler(input, ctx).await },
    )
}

fn with_echo_and_reverse<A: Agent>(toolbox: Toolbox<A>) -> Toolbox<A> {
    with_echo(toolbox).tool_handler(|input: ReverseInput, ctx: Context| async move {
        reverse_handler(input, ctx).await
    })
}

// ---------------------------------------------------------------------------
// Agent / Flow definitions
// ---------------------------------------------------------------------------

macro_rules! simple_agent {
    ($in:ident, $out:ident, $preamble:expr) => {
        impl Agent for $in {
            type Output = $out;
            fn configure() -> AgentConfig {
                AgentConfig::new($preamble, "test://model")
            }
        }
    };
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectIn {
    prompt: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectOut {
    answer: String,
}
simple_agent!(DirectIn, DirectOut, "Answer directly.");
impl Flow for DirectIn {
    type Output = DirectOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ValidIn {
    query: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ValidOut {
    result: String,
}
simple_agent!(ValidIn, ValidOut, "Use echo then answer.");
impl Flow for ValidIn {
    type Output = ValidOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MultiToolIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MultiToolOut {
    summary: String,
}
simple_agent!(MultiToolIn, MultiToolOut, "Use echo and reverse.");
impl Flow for MultiToolIn {
    type Output = MultiToolOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo_and_reverse)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DupIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DupOut {
    result: String,
}
simple_agent!(DupIn, DupOut, "Use echo.");
impl Flow for DupIn {
    type Output = DupOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct UnknownIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct UnknownOut {
    result: String,
}
simple_agent!(UnknownIn, UnknownOut, "Use echo.");
impl Flow for UnknownIn {
    type Output = UnknownOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct HistoryIn {
    q: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct HistoryOut {
    a: String,
}
simple_agent!(HistoryIn, HistoryOut, "Use echo.");
impl Flow for HistoryIn {
    type Output = HistoryOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChainIn {
    start: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChainOut {
    final_value: String,
}
simple_agent!(ChainIn, ChainOut, "Chain echo.");
impl Flow for ChainIn {
    type Output = ChainOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LlmErrIn {
    x: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LlmErrOut {
    y: String,
}
simple_agent!(LlmErrIn, LlmErrOut, "Call echo.");
impl Flow for LlmErrIn {
    type Output = LlmErrOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StructModeIn {
    q: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StructModeOut {
    answer: String,
}
simple_agent!(StructModeIn, StructModeOut, "Answer directly.");
impl Flow for StructModeIn {
    type Output = StructModeOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StoreIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StoreOut {
    echoed: String,
}
simple_agent!(StoreIn, StoreOut, "Echo and answer.");
impl Flow for StoreIn {
    type Output = StoreOut;
    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(with_echo)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_direct_response() {
    let factory = ScriptedFactory::new().then_output(json!({ "answer": "42" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(DirectIn {
        prompt: "what is the answer?".into(),
    })
    .unwrap()
    .with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.answer, "42");
    assert_eq!(spy.calls().len(), 1);
    assert_eq!(spy.remaining(), 0);
}

#[tokio::test]
async fn test_valid_tool_call_then_exit() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "c1",
            "echo",
            json!({ "text": "hello" }),
        )])
        .then_output(json!({ "result": "echoed:hello" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ValidIn {
        query: "echo hello".into(),
    })
    .unwrap()
    .with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "echoed:hello");
    assert_eq!(spy.calls().len(), 2);
}

#[tokio::test]
async fn test_inspector_tracks_tool_turns() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "c1",
            "echo",
            json!({ "text": "hello" }),
        )])
        .then_output(json!({ "result": "echoed:hello" }));
    let mut rt = FlowRuntime::new(ValidIn {
        query: "echo hello".into(),
    })
    .unwrap()
    .with_factory(factory);

    // Initial: 1 frame, agent not yet dispatched.
    assert_eq!(rt.inspector().depth(), 1);
    let top = rt.inspector().top_frame().unwrap();
    assert_eq!(top.callable_entry, "ValidIn");
    assert!(top.agent_phases.is_empty());

    // After first step: agent state initialised (Dispatch), no LLM call yet.
    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    assert!(
        rt.inspector()
            .top_frame()
            .unwrap()
            .agent_phases
            .first()
            .is_some_and(|ap| matches!(ap.phase, PhaseKind::Dispatch))
    );

    // After second step: LLM called → tool calls returned → PendingTool.
    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    let inspector = rt.inspector();
    assert_eq!(inspector.depth(), 1, "agent lives in same frame");
    let top = inspector.top_frame().unwrap();
    match top.agent_phases.first().map(|ap| &ap.phase) {
        Some(PhaseKind::PendingTool {
            active_calls,
            waiting_count,
        }) => {
            assert_eq!(active_calls, &vec!["echo".to_owned()]);
            assert_eq!(*waiting_count, 0);
        }
        other => panic!("expected PendingTool, got {other:?}"),
    }

    // Run to completion.
    loop {
        match rt.next(ctx()).await.unwrap() {
            FlowStep::Done(out) => {
                assert_eq!(out.result, "echoed:hello");
                break;
            }
            FlowStep::Continue => {}
            FlowStep::Suspend(_) => panic!("unexpected suspend"),
        }
    }
    assert_eq!(rt.inspector().depth(), 0);
}

#[tokio::test]
async fn test_multiple_tool_calls_in_one_turn() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo", json!({ "text": "hi" })),
            mock_tool_call("c2", "reverse", json!({ "text": "hi" })),
        ])
        .then_output(json!({ "summary": "done" }));
    let rt = FlowRuntime::new(MultiToolIn { text: "hi".into() })
        .unwrap()
        .with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "done");
}

#[tokio::test]
async fn test_same_tool_twice_runs_serially() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo", json!({ "text": "a" })),
            mock_tool_call("c2", "echo", json!({ "text": "b" })),
        ])
        .then_output(json!({ "result": "done" }));
    let rt = FlowRuntime::new(DupIn {
        text: "test".into(),
    })
    .unwrap()
    .with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "done");
}

#[tokio::test]
async fn test_duplicate_call_id_is_rejected() {
    let factory = ScriptedFactory::new().then_tool_calls(vec![
        mock_tool_call("c1", "echo", json!({ "text": "a" })),
        mock_tool_call("c1", "echo", json!({ "text": "b" })),
    ]);
    let rt = FlowRuntime::new(DupIn {
        text: "test".into(),
    })
    .unwrap()
    .with_factory(factory);
    let err = run_to_err(rt).await;
    assert!(matches!(
        err,
        FlowError::Agent(AgentError::DuplicateToolCall { .. })
    ));
}

#[tokio::test]
async fn test_unknown_tool_name_becomes_error_result() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "nonexistent_tool", json!({}))])
        .then_output(json!({ "result": "recovered" }));
    let rt = FlowRuntime::new(UnknownIn {
        text: "test".into(),
    })
    .unwrap()
    .with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "recovered");
}

#[tokio::test]
async fn test_tool_result_present_in_second_dispatch() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "tc1",
            "echo",
            json!({ "text": "ping" }),
        )])
        .then_output(json!({ "a": "pong" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(HistoryIn { q: "ping?".into() })
        .unwrap()
        .with_factory(factory);
    run_to_done(rt).await.unwrap();

    let calls = spy.calls();
    assert_eq!(calls.len(), 2);
    let second = &calls[1].1;
    assert!(second.iter().any(|m| matches!(&m.role, Role::AssistantToolCalls { calls } if calls.iter().any(|c| c.id == "tc1"))));
    assert!(
        second
            .iter()
            .any(|m| matches!(&m.role, Role::Tool { call_id } if call_id == "tc1"))
    );
    let atc_pos = second
        .iter()
        .position(|m| matches!(&m.role, Role::AssistantToolCalls { .. }))
        .unwrap();
    let tool_pos = second
        .iter()
        .position(|m| matches!(&m.role, Role::Tool { call_id } if call_id == "tc1"))
        .unwrap();
    assert!(atc_pos < tool_pos);
}

#[tokio::test]
async fn test_malformed_tool_input_remains_recoverable() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("tc1", "echo", json!({ "text": 3 }))])
        .then_output(json!({ "result": "recovered" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ValidIn {
        query: "test".into(),
    })
    .unwrap()
    .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();

    assert_eq!(out.result, "recovered");
    let calls = spy.calls();
    assert_eq!(calls.len(), 2);
    let second = &calls[1].1;
    let tool_error = second
        .iter()
        .find(|m| matches!(&m.role, Role::Tool { call_id } if call_id == "tc1"))
        .expect("recoverable tool error should be sent back to model");
    assert!(tool_error.content.contains(r#""error_kind":"TypeError""#));
    assert!(tool_error.content.contains(r#""recoverable":true"#));
}

#[tokio::test]
async fn test_both_tool_results_present_after_multi_call_turn() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("id1", "echo", json!({ "text": "a" })),
            mock_tool_call("id2", "reverse", json!({ "text": "b" })),
        ])
        .then_output(json!({ "summary": "ok" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(MultiToolIn {
        text: "test".into(),
    })
    .unwrap()
    .with_factory(factory);
    run_to_done(rt).await.unwrap();
    let second = &spy.calls()[1].1;
    for call_id in ["id1", "id2"] {
        assert!(
            second
                .iter()
                .any(|m| matches!(&m.role, Role::Tool { call_id: cid } if cid == call_id))
        );
    }
}

#[tokio::test]
async fn test_three_sequential_tool_call_rounds() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("r1", "echo", json!({ "text": "one" }))])
        .then_tool_calls(vec![mock_tool_call("r2", "echo", json!({ "text": "two" }))])
        .then_tool_calls(vec![mock_tool_call(
            "r3",
            "echo",
            json!({ "text": "three" }),
        )])
        .then_output(json!({ "final_value": "done" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ChainIn { start: "go".into() })
        .unwrap()
        .with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.final_value, "done");
    assert_eq!(spy.calls().len(), 4);
    assert_eq!(spy.remaining(), 0);
}

#[tokio::test]
async fn test_tool_results_accumulate_across_rounds() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("r1", "echo", json!({ "text": "a" }))])
        .then_tool_calls(vec![mock_tool_call("r2", "echo", json!({ "text": "b" }))])
        .then_output(json!({ "final_value": "done" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ChainIn {
        start: "chain".into(),
    })
    .unwrap()
    .with_factory(factory);
    run_to_done(rt).await.unwrap();
    let calls = spy.calls();
    let t2 = &calls[1].1;
    assert!(
        t2.iter()
            .any(|m| matches!(&m.role, Role::Tool { call_id } if call_id == "r1"))
    );
    let t3 = &calls[2].1;
    for id in ["r1", "r2"] {
        assert!(
            t3.iter()
                .any(|m| matches!(&m.role, Role::Tool { call_id } if call_id == id))
        );
    }
}

#[tokio::test]
async fn test_llm_error_on_first_dispatch_propagates() {
    let factory = ScriptedFactory::new().then_err(ClientError::Llm("network timeout".into()));
    let rt = FlowRuntime::new(LlmErrIn { x: "hello".into() })
        .unwrap()
        .with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("network timeout"));
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}

#[tokio::test]
async fn test_llm_error_on_second_dispatch_propagates() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "hi" }))])
        .then_err(ClientError::Llm("server error on retry".into()));
    let rt = FlowRuntime::new(LlmErrIn { x: "hello".into() })
        .unwrap()
        .with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("server error on retry"));
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}

#[tokio::test]
async fn test_direct_output_in_tool_mode_completes_agent() {
    let factory = ScriptedFactory::new().then_output(json!({ "answer": "shortcut" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(StructModeIn { q: "quick?".into() })
        .unwrap()
        .with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.answer, "shortcut");
    assert_eq!(spy.calls().len(), 1);
}

#[tokio::test]
async fn test_capturing_store_receives_all_messages() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "s1",
            "echo",
            json!({ "text": "world" }),
        )])
        .then_output(json!({ "echoed": "world" }));
    let store = CapturingHistoryStore::new();
    let store_spy = store.clone();
    let rt = FlowRuntime::new(StoreIn {
        text: "world".into(),
    })
    .unwrap()
    .with_factory(factory)
    .with_store(store);
    run_to_done(rt).await.unwrap();
    assert!(store_spy.flush_count() >= 2);
    let snapshot = store_spy.last_snapshot().unwrap();
    let live: Vec<_> = snapshot.iter().filter(|e| !e.evicted).collect();
    assert!(live.iter().any(|e| matches!(e.message.role, Role::User)));
    assert!(
        live.iter()
            .any(|e| matches!(&e.message.role, Role::AssistantToolCalls { .. }))
    );
    assert!(
        live.iter()
            .any(|e| matches!(&e.message.role, Role::Tool { call_id } if call_id == "s1"))
    );
}

#[tokio::test]
async fn test_store_flush_count_matches_steps() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("f1", "echo", json!({ "text": "x" }))])
        .then_output(json!({ "echoed": "x" }));
    let store = CapturingHistoryStore::new();
    let store_spy = store.clone();
    let rt = FlowRuntime::new(StoreIn { text: "x".into() })
        .unwrap()
        .with_factory(factory)
        .with_store(store);
    run_to_done(rt).await.unwrap();
    assert!(store_spy.flush_count() >= 2);
}

#[tokio::test]
async fn test_no_scripted_responses_remain_after_valid_run() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("q1", "echo", json!({ "text": "a" }))])
        .then_output(json!({ "result": "done" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ValidIn {
        query: "consume all".into(),
    })
    .unwrap()
    .with_factory(factory);
    run_to_done(rt).await.unwrap();
    assert_eq!(spy.remaining(), 0);
}

#[tokio::test]
async fn test_no_scripted_responses_remain_after_direct_run() {
    let factory = ScriptedFactory::new().then_output(json!({ "answer": "yes" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(DirectIn {
        prompt: "any?".into(),
    })
    .unwrap()
    .with_factory(factory);
    run_to_done(rt).await.unwrap();
    assert_eq!(spy.remaining(), 0);
}

// ---------------------------------------------------------------------------
// Validation tests
// ---------------------------------------------------------------------------
