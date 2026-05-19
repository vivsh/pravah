//! Integration tests for parallel and queued tool dispatch.

use pravah::clients::ClientError;
use pravah::flows::{Agent, AgentConfig, AgentError, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::tools::ToolOutput;
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
                })
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
// Tool types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "echo")]
struct EchoInput { text: String }
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EchoOutput { echoed: String }
impl ToolOutput for EchoOutput {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "reverse")]
struct ReverseInput { text: String }
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReverseOutput { reversed: String }
impl ToolOutput for ReverseOutput {}

// ---------------------------------------------------------------------------
// Work handlers
// ---------------------------------------------------------------------------

async fn echo_handler(input: EchoInput, _ctx: Context) -> Result<EchoOutput, FlowError> {
    Ok(EchoOutput { echoed: input.text })
}
async fn reverse_handler(input: ReverseInput, _ctx: Context) -> Result<ReverseOutput, FlowError> {
    Ok(ReverseOutput { reversed: input.text.chars().rev().collect() })
}

// ---------------------------------------------------------------------------
// Agent / Flow definitions
// ---------------------------------------------------------------------------

macro_rules! simple_agent {
    ($in:ident, $out:ident, $preamble:expr) => {
        impl Agent for $in {
            type Output = $out;
            fn build() -> AgentConfig {
                AgentConfig::new($preamble, "test://model")
            }
        }
    };
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)] struct ParallelIn { text: String }
#[derive(Debug, Serialize, Deserialize, JsonSchema)] struct ParallelOut { summary: String }
simple_agent!(ParallelIn, ParallelOut, "Use echo and reverse.");
impl Flow for ParallelIn {
    type Output = ParallelOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<ParallelIn>()
            .tool::<ParallelIn, EchoInput, EchoOutput>()
            .tool::<ParallelIn, ReverseInput, ReverseOutput>()
            .work(echo_handler)
            .work(reverse_handler)
            .build()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)] struct LlmFailIn { x: i64 }
#[derive(Debug, Serialize, Deserialize, JsonSchema)] struct LlmFailOut { result: String }
simple_agent!(LlmFailIn, LlmFailOut, "Use echo.");
impl Flow for LlmFailIn {
    type Output = LlmFailOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<LlmFailIn>()
            .tool::<LlmFailIn, EchoInput, EchoOutput>()
            .work(echo_handler)
            .build()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)] struct UnknownParallelIn { text: String }
#[derive(Debug, Serialize, Deserialize, JsonSchema)] struct UnknownParallelOut { summary: String }
simple_agent!(UnknownParallelIn, UnknownParallelOut, "Use echo and reverse.");
impl Flow for UnknownParallelIn {
    type Output = UnknownParallelOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<UnknownParallelIn>()
            .tool::<UnknownParallelIn, EchoInput, EchoOutput>()
            .tool::<UnknownParallelIn, ReverseInput, ReverseOutput>()
            .work(echo_handler)
            .work(reverse_handler)
            .build()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_two_distinct_tools_run_in_parallel() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "hello" })),
            mock_tool_call("c2", "reverse", json!({ "text": "world" })),
        ])
        .then_output(json!({ "summary": "done" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ParallelIn { text: "test".into() }).unwrap().with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "done");
    assert_eq!(spy.calls().len(), 2);
}

#[tokio::test]
async fn test_unknown_tool_alongside_known_gets_error_result() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",         json!({ "text": "hi" })),
            mock_tool_call("c2", "no_such_tool", json!({})),
        ])
        .then_output(json!({ "summary": "recovered" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(UnknownParallelIn { text: "test".into() }).unwrap().with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "recovered");
    assert_eq!(spy.calls().len(), 2);
}

#[tokio::test]
async fn test_all_unknown_tools_in_one_turn_re_dispatch_immediately() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "ghost1", json!({})),
            mock_tool_call("c2", "ghost2", json!({})),
        ])
        .then_output(json!({ "summary": "ok" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ParallelIn { text: "test".into() }).unwrap().with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "ok");
    assert_eq!(spy.calls().len(), 2);
}

#[tokio::test]
async fn test_same_tool_three_times_drains_queue_serially() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo", json!({ "text": "first" })),
            mock_tool_call("c2", "echo", json!({ "text": "second" })),
            mock_tool_call("c3", "echo", json!({ "text": "third" })),
        ])
        .then_output(json!({ "summary": "3-done" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ParallelIn { text: "test".into() }).unwrap().with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "3-done");
    assert_eq!(spy.calls().len(), 2);
}

#[tokio::test]
async fn test_parallel_plus_serial_queue_combined() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "a" })),
            mock_tool_call("c2", "echo",    json!({ "text": "b" })),
            mock_tool_call("c3", "reverse", json!({ "text": "z" })),
        ])
        .then_output(json!({ "summary": "combo" }));
    let spy = factory.clone();
    let rt = FlowRuntime::new(ParallelIn { text: "test".into() }).unwrap().with_factory(factory);
    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "combo");
    assert_eq!(spy.calls().len(), 2);
}

#[tokio::test]
async fn test_duplicate_call_id_across_different_tools_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "hi" })),
            mock_tool_call("c1", "reverse", json!({ "text": "world" })),
        ]);
    let rt = FlowRuntime::new(ParallelIn { text: "test".into() }).unwrap().with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::DuplicateToolCall { .. }) => {}
        other => panic!("expected DuplicateToolCall, got: {other}"),
    }
}

#[tokio::test]
async fn test_llm_error_on_first_dispatch_propagates_as_agent_error() {
    let factory = ScriptedFactory::new()
        .then_err(ClientError::Llm("upstream rate limited".into()));
    let rt = FlowRuntime::new(LlmFailIn { x: 0 }).unwrap().with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("rate limited"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}

#[tokio::test]
async fn test_llm_error_on_re_dispatch_propagates_as_agent_error() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "hi" }))])
        .then_err(ClientError::Llm("second call failed".into()));
    let rt = FlowRuntime::new(LlmFailIn { x: 0 }).unwrap().with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("second call failed"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}

#[tokio::test]
async fn test_exhausted_scripted_factory_propagates_as_llm_failed() {
    let factory = ScriptedFactory::new();
    let rt = FlowRuntime::new(LlmFailIn { x: 0 }).unwrap().with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("exhausted"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed from exhausted factory, got: {other}"),
    }
}

#[tokio::test]
async fn test_empty_llm_response_propagates_as_llm_failed() {
    let factory = ScriptedFactory::new().then_err(ClientError::EmptyResponse);
    let rt = FlowRuntime::new(LlmFailIn { x: 0 }).unwrap().with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { .. }) => {}
        other => panic!("expected LlmFailed, got: {other}"),
    }
}

#[tokio::test]
async fn test_factory_exhausted_after_tool_results_propagates_as_llm_failed() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "ok" }))]);
    let rt = FlowRuntime::new(LlmFailIn { x: 0 }).unwrap().with_factory(factory);
    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("exhausted"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed after tool result, got: {other}"),
    }
}


