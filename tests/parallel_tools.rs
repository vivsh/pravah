//! Integration tests for parallel and queued tool dispatch.

use pravah::clients::{ClientError, Role};
use pravah::flows::{Agent, AgentConfig, AgentError, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::tools::{Tool, ToolBox, ToolError};
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


#[derive(Debug, Deserialize, JsonSchema)]
struct EchoInput {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EchoOutput {
    echoed: String,
}
impl Tool for EchoInput {
    type Output = EchoOutput;
    fn name() -> &'static str { "echo" }
    fn description() -> &'static str { "Echoes the input text." }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Ok(EchoOutput { echoed: self.text })
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReverseInput {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReverseOutput {
    reversed: String,
}
impl Tool for ReverseInput {
    type Output = ReverseOutput;
    fn name() -> &'static str { "reverse" }
    fn description() -> &'static str { "Reverses the input text." }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Ok(ReverseOutput { reversed: self.text.chars().rev().collect() })
    }
}
#[derive(Debug, Deserialize, JsonSchema)]
struct BrokenInput {
    _x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BrokenOutput;
impl Tool for BrokenInput {
    type Output = BrokenOutput;
    fn name() -> &'static str { "broken" }
    fn description() -> &'static str { "Always fails with a non-fatal error." }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Err(ToolError::Other("tool deliberately broken".into()))
    }
}
#[derive(Debug, Deserialize, JsonSchema)]
struct Broken2Input {
    _y: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Broken2Output;
impl Tool for Broken2Input {
    type Output = Broken2Output;
    fn name() -> &'static str { "broken2" }
    fn description() -> &'static str { "A second tool that always fails non-fatally." }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Err(ToolError::Other("broken2 deliberately failed".into()))
    }
}
/// This causes the flow engine to abort with `AgentError::ToolFailed`.
#[derive(Debug, Deserialize, JsonSchema)]
struct FatalEscapeInput {
    _x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FatalEscapeOutput;
impl Tool for FatalEscapeInput {
    type Output = FatalEscapeOutput;
    fn name() -> &'static str { "fatal_escape" }
    fn description() -> &'static str { "Simulates a path-escape security violation." }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Err(ToolError::PathEscape("../secret".to_owned()))
    }
}


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ParallelIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ParallelOut {
    summary: String,
}
impl Agent for ParallelIn {
    type Output = ParallelOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo and reverse, then submit.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>().tool::<ReverseInput>())
    }
}
impl Flow for ParallelIn {
    type Output = ParallelOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<ParallelIn>().build()
    }
}
/// Verifies that independent tool slots execute concurrently and both results reach history.
#[tokio::test]
async fn test_two_distinct_tools_run_in_parallel() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "hello" })),
            mock_tool_call("c2", "reverse", json!({ "text": "world" })),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "summary": "done" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "done");
    assert_eq!(spy.calls().len(), 2, "expected initial dispatch + one re-dispatch");
}
/// All three results (two successes and one `{\"error\":…}`) must land in history
#[tokio::test]
async fn test_three_distinct_tools_one_non_fatal() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "hi" })),
            mock_tool_call("c2", "reverse", json!({ "text": "ok" })),
            mock_tool_call("c3", "broken",  json!({ "_x": 0 })),
        ])
        .then_tool_calls(vec![mock_tool_call("c4", "submit", json!({ "result": "all three" }))]);

    let rt = FlowRuntime::new(ThreeToolIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "all three");
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ThreeToolIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ThreeToolOut {
    result: String,
}
impl Agent for ThreeToolIn {
    type Output = ThreeToolOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo, reverse and broken.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>().tool::<ReverseInput>().tool::<BrokenInput>())
    }
}
impl Flow for ThreeToolIn {
    type Output = ThreeToolOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<ThreeToolIn>().build()
    }
}
/// tool-result immediately, the known call executes normally. Both results arrive in
#[tokio::test]
async fn test_unknown_tool_alongside_known_gets_error_result() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",           json!({ "text": "hi" })),
            mock_tool_call("c2", "no_such_tool",   json!({})),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "summary": "recovered" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "recovered");
    assert_eq!(spy.calls().len(), 2, "unknown tool should trigger a re-dispatch");
}
/// synchronously inside `dispatch_agent`, so `active` stays empty and a
#[tokio::test]
async fn test_all_unknown_tools_in_one_turn_re_dispatch_immediately() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "ghost1", json!({})),
            mock_tool_call("c2", "ghost2", json!({})),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "summary": "ok" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "ok");
    assert_eq!(spy.calls().len(), 2);
}


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MixedIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MixedOut {
    summary: String,
}
impl Agent for MixedIn {
    type Output = MixedOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo and broken.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>().tool::<BrokenInput>())
    }
}
impl Flow for MixedIn {
    type Output = MixedOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<MixedIn>().build()
    }
}
/// a success value and a `{\"error\":…}` object — are placed in history, and
#[tokio::test]
async fn test_non_fatal_error_alongside_success_in_parallel() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",   json!({ "text": "good" })),
            mock_tool_call("c2", "broken", json!({ "_x": 1 })),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "summary": "partial" }))]);

    let rt = FlowRuntime::new(MixedIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "partial");
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AllFailIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AllFailOut {
    result: String,
}
impl Agent for AllFailIn {
    type Output = AllFailOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use broken and broken2.", "test://model")
            .with_tools(ToolBox::new().tool::<BrokenInput>().tool::<Broken2Input>())
    }
}
impl Flow for AllFailIn {
    type Output = AllFailOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<AllFailIn>().build()
    }
}
/// the agent re-dispatches once and can still submit successfully.
#[tokio::test]
async fn test_all_tools_fail_non_fatally_in_parallel() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "broken",  json!({ "_x": 1 })),
            mock_tool_call("c2", "broken2", json!({ "_y": 2 })),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "result": "all-failed" }))]);

    let rt = FlowRuntime::new(AllFailIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "all-failed");
}
/// tool across consecutive turns and still reach a successful submit.
#[tokio::test]
async fn test_non_fatal_errors_persist_across_multiple_rounds() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "broken", json!({ "_x": 1 }))])
        .then_tool_calls(vec![mock_tool_call("c2", "broken", json!({ "_x": 2 }))])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "result": "gave up" }))]);

    let rt = FlowRuntime::new(RecoveryIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "gave up");
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RecoveryIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RecoveryOut {
    result: String,
}
impl Agent for RecoveryIn {
    type Output = RecoveryOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use broken.", "test://model")
            .with_tools(ToolBox::new().tool::<BrokenInput>())
    }
}
impl Flow for RecoveryIn {
    type Output = RecoveryOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<RecoveryIn>().build()
    }
}


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FatalIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FatalOut {
    result: String,
}
impl Agent for FatalIn {
    type Output = FatalOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo and fatal_escape.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>().tool::<FatalEscapeInput>())
    }
}
impl Flow for FatalIn {
    type Output = FatalOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<FatalIn>().build()
    }
}
/// It propagates immediately as `AgentError::ToolFailed`, aborting the flow.
#[tokio::test]
async fn test_fatal_tool_error_aborts_flow() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "fatal_escape", json!({ "_x": 0 }))]);

    let rt = FlowRuntime::new(FatalIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::ToolFailed { tool, .. }) => {
            assert_eq!(tool, "fatal_escape");
        }
        other => panic!("expected ToolFailed, got: {other}"),
    }
}
/// The fatal error is non-recoverable regardless of the other tool's outcome.
#[tokio::test]
async fn test_fatal_tool_in_parallel_aborts_flow_regardless_of_other_tools() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "fatal_escape", json!({ "_x": 0 })),
            mock_tool_call("c2", "echo",         json!({ "text": "hi" })),
        ]);

    let rt = FlowRuntime::new(FatalIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::ToolFailed { tool, .. }) => {
            assert_eq!(tool, "fatal_escape");
        }
        other => panic!("expected ToolFailed from fatal_escape, got: {other}"),
    }
}
/// behind the first. All three drain serially before a single re-dispatch occurs.
#[tokio::test]
async fn test_same_tool_three_times_drains_queue_serially() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo", json!({ "text": "first" })),
            mock_tool_call("c2", "echo", json!({ "text": "second" })),
            mock_tool_call("c3", "echo", json!({ "text": "third" })),
        ])
        .then_tool_calls(vec![mock_tool_call("c4", "submit", json!({ "summary": "3-done" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "3-done");
    assert_eq!(spy.calls().len(), 2, "all three queue slots drain before re-dispatch");
}
/// fail; each produces an error tool-result. After both drain the queue, the agent
#[tokio::test]
async fn test_queued_tool_both_calls_fail_non_fatally() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "broken", json!({ "_x": 1 })),
            mock_tool_call("c2", "broken", json!({ "_x": 2 })),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "result": "both-broken" }))]);

    let rt = FlowRuntime::new(RecoveryIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "both-broken");
}
/// queued calls while the other slot has one call. All results land in history
#[tokio::test]
async fn test_parallel_plus_serial_queue_combined() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "a" })),
            mock_tool_call("c2", "echo",    json!({ "text": "b" })),
            mock_tool_call("c3", "reverse", json!({ "text": "z" })),
        ])
        .then_tool_calls(vec![mock_tool_call("c4", "submit", json!({ "summary": "combo" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "combo");
    assert_eq!(spy.calls().len(), 2);
}
/// `DuplicateToolCall` before either tool executes. The dedup check fires first,
#[tokio::test]
async fn test_duplicate_call_id_across_different_tools_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "hi" })),
            mock_tool_call("c1", "reverse", json!({ "text": "world" })),
        ]);

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::DuplicateToolCall { .. }) => {}
        other => panic!("expected DuplicateToolCall, got: {other}"),
    }
}
/// The exit sentinel path has its own guard (`exit_id == node.exit`) that fires
#[tokio::test]
async fn test_duplicate_call_id_on_submit_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "submit", json!({ "summary": "first" })),
            mock_tool_call("c1", "submit", json!({ "summary": "second" })),
        ]);

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::DuplicateToolCall { .. }) => {}
        other => panic!("expected DuplicateToolCall for submit, got: {other}"),
    }
}
/// guard (`exit_id == node.exit`) fires after call_id dedup, catching this case.
#[tokio::test]
async fn test_two_submit_calls_with_distinct_ids_are_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "submit", json!({ "summary": "first" })),
            mock_tool_call("c2", "submit", json!({ "summary": "second" })),
        ]);

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::DuplicateToolCall { tool, .. }) => {
            assert_eq!(tool, "submit");
        }
        other => panic!("expected DuplicateToolCall for submit, got: {other}"),
    }
}
/// The deserialize error is non-fatal: the call gets an error tool-result, while the
#[tokio::test]
async fn test_invalid_args_on_one_parallel_tool_becomes_error_result() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "good" })),
            mock_tool_call("c2", "reverse", json!({ "bad_field": 99 })),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "summary": "recovered" }))]);

    let rt = FlowRuntime::new(ParallelIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "recovered");
}


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LlmFailIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LlmFailOut {
    result: String,
}
impl Agent for LlmFailIn {
    type Output = LlmFailOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for LlmFailIn {
    type Output = LlmFailOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<LlmFailIn>().build()
    }
}
/// The failure reason string is preserved in the error.
#[tokio::test]
async fn test_llm_error_on_first_dispatch_propagates_as_agent_error() {
    let factory = ScriptedFactory::new()
        .then_err(ClientError::Llm("upstream rate limited".into()));

    let rt = FlowRuntime::new(LlmFailIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("rate limited"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}
/// propagates as `AgentError::LlmFailed`. Mid-flow LLM failures are not swallowed.
#[tokio::test]
async fn test_llm_error_on_re_dispatch_propagates_as_agent_error() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "hi" }))])
        .then_err(ClientError::Llm("second call failed".into()));

    let rt = FlowRuntime::new(LlmFailIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("second call failed"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}
/// "response queue exhausted" error, which propagates as `AgentError::LlmFailed`.
#[tokio::test]
async fn test_exhausted_scripted_factory_propagates_as_llm_failed() {
    let factory = ScriptedFactory::new();

    let rt = FlowRuntime::new(LlmFailIn { x: 0 })
        .unwrap()
        .with_factory(factory);

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
    let factory = ScriptedFactory::new()
        .then_err(ClientError::EmptyResponse);

    let rt = FlowRuntime::new(LlmFailIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { .. }) => {}
        other => panic!("expected LlmFailed, got: {other}"),
    }
}
/// a second dispatch after processing queued tool results).
#[tokio::test]
async fn test_factory_exhausted_after_tool_results_propagates_as_llm_failed() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "ok" }))]);

    let rt = FlowRuntime::new(LlmFailIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("exhausted"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed after tool result, got: {other}"),
    }
}
/// `ToolOutput` entry for every call_id issued, in the order the LLM dispatched
#[tokio::test]
async fn test_history_contains_tool_result_for_every_call_id() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",      json!({ "text": "hi" })),
            mock_tool_call("c2", "ghost",     json!({})),
            mock_tool_call("c3", "broken",    json!({ "_x": 1 })),
        ])
        .then_tool_calls(vec![mock_tool_call("c4", "submit", json!({ "summary": "ok" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ThreeWithMixedIn { x: 0 })
        .unwrap()
        .with_factory(factory);

    run_to_done(rt).await.unwrap();

    let calls = spy.calls();
    assert_eq!(calls.len(), 2);
    let second_call_messages = &calls[1].1;
    let tool_output_count = second_call_messages
        .iter()
        .filter(|m| matches!(m.role, Role::Tool { .. }))
        .count();
    assert_eq!(tool_output_count, 3, "every call_id must produce a ToolOutput message");
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ThreeWithMixedIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ThreeWithMixedOut {
    summary: String,
}
impl Agent for ThreeWithMixedIn {
    type Output = ThreeWithMixedOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo, unknown, and broken.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>().tool::<BrokenInput>())
    }
}
impl Flow for ThreeWithMixedIn {
    type Output = ThreeWithMixedOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<ThreeWithMixedIn>().build()
    }
}
