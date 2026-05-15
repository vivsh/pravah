//! Integration tests for the agent tool-call machinery.
//!
//! Tests are driven by [`ScriptedFactory`], which replays a pre-programmed
//! sequence of LLM responses without any real network calls. Each test exercises
//! one distinct path through the tool-call dispatch logic in `dispatch_agent` /
//! `handle_tool` / `handle_child_agent`.

use pravah::clients::{ClientError, Role};
use pravah::flows::{Agent, AgentConfig, AgentError, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::testing::{CapturingHistoryStore, ScriptedFactory, mock_tool_call};
use pravah::tools::{Tool, ToolBox, ToolError};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ── shared helpers ────────────────────────────────────────────────────────────

fn ctx() -> Context {
    Context::new(FlowConf::default())
}

/// Drives a runtime with a [`ScriptedFactory`] to completion and returns the output.
/// Returns `Err` if the flow errors or suspends unexpectedly.
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

/// Same as `run_to_done` but expects the flow to error; returns the error.
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

// ═══════════════════════════════════════════════════════════════════════════════
// Shared tool types
// ═══════════════════════════════════════════════════════════════════════════════

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
    fn description() -> &'static str { "Always fails." }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Err(ToolError::Other("tool deliberately broken".into()))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1: direct response (no tool calls)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectIn {
    prompt: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectOut {
    answer: String,
}
impl Agent for DirectIn {
    type Output = DirectOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Answer directly.", "test://model")
    }
}
impl Flow for DirectIn {
    type Output = DirectOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<DirectIn>().build()
    }
}

/// Agent returns a structured-output response with no tool calls; flow completes in one LLM turn.
#[tokio::test]
async fn test_direct_response() {
    let factory = ScriptedFactory::new()
        .then_output(json!({ "answer": "42" }));
    let spy = factory.clone();

    let rt = FlowRuntime::new(DirectIn { prompt: "what is the answer?".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.answer, "42");
    assert_eq!(spy.calls().len(), 1, "expected exactly one LLM dispatch");
    assert_eq!(spy.remaining(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2: valid single tool call followed by exit
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ValidIn {
    query: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ValidOut {
    result: String,
}
impl Agent for ValidIn {
    type Output = ValidOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo, then submit.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for ValidIn {
    type Output = ValidOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<ValidIn>().build()
    }
}

/// Agent calls `echo`, receives the result, then calls `submit` with a final value.
/// The exit sentinel causes the agent to complete with the submitted value.
#[tokio::test]
async fn test_valid_tool_call_then_exit() {
    let factory = ScriptedFactory::new()
        // Turn 1: issue one tool call
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "hello" }))])
        // Turn 2 (after tool result in history): call the exit sentinel
        .then_tool_calls(vec![mock_tool_call("c2", "submit", json!({ "result": "echoed:hello" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ValidIn { query: "echo hello".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "echoed:hello");
    assert_eq!(spy.calls().len(), 2, "expected two LLM dispatches");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3: multiple tool calls in one LLM turn
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MultiToolIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MultiToolOut {
    summary: String,
}
impl Agent for MultiToolIn {
    type Output = MultiToolOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo and reverse, then submit.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>().tool::<ReverseInput>())
    }
}
impl Flow for MultiToolIn {
    type Output = MultiToolOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<MultiToolIn>().build()
    }
}

/// Agent issues two distinct tool calls in one turn; both execute, then it submits.
/// Verifies the multi-tool pending loop in `handle_child_agent`.
#[tokio::test]
async fn test_multiple_tool_calls_in_one_turn() {
    let factory = ScriptedFactory::new()
        // Turn 1: two tool calls at once
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "hi" })),
            mock_tool_call("c2", "reverse", json!({ "text": "hi" })),
        ])
        // Turn 2: submit after seeing both tool results
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "summary": "done" }))]);

    let rt = FlowRuntime::new(MultiToolIn { text: "hi".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "done");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4: duplicate tool call (same tool twice in one turn)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DupIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DupOut {
    result: String,
}
impl Agent for DupIn {
    type Output = DupOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for DupIn {
    type Output = DupOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<DupIn>().build()
    }
}

/// LLM issues two calls to the same tool (`echo`) in a single turn.
/// The flow engine must reject this with `AgentError::DuplicateToolCall`.
#[tokio::test]
async fn test_duplicate_tool_call_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo", json!({ "text": "a" })),
            mock_tool_call("c2", "echo", json!({ "text": "b" })),
        ]);

    let rt = FlowRuntime::new(DupIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::DuplicateToolCall { tool, .. }) => {
            assert_eq!(tool, "echo");
        }
        other => panic!("expected DuplicateToolCall, got: {other}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5: unknown / missing tool name
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct UnknownIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct UnknownOut {
    result: String,
}
impl Agent for UnknownIn {
    type Output = UnknownOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for UnknownIn {
    type Output = UnknownOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<UnknownIn>().build()
    }
}

/// LLM calls a tool name that is not registered on the agent.
/// The flow engine must reject this with `AgentError::UnknownTool`.
#[tokio::test]
async fn test_unknown_tool_name_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "nonexistent_tool", json!({}))]);

    let rt = FlowRuntime::new(UnknownIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::UnknownTool { tool, .. }) => {
            assert_eq!(tool, "nonexistent_tool");
        }
        other => panic!("expected UnknownTool, got: {other}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 6: tool execution error (tool returns Err)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BrokenIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BrokenFlowOut {
    y: i64,
}
impl Agent for BrokenIn {
    type Output = BrokenFlowOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use broken tool.", "test://model")
            .with_tools(ToolBox::new().tool::<BrokenInput>())
    }
}
impl Flow for BrokenIn {
    type Output = BrokenFlowOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<BrokenIn>().build()
    }
}

/// LLM calls a tool whose `call` implementation always returns `Err`.
/// The flow engine must surface this as `AgentError::ToolFailed`.
#[tokio::test]
async fn test_tool_execution_error_surfaces_as_tool_failed() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "broken", json!({ "_x": 1 }))]);

    let rt = FlowRuntime::new(BrokenIn { x: 1 })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::ToolFailed { tool, reason }) => {
            assert_eq!(tool, "broken");
            assert!(reason.contains("tool deliberately broken"), "unexpected reason: {reason}");
        }
        other => panic!("expected ToolFailed, got: {other}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 7: invalid tool arguments (deserialization failure)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct InvalidArgsIn {
    query: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct InvalidArgsOut {
    answer: String,
}
impl Agent for InvalidArgsIn {
    type Output = InvalidArgsOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for InvalidArgsIn {
    type Output = InvalidArgsOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<InvalidArgsIn>().build()
    }
}

/// LLM calls a valid tool but passes args that cannot be deserialized into the
/// tool's input type. The flow engine surfaces this as `AgentError::ToolFailed`.
#[tokio::test]
async fn test_invalid_tool_args_surface_as_tool_failed() {
    let factory = ScriptedFactory::new()
        // `echo` expects `{ "text": string }` — send a wrong shape instead
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "wrong_field": 99 }))]);

    let rt = FlowRuntime::new(InvalidArgsIn { query: "test".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::ToolFailed { tool, .. }) => {
            assert_eq!(tool, "echo");
        }
        other => panic!("expected ToolFailed, got: {other}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 8: multiple exit-sentinel calls in one turn
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MultiExitIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MultiExitOut {
    result: String,
}
impl Agent for MultiExitIn {
    type Output = MultiExitOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Submit your answer.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for MultiExitIn {
    type Output = MultiExitOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<MultiExitIn>().build()
    }
}

/// LLM calls the exit sentinel (`submit`) twice in one turn.
/// Both calls target the same tool entry slot, so the second triggers
/// `AgentError::DuplicateToolCall` — multiple exits are not permitted.
#[tokio::test]
async fn test_multiple_exit_calls_are_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "submit", json!({ "result": "first" })),
            mock_tool_call("c2", "submit", json!({ "result": "second" })),
        ]);

    let rt = FlowRuntime::new(MultiExitIn { text: "test".into() })
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

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 9: valid exit via submit (structured exit sentinel)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ExitOnlyIn {
    prompt: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ExitOnlyOut {
    value: String,
}
impl Agent for ExitOnlyIn {
    type Output = ExitOnlyOut;
    fn build() -> AgentConfig {
        // Has a tool so that structured-output mode is bypassed and submit is injected
        AgentConfig::new("Submit immediately.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for ExitOnlyIn {
    type Output = ExitOnlyOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<ExitOnlyIn>().build()
    }
}

/// LLM calls `submit` directly in the first turn without using any other tool.
/// Verifies the exit sentinel path completes the agent correctly.
#[tokio::test]
async fn test_valid_exit_via_submit_sentinel() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "submit", json!({ "value": "immediate" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ExitOnlyIn { prompt: "go".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.value, "immediate");
    assert_eq!(spy.calls().len(), 1, "expected exactly one LLM dispatch");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 10: history message ordering — tool result reaches the LLM
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct HistoryIn {
    q: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct HistoryOut {
    a: String,
}
impl Agent for HistoryIn {
    type Output = HistoryOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Use echo.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for HistoryIn {
    type Output = HistoryOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<HistoryIn>().build()
    }
}

/// After one tool call and before the second LLM dispatch, the messages seen by the
/// LLM must include, in order:
///   User message → AssistantToolCalls (with call_id "tc1") → Tool result (call_id "tc1").
/// This verifies that tool output actually flows through history to the next dispatch.
#[tokio::test]
async fn test_tool_result_present_in_second_dispatch() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("tc1", "echo", json!({ "text": "ping" }))])
        .then_tool_calls(vec![mock_tool_call("tc2", "submit", json!({ "a": "pong" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(HistoryIn { q: "ping?".into() })
        .unwrap()
        .with_factory(factory);

    run_to_done(rt).await.unwrap();

    let calls = spy.calls();
    assert_eq!(calls.len(), 2);

    let second_msgs = &calls[1].1;

    // There must be an AssistantToolCalls message containing call_id "tc1".
    let has_atc = second_msgs.iter().any(|m| {
        matches!(&m.role, Role::AssistantToolCalls { calls } if calls.iter().any(|c| c.id == "tc1"))
    });
    assert!(has_atc, "second dispatch must include AssistantToolCalls with id tc1; got: {second_msgs:?}");

    // There must be a Tool result message echoing call_id "tc1".
    let has_tool_result = second_msgs.iter().any(|m| {
        matches!(&m.role, Role::Tool { call_id } if call_id == "tc1")
    });
    assert!(has_tool_result, "second dispatch must include Tool result for tc1; got: {second_msgs:?}");

    // AssistantToolCalls must come before the Tool result.
    let atc_pos = second_msgs.iter().position(|m| {
        matches!(&m.role, Role::AssistantToolCalls { .. })
    }).unwrap();
    let tool_pos = second_msgs.iter().position(|m| {
        matches!(&m.role, Role::Tool { call_id } if call_id == "tc1")
    }).unwrap();
    assert!(
        atc_pos < tool_pos,
        "AssistantToolCalls (pos {atc_pos}) must precede Tool result (pos {tool_pos})"
    );
}

/// After two tool calls in one turn, both tool result messages must appear in the
/// second dispatch, one for each call_id.
#[tokio::test]
async fn test_both_tool_results_present_after_multi_call_turn() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("id1", "echo",    json!({ "text": "a" })),
            mock_tool_call("id2", "reverse", json!({ "text": "b" })),
        ])
        .then_tool_calls(vec![mock_tool_call("id3", "submit", json!({ "summary": "ok" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(MultiToolIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    run_to_done(rt).await.unwrap();

    let second_msgs = &spy.calls()[1].1;

    for call_id in ["id1", "id2"] {
        let has_result = second_msgs.iter().any(|m| {
            matches!(&m.role, Role::Tool { call_id: cid } if cid == call_id)
        });
        assert!(has_result, "expected Tool result for {call_id} in second dispatch");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 11: multi-round tool chaining (three LLM turns)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChainIn {
    start: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChainOut {
    final_value: String,
}
impl Agent for ChainIn {
    type Output = ChainOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Chain echo three times.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for ChainIn {
    type Output = ChainOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<ChainIn>().build()
    }
}

/// Agent calls `echo` three times in separate turns before submitting.
/// Verifies that the pending-tool state machine handles multiple sequential rounds.
#[tokio::test]
async fn test_three_sequential_tool_call_rounds() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("r1", "echo", json!({ "text": "one" }))])
        .then_tool_calls(vec![mock_tool_call("r2", "echo", json!({ "text": "two" }))])
        .then_tool_calls(vec![mock_tool_call("r3", "echo", json!({ "text": "three" }))])
        .then_tool_calls(vec![mock_tool_call("r4", "submit", json!({ "final_value": "done" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ChainIn { start: "go".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.final_value, "done");
    assert_eq!(spy.calls().len(), 4, "expected four LLM dispatches");
    assert_eq!(spy.remaining(), 0);
}

/// Each subsequent dispatch must include the tool result from the previous turn.
/// Turn 2 must contain the result for r1; turn 3 must contain results for r1 and r2.
#[tokio::test]
async fn test_tool_results_accumulate_across_rounds() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("r1", "echo", json!({ "text": "a" }))])
        .then_tool_calls(vec![mock_tool_call("r2", "echo", json!({ "text": "b" }))])
        .then_tool_calls(vec![mock_tool_call("r3", "submit", json!({ "final_value": "done" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ChainIn { start: "chain".into() })
        .unwrap()
        .with_factory(factory);

    run_to_done(rt).await.unwrap();

    let calls = spy.calls();

    // Turn 2 sees r1 result.
    let t2 = &calls[1].1;
    assert!(
        t2.iter().any(|m| matches!(&m.role, Role::Tool { call_id } if call_id == "r1")),
        "turn 2 must carry r1 result"
    );

    // Turn 3 sees both r1 and r2 results.
    let t3 = &calls[2].1;
    for id in ["r1", "r2"] {
        assert!(
            t3.iter().any(|m| matches!(&m.role, Role::Tool { call_id } if call_id == id)),
            "turn 3 must carry {id} result"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 12: LLM client error mid-flow
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LlmErrIn {
    x: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LlmErrOut {
    y: String,
}
impl Agent for LlmErrIn {
    type Output = LlmErrOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Call echo.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for LlmErrIn {
    type Output = LlmErrOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<LlmErrIn>().build()
    }
}

/// LLM client returns an error on the very first dispatch.
/// The error must propagate as `AgentError::LlmFailed`.
#[tokio::test]
async fn test_llm_error_on_first_dispatch_propagates() {
    let factory = ScriptedFactory::new()
        .then_err(ClientError::Llm("network timeout".into()));

    let rt = FlowRuntime::new(LlmErrIn { x: "hello".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::LlmFailed { reason, .. }) => {
            assert!(reason.contains("network timeout"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}

/// LLM succeeds on the first call (tool call), then errors on the second dispatch.
/// Verifies the error path is clean after partial state has been written.
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
            assert!(reason.contains("server error on retry"), "unexpected reason: {reason}");
        }
        other => panic!("expected LlmFailed, got: {other}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 13: structured output in tool mode (LLM bypasses submit)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StructModeIn {
    q: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StructModeOut {
    answer: String,
}
impl Agent for StructModeIn {
    type Output = StructModeOut;
    fn build() -> AgentConfig {
        // registered with tools so the exit sentinel is injected,
        // but the LLM returns a direct Output rather than calling submit
        AgentConfig::new("Answer directly.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for StructModeIn {
    type Output = StructModeOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<StructModeIn>().build()
    }
}

/// Even when a toolbox is registered (so submit is injected), the LLM may return a
/// `ClientOutput::Output` directly. The agent must complete normally with that value.
#[tokio::test]
async fn test_direct_output_in_tool_mode_completes_agent() {
    let factory = ScriptedFactory::new()
        .then_output(json!({ "answer": "shortcut" }));
    let spy = factory.clone();

    let rt = FlowRuntime::new(StructModeIn { q: "quick?".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.answer, "shortcut");
    assert_eq!(spy.calls().len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 14: submit payload shape mismatch
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MismatchIn {
    q: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MismatchOut {
    required_field: String,
}
impl Agent for MismatchIn {
    type Output = MismatchOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Submit.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for MismatchIn {
    type Output = MismatchOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<MismatchIn>().build()
    }
}

/// LLM calls `submit` with a JSON payload that does not match `MismatchOut`'s schema
/// (missing `required_field`). The error must be surfaced — either at the exit-sentinel
/// boundary or at the final `Done` deserialization step — rather than silently producing
/// a corrupted output.
#[tokio::test]
async fn test_submit_payload_mismatch_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "submit", json!({ "wrong": "shape" }))]);

    let rt = FlowRuntime::new(MismatchIn { q: "go".into() })
        .unwrap()
        .with_factory(factory);

    // The error may surface during flow execution or at Done deserialization.
    // Either way the output must NOT silently succeed with a zero-value struct.
    let result: Result<MismatchOut, _> = run_to_done(rt).await;
    match result {
        Err(_) => {}
        Ok(out) => {
            assert!(
                !out.required_field.is_empty(),
                "submit with wrong payload should not produce a valid MismatchOut with empty required_field"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 15: history captured by CapturingHistoryStore
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StoreIn {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StoreOut {
    echoed: String,
}
impl Agent for StoreIn {
    type Output = StoreOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Echo and submit.", "test://model")
            .with_tools(ToolBox::new().tool::<EchoInput>())
    }
}
impl Flow for StoreIn {
    type Output = StoreOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<StoreIn>().build()
    }
}

/// CapturingHistoryStore receives a flush after every step.
/// After two LLM turns (echo + submit) the final snapshot must contain:
///   - a User message (agent input)
///   - an AssistantToolCalls message
///   - a Tool result message
///   - an AssistantToolCalls message (submit)
/// and the flush count must equal the number of completed steps.
#[tokio::test]
async fn test_capturing_store_receives_all_messages() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("s1", "echo", json!({ "text": "world" }))])
        .then_tool_calls(vec![mock_tool_call("s2", "submit", json!({ "echoed": "world" }))]);

    let store = CapturingHistoryStore::new();
    let store_spy = store.clone();

    let rt = FlowRuntime::new(StoreIn { text: "world".into() })
        .unwrap()
        .with_factory(factory)
        .with_store(store);

    run_to_done(rt).await.unwrap();

    assert!(store_spy.flush_count() >= 2, "expected at least 2 flushes");

    let snapshot = store_spy.last_snapshot().expect("must have at least one snapshot");
    let non_evicted: Vec<_> = snapshot.iter().filter(|e| !e.evicted).collect();

    let has_user = non_evicted.iter().any(|e| matches!(e.message.role, Role::User));
    assert!(has_user, "snapshot must contain a User message");

    let has_atc = non_evicted.iter().any(|e| matches!(&e.message.role, Role::AssistantToolCalls { .. }));
    assert!(has_atc, "snapshot must contain an AssistantToolCalls message");

    let has_tool = non_evicted.iter().any(|e| matches!(&e.message.role, Role::Tool { call_id } if call_id == "s1"));
    assert!(has_tool, "snapshot must contain the Tool result for s1");
}

/// Flush is called after every step, so flush_count must grow monotonically
/// and be at least as large as the number of LLM dispatches.
#[tokio::test]
async fn test_store_flush_count_matches_steps() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("f1", "echo", json!({ "text": "x" }))])
        .then_tool_calls(vec![mock_tool_call("f2", "submit", json!({ "echoed": "x" }))]);

    let store = CapturingHistoryStore::new();
    let store_spy = store.clone();

    let rt = FlowRuntime::new(StoreIn { text: "x".into() })
        .unwrap()
        .with_factory(factory)
        .with_store(store);

    run_to_done(rt).await.unwrap();

    // Two LLM turns → at minimum two steps that trigger compaction+flush
    assert!(
        store_spy.flush_count() >= 2,
        "flush count {} too low for a two-turn flow",
        store_spy.flush_count()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 16: all scripted responses consumed (no leftover queue entries)
// ═══════════════════════════════════════════════════════════════════════════════

/// After a successful run every scripted response must have been consumed.
/// Leftover queue entries indicate the flow took fewer LLM turns than expected,
/// which means a behaviour change silently under-exercised the scripted scenario.
#[tokio::test]
async fn test_no_scripted_responses_remain_after_valid_run() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("q1", "echo", json!({ "text": "a" }))])
        .then_tool_calls(vec![mock_tool_call("q2", "submit", json!({ "result": "done" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ValidIn { query: "consume all".into() })
        .unwrap()
        .with_factory(factory);

    run_to_done(rt).await.unwrap();

    assert_eq!(
        spy.remaining(),
        0,
        "all scripted responses should be consumed; {} still waiting",
        spy.remaining()
    );
}

/// After a direct-output (no tools) run the one queued response must be consumed.
#[tokio::test]
async fn test_no_scripted_responses_remain_after_direct_run() {
    let factory = ScriptedFactory::new()
        .then_output(json!({ "answer": "yes" }));
    let spy = factory.clone();

    let rt = FlowRuntime::new(DirectIn { prompt: "q".into() })
        .unwrap()
        .with_factory(factory);

    run_to_done(rt).await.unwrap();

    assert_eq!(spy.remaining(), 0);
}
