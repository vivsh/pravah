//! Integration tests for agent tool-call dispatch.

use pravah::clients::{ClientError, Role};
use pravah::flows::{Agent, AgentConfig, AgentError, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep, PhaseKind};
use pravah::testing::{CapturingHistoryStore, ScriptedFactory, mock_tool_call};
use pravah::tools::{Tool, ToolBox, ToolError, ToolOutput};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;


fn ctx() -> Context {
    Context::new(FlowConf::default())
}
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


/// Echoes the input text.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(rename = "echo")]
struct EchoInput {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EchoOutput {
    echoed: String,
}
impl Tool for EchoInput {
    type Output = EchoOutput;
    async fn call(self, _ctx: Context) -> Result<ToolOutput<Self::Output>, ToolError> {
        Ok(ToolOutput::plain(EchoOutput { echoed: self.text }))
    }
}

/// Reverses the input text.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(rename = "reverse")]
struct ReverseInput {
    text: String,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReverseOutput {
    reversed: String,
}
impl Tool for ReverseInput {
    type Output = ReverseOutput;
    async fn call(self, _ctx: Context) -> Result<ToolOutput<Self::Output>, ToolError> {
        Ok(ToolOutput::plain(ReverseOutput { reversed: self.text.chars().rev().collect() }))
    }
}

/// Always fails.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(rename = "broken")]
struct BrokenInput {
    _x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BrokenOutput;
impl Tool for BrokenInput {
    type Output = BrokenOutput;
    async fn call(self, _ctx: Context) -> Result<ToolOutput<Self::Output>, ToolError> {
        Err(ToolError::Other("tool deliberately broken".into()))
    }
}


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
/// The exit sentinel causes the agent to complete with the submitted value.
#[tokio::test]
async fn test_valid_tool_call_then_exit() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "hello" }))])
        .then_tool_calls(vec![mock_tool_call("c2", "submit", json!({ "result": "echoed:hello" }))]);
    let spy = factory.clone();

    let rt = FlowRuntime::new(ValidIn { query: "echo hello".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "echoed:hello");
    assert_eq!(spy.calls().len(), 2, "expected two LLM dispatches");
}
#[tokio::test]
async fn test_inspector_tracks_tool_turns() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "text": "hello" }))])
        .then_tool_calls(vec![mock_tool_call("c2", "submit", json!({ "result": "echoed:hello" }))]);

    let mut rt = FlowRuntime::new(ValidIn { query: "echo hello".into() })
        .unwrap()
        .with_factory(factory);

    let inspector = rt.inspector();
    assert_eq!(inspector.depth(), 1);
    let top = inspector.top_frame().expect("root frame should exist");
    assert_eq!(top.phase, PhaseKind::None);
    assert_eq!(top.callable_entry, "ValidIn");
    assert!(top.locals.iter().any(|local| local.name == "ValidIn"));
    let root_local = top
        .locals
        .iter()
        .find(|local| local.name == "ValidIn")
        .expect("root local should be name-resolved");
    assert_eq!(inspector.name_of(root_local.node_id), Some("ValidIn"));
    assert_eq!(inspector.history().entries().len(), 1);

    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    let inspector = rt.inspector();
    assert_eq!(inspector.depth(), 2);
    assert_eq!(
        inspector.top_frame().expect("agent frame should exist").phase,
        PhaseKind::Entry
    );

    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    let inspector = rt.inspector();
    let top = inspector.top_frame().expect("pending tool frame should exist");
    match &top.phase {
        PhaseKind::PendingTool {
            active_calls,
            waiting_count,
        } => {
            assert_eq!(active_calls, &vec!["echo".to_owned()]);
            assert_eq!(*waiting_count, 0);
        }
        other => panic!("expected pending tool phase, got {other:?}"),
    }
    let tool_local = top
        .locals
        .iter()
        .find(|local| local.name.starts_with("ValidIn::"))
        .expect("tool local should be visible while pending");
    assert_eq!(inspector.name_of(tool_local.node_id), Some(tool_local.name));
    assert_eq!(inspector.history().entries().len(), 3);

    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    let inspector = rt.inspector();
    assert_eq!(
        inspector.top_frame().expect("agent frame should still exist").phase,
        PhaseKind::Dispatch
    );
    let entries = inspector.history().entries();
    assert!(matches!(entries.last().map(|entry| &entry.message.role), Some(Role::Tool { call_id }) if call_id == "c1"));

    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));
    let step = rt.next(ctx()).await.unwrap();
    match step {
        FlowStep::Done(out) => assert_eq!(out.result, "echoed:hello"),
        other => panic!("expected completion, got {other:?}"),
    }

    let inspector = rt.inspector();
    assert_eq!(inspector.depth(), 0);
    assert!(inspector.top_frame().is_none());
    assert!(!inspector.is_suspended());
    assert_eq!(inspector.suspension_type(), None);
}


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
/// Verifies the multi-tool pending loop in `handle_child_agent`.
#[tokio::test]
async fn test_multiple_tool_calls_in_one_turn() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo",    json!({ "text": "hi" })),
            mock_tool_call("c2", "reverse", json!({ "text": "hi" })),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "summary": "done" }))]);

    let rt = FlowRuntime::new(MultiToolIn { text: "hi".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.summary, "done");
}


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
/// Both must execute serially (the second is queued behind the first) and then the agent submits.
#[tokio::test]
async fn test_same_tool_twice_runs_serially() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo", json!({ "text": "a" })),
            mock_tool_call("c2", "echo", json!({ "text": "b" })),
        ])
        .then_tool_calls(vec![mock_tool_call("c3", "submit", json!({ "result": "done" }))]);

    let rt = FlowRuntime::new(DupIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "done");
}
/// The flow engine must reject this with `AgentError::DuplicateToolCall`.
#[tokio::test]
async fn test_duplicate_call_id_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("c1", "echo", json!({ "text": "a" })),
            mock_tool_call("c1", "echo", json!({ "text": "b" })),
        ]);

    let rt = FlowRuntime::new(DupIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let err = run_to_err(rt).await;
    match err {
        FlowError::Agent(AgentError::DuplicateToolCall { .. }) => {}
        other => panic!("expected DuplicateToolCall, got: {other}"),
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
/// The error is surfaced as a tool-result so the LLM can recover; the flow must not crash.
#[tokio::test]
async fn test_unknown_tool_name_becomes_error_result() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "nonexistent_tool", json!({}))])
        .then_tool_calls(vec![mock_tool_call("c2", "submit", json!({ "result": "recovered" }))]);

    let rt = FlowRuntime::new(UnknownIn { text: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.result, "recovered");
}


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
/// The error is surfaced as a tool-result so the LLM can recover; the flow must not crash.
#[tokio::test]
async fn test_non_fatal_tool_error_becomes_tool_result() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "broken", json!({ "_x": 1 }))])
        .then_tool_calls(vec![mock_tool_call("c2", "submit", json!({ "y": 42 }))]);

    let rt = FlowRuntime::new(BrokenIn { x: 1 })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.y, 42);
}


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
/// The deserialization error is non-fatal: surfaced as a tool-result so the LLM can recover.
#[tokio::test]
async fn test_invalid_tool_args_become_error_result() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "echo", json!({ "wrong_field": 99 }))])
        .then_tool_calls(vec![mock_tool_call("c2", "submit", json!({ "answer": "recovered" }))]);

    let rt = FlowRuntime::new(InvalidArgsIn { query: "test".into() })
        .unwrap()
        .with_factory(factory);

    let out = run_to_done(rt).await.unwrap();
    assert_eq!(out.answer, "recovered");
}


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
/// Both calls target the same tool entry slot, so the second triggers
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
/// LLM must include, in order:
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

    let has_atc = second_msgs.iter().any(|m| {
        matches!(&m.role, Role::AssistantToolCalls { calls } if calls.iter().any(|c| c.id == "tc1"))
    });
    assert!(has_atc, "second dispatch must include AssistantToolCalls with id tc1; got: {second_msgs:?}");

    let has_tool_result = second_msgs.iter().any(|m| {
        matches!(&m.role, Role::Tool { call_id } if call_id == "tc1")
    });
    assert!(has_tool_result, "second dispatch must include Tool result for tc1; got: {second_msgs:?}");

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

    let t2 = &calls[1].1;
    assert!(
        t2.iter().any(|m| matches!(&m.role, Role::Tool { call_id } if call_id == "r1")),
        "turn 2 must carry r1 result"
    );

    let t3 = &calls[2].1;
    for id in ["r1", "r2"] {
        assert!(
            t3.iter().any(|m| matches!(&m.role, Role::Tool { call_id } if call_id == id)),
            "turn 3 must carry {id} result"
        );
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
/// (missing `required_field`). The error must be surfaced — either at the exit-sentinel
/// a corrupted output.
#[tokio::test]
async fn test_submit_payload_mismatch_is_rejected() {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call("c1", "submit", json!({ "wrong": "shape" }))]);

    let rt = FlowRuntime::new(MismatchIn { q: "go".into() })
        .unwrap()
        .with_factory(factory);

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
/// After two LLM turns (echo + submit) the final snapshot must contain:
///   - an AssistantToolCalls message
///   - an AssistantToolCalls message (submit)
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

    assert!(
        store_spy.flush_count() >= 2,
        "flush count {} too low for a two-turn flow",
        store_spy.flush_count()
    );
}
/// Leftover queue entries indicate the flow took fewer LLM turns than expected,
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
