//! Integration tests for `FlowBuilder::flow()` (nested flow graph inlining)
//! and `FlowRuntime::snapshot()` / `FlowRuntime::from_snapshot()`.
//!
//! Covers:
//! - Work-only nested flows: multi-step inner flow inlined into outer graph
//! - Either inside nested flow: branch state-node name rewriting
//! - Fork+join inside nested flow: fork child state-node name rewriting
//! - Agent inside nested flow: AgentInfo correctly forwarded through rename
//! - Error cases: duplicate entry, inner build failure
//! - Snapshot round-trip: serialize → deserialize → continue to Done

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use either::Either;
use pravah::clients::{
    Client, ClientError, ClientFactory, ClientOptions, ClientOutput, ClientResponse, Provider,
    ToolCall,
};
use pravah::commons::Agent;
use pravah::flows::{AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, FlowSnapshot, RunOut};
use pravah::tools::{Tool, ToolBox, ToolError};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ── Mock client infrastructure ────────────────────────────────────────────────

struct MockHandle {
    responses: Arc<Mutex<VecDeque<ClientResponse>>>,
}

#[async_trait]
impl Client for MockHandle {
    async fn execute(
        &self,
        _messages: &[pravah::clients::Message],
    ) -> Result<ClientResponse, ClientError> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| ClientError::Llm("mock: response queue exhausted".into()))
    }
}

struct MockFactory {
    responses: Arc<Mutex<VecDeque<ClientResponse>>>,
}

impl MockFactory {
    fn new(responses: Vec<ClientResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into())),
        }
    }
}

impl ClientFactory for MockFactory {
    fn create(&self, _url: &str, _opts: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
        Ok(Box::new(MockHandle {
            responses: Arc::clone(&self.responses),
        }))
    }
}

fn resp(val: serde_json::Value) -> ClientResponse {
    ClientResponse::new(Provider::OpenAi, ClientOutput::Output(val))
}

fn tool_resp(calls: Vec<ToolCall>) -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::ToolCalls {
            thought: None,
            calls,
        },
    )
}

fn make_call(name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: format!("id-{name}"),
        name: name.to_string(),
        args,
        thought_signatures: None,
    }
}

fn ctx() -> Context {
    Context::new(FlowConf {
        working_dir: Some(std::env::temp_dir()),
        ..Default::default()
    })
}

macro_rules! run_to_done {
    ($rt:expr) => {{
        let c = ctx();
        loop {
            match $rt.next(c.clone()).await.expect("next() failed") {
                RunOut::Continue => {}
                RunOut::Done(v) => break v,
                RunOut::Suspend { .. } => panic!("unexpected suspension"),
            }
        }
    }};
}

// ── Nested flow — work only ───────────────────────────────────────────────────
//
// Inner flow (NwInner):  NwInner → work(v*2) → NwMid → work(v+10) → NwInnerOut
// Outer flow (NwOuter):  NwOuter → work(copy) → NwInner
//                                                       ╰─ flow::<NwInner> ─╯
//                               → NwInnerOut → work(v+1) → NwFinal

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NwOuter {
    v: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NwInner {
    v: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NwMid {
    v: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NwInnerOut {
    v: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NwFinal {
    v: i32,
}

impl Flow for NwInner {
    type Output = NwInnerOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work::<NwInner, NwMid, _, _>(|i, _| async move { Ok(NwMid { v: i.v * 2 }) })
            .work::<NwMid, NwInnerOut, _, _>(|m, _| async move { Ok(NwInnerOut { v: m.v + 10 }) })
            .build()
    }
}

impl Flow for NwOuter {
    type Output = NwFinal;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work::<NwOuter, NwInner, _, _>(|o, _| async move { Ok(NwInner { v: o.v }) })
            .flow::<NwInner>()
            .work::<NwInnerOut, NwFinal, _, _>(|i, _| async move { Ok(NwFinal { v: i.v + 1 }) })
            .build()
    }
}

/// Verifies the complete step sequence of a work-only nested flow.
/// NwOuter(v=5) → NwInner(5) → NwMid(10) → NwInnerOut(20) → NwFinal(21)
#[tokio::test]
async fn nested_work_flow_all_steps_complete() {
    let mut rt = FlowRuntime::new(NwOuter { v: 5 }).unwrap();

    // outer work: NwOuter → NwInner
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // sub-flow node: push frame, seed inner state
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // inner work 1: NwInner → NwMid
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // inner work 2: NwMid → NwInnerOut
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // inner Done: pop frame, write NwInnerOut to parent
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // outer work: NwInnerOut → NwFinal
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // terminal
    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out, NwFinal { v: 21 }),
        other => panic!("expected Done, got {other:?}"),
    }
}

/// Verifies that the output value is computed through all inner-flow work nodes.
/// Uses run_to_done! for brevity; the value check is the assertion.
#[tokio::test]
async fn nested_work_flow_output_value_is_correct() {
    let mut rt = FlowRuntime::new(NwOuter { v: 3 }).unwrap();
    // (3 * 2 + 10) + 1 = 17
    assert_eq!(run_to_done!(rt), NwFinal { v: 17 });
}

/// The inner flow is standalone: running NwInner directly must still work.
#[tokio::test]
async fn nested_work_inner_flow_runs_standalone() {
    let mut rt = FlowRuntime::new(NwInner { v: 4 }).unwrap();
    // inner work 1: 4*2 = 8
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // inner work 2: 8+10 = 18
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out, NwInnerOut { v: 18 }),
        other => panic!("expected Done, got {other:?}"),
    }
}

// ── Nested flow — either ──────────────────────────────────────────────────────
//
// Inner flow (NeIn):   NeIn → either(NeLeft | NeRight) → work → NeOut
// Outer flow (NeOuter): NeOuter → work → NeIn → flow::<NeIn> → NeOut → work → NeFinal
//
// The Either shim bakes NeLeft/NeRight as state-node names. The rename wrapper must
// rewrite them to "NeIn::NeLeft" / "NeIn::NeRight" so the dispatch loop finds the
// work nodes that are now stored under those prefixed keys.

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NeOuter {
    go_left: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NeIn {
    go_left: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NeLeft {
    v: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NeRight {
    v: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NeOut {
    v: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NeFinal {
    v: i32,
}

impl Flow for NeIn {
    type Output = NeOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .either::<NeIn, NeLeft, NeRight, _>(|inp, _| {
                if inp.go_left {
                    Ok(Either::Left(NeLeft { v: 1 }))
                } else {
                    Ok(Either::Right(NeRight { v: 2 }))
                }
            })
            .work::<NeLeft, NeOut, _, _>(|l, _| async move { Ok(NeOut { v: l.v * 10 }) })
            .work::<NeRight, NeOut, _, _>(|r, _| async move { Ok(NeOut { v: r.v * 10 }) })
            .build()
    }
}

impl Flow for NeOuter {
    type Output = NeFinal;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work::<NeOuter, NeIn, _, _>(|o, _| async move { Ok(NeIn { go_left: o.go_left }) })
            .flow::<NeIn>()
            .work::<NeOut, NeFinal, _, _>(|n, _| async move { Ok(NeFinal { v: n.v + 100 }) })
            .build()
    }
}

/// Left branch through a nested either: name rewriting routes to the correct work node.
/// NeOuter(go_left=true) → either → NeIn::NeLeft(1) → work(*10) → NeOut(10) → work(+100) → NeFinal(110)
#[tokio::test]
async fn nested_either_left_branch_correct_output() {
    let mut rt = FlowRuntime::new(NeOuter { go_left: true }).unwrap();
    run_to_done!(rt); // just run through; value assertion is the point
    let mut rt = FlowRuntime::new(NeOuter { go_left: true }).unwrap();
    assert_eq!(run_to_done!(rt), NeFinal { v: 110 });
}

/// Right branch through a nested either: same name-rewriting path, different branch.
/// NeOuter(go_left=false) → either → NeIn::NeRight(2) → work(*10) → NeOut(20) → work(+100) → NeFinal(120)
#[tokio::test]
async fn nested_either_right_branch_correct_output() {
    let mut rt = FlowRuntime::new(NeOuter { go_left: false }).unwrap();
    assert_eq!(run_to_done!(rt), NeFinal { v: 120 });
}

/// Running outer flow left vs right produces independent results — no state leak between runs.
#[tokio::test]
async fn nested_either_branches_are_independent() {
    let left = {
        let mut rt = FlowRuntime::new(NeOuter { go_left: true }).unwrap();
        run_to_done!(rt)
    };
    let right = {
        let mut rt = FlowRuntime::new(NeOuter { go_left: false }).unwrap();
        run_to_done!(rt)
    };
    assert_ne!(left, right);
    assert_eq!(left.v, 110);
    assert_eq!(right.v, 120);
}

/// After Done via one branch, no dangling state from the other branch remains.
/// The second next() call must return Done, not Continue.
#[tokio::test]
async fn nested_either_no_dangling_state_after_done() {
    let mut rt = FlowRuntime::new(NeOuter { go_left: true }).unwrap();
    run_to_done!(rt);
    // All intermediate states cleared; next call should be Done again.
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Done(_)));
}

// ── Nested flow — fork + join ─────────────────────────────────────────────────
//
// Inner flow (NfIn):  NfIn → fork(NfA + NfB) → join(sum) → NfOut
// Outer flow (NfOuter): NfOuter → work → NfIn → flow::<NfIn> → NfOut → work(+1) → NfFinal
//
// Fork children are renamed from "NfA"/"NfB" to "NfIn::NfA"/"NfIn::NfB".
// The fork closure wrapper rewrites the baked state-node names to the prefixed keys.
// Join nodes have their parents list renamed accordingly.

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NfOuter {
    val: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NfIn {
    val: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NfA {
    val: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NfB {
    val: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NfOut {
    sum: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NfFinal {
    result: i32,
}

impl Flow for NfIn {
    type Output = NfOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork::<NfIn, NfA, NfB, _>(|inp, _| {
                Ok((NfA { val: inp.val }, NfB { val: inp.val * 2 }))
            })
            .join::<NfA, NfB, NfOut, _>(|a, b, _| Ok(NfOut { sum: a.val + b.val }))
            .build()
    }
}

impl Flow for NfOuter {
    type Output = NfFinal;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work::<NfOuter, NfIn, _, _>(|o, _| async move { Ok(NfIn { val: o.val }) })
            .flow::<NfIn>()
            .work::<NfOut, NfFinal, _, _>(|n, _| async move { Ok(NfFinal { result: n.sum + 1 }) })
            .build()
    }
}

/// Fork + join inside a nested flow: NfOuter(val=4) → NfIn(4) → fork(A=4, B=8) → join(sum=12) → NfOut(12) → NfFinal(13)
#[tokio::test]
async fn nested_fork_join_correct_output() {
    let mut rt = FlowRuntime::new(NfOuter { val: 4 }).unwrap();
    // outer work: NfOuter → NfIn
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // sub-flow node: push frame, seed inner state
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // fork: NfIn → NfA + NfB
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // join: NfA + NfB → NfOut
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // inner Done: pop frame, write NfOut to parent
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // outer work: NfOut → NfFinal
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out, NfFinal { result: 13 }),
        other => panic!("expected Done, got {other:?}"),
    }
}

/// After Done, no dangling fork-branch states remain (same invariant as non-nested).
#[tokio::test]
async fn nested_fork_join_no_dangling_state_after_done() {
    let mut rt = FlowRuntime::new(NfOuter { val: 2 }).unwrap();
    run_to_done!(rt);
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Done(_)));
}

// ── Nested flow — agent ───────────────────────────────────────────────────────
//
// Inner flow (NaIn): NaIn (agent) → NaOut
// Outer flow (NaOuter): NaOuter → work → NaIn → flow::<NaIn> → NaOut → work(uppercase) → NaFinal

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NaOuter {
    data: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct NaIn {
    prompt: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NaOut {
    result: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NaFinal {
    final_result: String,
}

impl Agent for NaIn {
    type Output = NaOut;
    fn build() -> AgentConfig {
        AgentConfig::new("test agent", "openai://test-model")
    }
}

impl Flow for NaIn {
    type Output = NaOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<NaIn>().build()
    }
}

impl Flow for NaOuter {
    type Output = NaFinal;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work::<NaOuter, NaIn, _, _>(|o, _| async move {
                Ok(NaIn {
                    prompt: o.data.clone(),
                })
            })
            .flow::<NaIn>()
            .work::<NaOut, NaFinal, _, _>(|a, _| async move {
                Ok(NaFinal {
                    final_result: a.result.to_uppercase(),
                })
            })
            .build()
    }
}

/// Agent inside a nested flow: the framed agent node receives and exits correctly.
/// The mock returns structured output; the outer work node uppercases it.
#[tokio::test]
async fn nested_agent_flow_produces_correct_output() {
    let factory = MockFactory::new(vec![resp(json!({"result": "hello"}))]);
    let mut rt = FlowRuntime::new(NaOuter {
        data: "test prompt".into(),
    })
    .unwrap()
    .with_factory(factory);

    // outer work: NaOuter → NaIn
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // sub-flow node: push frame, seed inner state
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // agent: NaIn → NaOut (structured output, no tool calls)
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // inner Done: pop frame, write NaOut to parent
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // outer work: NaOut → NaFinal (uppercase)
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out, NaFinal { final_result: "HELLO".into() }),
        other => panic!("expected Done, got {other:?}"),
    }
}

/// Inlined agent with a tool call round-trip inside the nested flow.
#[tokio::test]
async fn nested_agent_with_tool_call_completes() {
    #[derive(Debug, Deserialize, JsonSchema)]
    struct EchoTool {
        text: String,
    }
    impl Tool for EchoTool {
        type Output = String;
        fn name() -> &'static str {
            "echo_nested"
        }
        fn description() -> &'static str {
            "Echo text"
        }
        async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
            Ok(self.text)
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct NaToolIn {
        prompt: String,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct NaToolOut {
        result: String,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct NaToolOuter {
        data: String,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct NaToolFinal {
        final_result: String,
    }

    impl Agent for NaToolIn {
        type Output = NaToolOut;
        fn build() -> AgentConfig {
            AgentConfig::new("tool agent", "openai://test-model")
                .with_tools(ToolBox::builder().tool::<EchoTool>().build())
        }
    }
    impl Flow for NaToolIn {
        type Output = NaToolOut;
        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder().agent::<NaToolIn>().build()
        }
    }
    impl Flow for NaToolOuter {
        type Output = NaToolFinal;
        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder()
                .work::<NaToolOuter, NaToolIn, _, _>(|o, _| async move {
                    Ok(NaToolIn { prompt: o.data })
                })
                .flow::<NaToolIn>()
                .work::<NaToolOut, NaToolFinal, _, _>(|a, _| async move {
                    Ok(NaToolFinal {
                        final_result: a.result,
                    })
                })
                .build()
        }
    }

    let factory = MockFactory::new(vec![
        tool_resp(vec![make_call("echo_nested", json!({"text": "echoed"}))]),
        tool_resp(vec![make_call("submit", json!({"result": "done"}))]),
    ]);
    let mut rt = FlowRuntime::new(NaToolOuter {
        data: "run echo".into(),
    })
    .unwrap()
    .with_factory(factory);

    run_to_done!(rt);
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResumeForkRoot {
    value: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResumeForkLeft {
    value: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResumeForkRight {
    value: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResumeForkLeftReady {
    value: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResumeForkRightPrompt {
    value: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResumeForkRightOut {
    value: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct ResumeForkOut {
    total: i32,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ResumeApprovalTool {
    note: String,
}

impl Tool for ResumeApprovalTool {
    type Output = serde_json::Value;

    fn name() -> &'static str {
        "approve_resume"
    }

    fn description() -> &'static str {
        "Suspend until an approval payload is supplied"
    }

    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Err(ToolError::suspend(json!({"note": self.note})))
    }
}

impl Agent for ResumeForkRightPrompt {
    type Output = ResumeForkRightOut;

    fn build() -> AgentConfig {
        AgentConfig::new("resume-test agent", "openai://test-model")
            .with_tools(ToolBox::builder().tool::<ResumeApprovalTool>().build())
    }
}

impl Flow for ResumeForkRoot {
    type Output = ResumeForkOut;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork::<ResumeForkRoot, ResumeForkLeft, ResumeForkRight, _>(|root, _| {
                Ok((
                    ResumeForkLeft { value: root.value },
                    ResumeForkRight {
                        value: root.value + 1,
                    },
                ))
            })
            .work::<ResumeForkLeft, ResumeForkLeftReady, _, _>(|left, _| async move {
                Ok(ResumeForkLeftReady {
                    value: left.value * 2,
                })
            })
            .work::<ResumeForkRight, ResumeForkRightPrompt, _, _>(|right, _| async move {
                Ok(ResumeForkRightPrompt { value: right.value })
            })
            .agent::<ResumeForkRightPrompt>()
            .join::<ResumeForkLeftReady, ResumeForkRightOut, ResumeForkOut, _>(|left, right, _| {
                Ok(ResumeForkOut {
                    total: left.value + right.value,
                })
            })
            .build()
    }
}

/// Resuming a suspended agent must target that agent even when an earlier state is a waiting join.
#[tokio::test]
async fn resume_targets_suspended_agent_when_join_state_is_first() {
    let factory = MockFactory::new(vec![
        tool_resp(vec![make_call("approve_resume", json!({"note": "wait"}))]),
        tool_resp(vec![make_call("submit", json!({"value": 7}))]),
    ]);
    let mut rt = FlowRuntime::new(ResumeForkRoot { value: 3 })
        .unwrap()
        .with_factory(factory);

    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // fork
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // work left
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // work right
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // agent dispatch → approve_resume pending

    let tool_id = match rt.next(ctx()).await.unwrap() {
        RunOut::Suspend { tool_id, .. } => tool_id,
        other => panic!("expected Suspend, got {other:?}"),
    };

    let after_resume = rt
        .resume(ctx(), (tool_id, json!({"approved": true})))
        .await
        .unwrap();
    assert!(matches!(after_resume, RunOut::Continue)); // resume injected into history

    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // agent re-dispatch → submit pending
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // submit tool sets submitted_value
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // agent reads submitted_value → exit
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue)); // join
    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out, ResumeForkOut { total: 13 }),
        other => panic!("expected Done, got {other:?}"),
    }
}

// ── Nested flow — error cases ─────────────────────────────────────────────────

/// Registering the same inner flow twice produces a build-time error.
#[tokio::test]
async fn nested_flow_duplicate_entry_detected() {
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct DupInner {
        v: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct DupInnerOut {
        v: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct DupOuter {
        v: i32,
    }

    impl Flow for DupInner {
        type Output = DupInnerOut;
        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder()
                .work::<DupInner, DupInnerOut, _, _>(|i, _| async move {
                    Ok(DupInnerOut { v: i.v })
                })
                .build()
        }
    }
    impl Flow for DupOuter {
        type Output = DupInnerOut;
        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder()
                .flow::<DupInner>()
                // Second registration of DupInner's entry key → duplicate
                .flow::<DupInner>()
                .build()
        }
    }

    let err = match DupOuter::build() {
        Ok(_) => panic!("expected build error for duplicate flow"),
        Err(e) => e,
    };
    match err {
        FlowError::Invalid(problems) => {
            assert!(
                problems.iter().any(|p| p.contains("duplicate node key")),
                "expected duplicate node key error, got: {problems:?}"
            );
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A join whose target type matches a parent must be rejected because the result would be deleted.
#[tokio::test]
async fn join_target_matching_parent_is_rejected() {
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct BadJoinIn {
        value: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct BadJoinLeft {
        value: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct BadJoinRight {
        value: i32,
    }

    impl Flow for BadJoinIn {
        type Output = BadJoinLeft;

        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder()
                .fork::<BadJoinIn, BadJoinLeft, BadJoinRight, _>(|input, _| {
                    Ok((
                        BadJoinLeft { value: input.value },
                        BadJoinRight { value: input.value + 1 },
                    ))
                })
                .join::<BadJoinLeft, BadJoinRight, BadJoinLeft, _>(|left, _right, _| {
                    Ok(BadJoinLeft { value: left.value })
                })
                .build()
        }
    }

    let err = FlowRuntime::new(BadJoinIn { value: 1 }).unwrap_err();
    match err {
        FlowError::Invalid(problems) => {
            assert!(
                problems.iter().any(|p| p.contains("target matches parent")),
                "expected join target validation error, got: {problems:?}"
            );
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A failed inner `Flow::build()` propagates as an `Invalid` error on the outer build.
#[tokio::test]
async fn nested_flow_inner_build_failure_propagates() {
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct FailInner {
        v: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct FailInnerOut {
        v: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct FailOuter {
        v: i32,
    }

    impl Flow for FailInner {
        type Output = FailInnerOut;
        fn build() -> Result<FlowGraph, FlowError> {
            Err(FlowError::BuildError("deliberate inner failure".into()))
        }
    }
    impl Flow for FailOuter {
        type Output = FailInnerOut;
        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder().flow::<FailInner>().build()
        }
    }

    let err = match FailOuter::build() {
        Ok(_) => panic!("expected build error for failing inner flow"),
        Err(e) => e,
    };
    match err {
        FlowError::Invalid(problems) => {
            assert!(
                problems
                    .iter()
                    .any(|p| p.contains("deliberate inner failure")),
                "inner error message not surfaced: {problems:?}"
            );
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A flow with multiple distinct terminal state ids must be rejected at construction time.
#[test]
fn multiple_terminal_state_ids_are_rejected() {
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct MultiTerminalIn {
        go_left: bool,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct MultiTerminalLeft {
        value: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct MultiTerminalRight {
        value: String,
    }

    impl Flow for MultiTerminalIn {
        type Output = MultiTerminalLeft;

        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder()
                .either::<MultiTerminalIn, MultiTerminalLeft, MultiTerminalRight, _>(|input, _| {
                    if input.go_left {
                        Ok(Either::Left(MultiTerminalLeft { value: 1 }))
                    } else {
                        Ok(Either::Right(MultiTerminalRight {
                            value: "right".into(),
                        }))
                    }
                })
                .build()
        }
    }

    let err = FlowRuntime::new(MultiTerminalIn { go_left: true }).unwrap_err();
    match err {
        FlowError::Invalid(problems) => {
            assert!(
                problems.iter().any(|p| p.contains("exactly one terminal state id")),
                "expected terminal state validation error, got: {problems:?}"
            );
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A flow's declared output type must match its single terminal state id.
#[test]
fn flow_output_must_match_terminal_state_id() {
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct OutputMismatchIn {
        value: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct OutputMismatchTerminal {
        value: i32,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct OutputMismatchDeclared {
        value: String,
    }

    impl Flow for OutputMismatchIn {
        type Output = OutputMismatchDeclared;

        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder()
                .work::<OutputMismatchIn, OutputMismatchTerminal, _, _>(|input, _| async move {
                    Ok(OutputMismatchTerminal { value: input.value })
                })
                .build()
        }
    }

    let err = FlowRuntime::new(OutputMismatchIn { value: 7 }).unwrap_err();
    match err {
        FlowError::Invalid(problems) => {
            assert!(
                problems.iter().any(|p| p.contains("does not match terminal state id")),
                "expected output mismatch validation error, got: {problems:?}"
            );
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

// ── Snapshot / from_snapshot ──────────────────────────────────────────────────

/// Snapshot taken mid-flow can be serialized to JSON, deserialized, and used to
/// reconstruct a runtime that continues correctly to Done.
///
/// Uses NwOuter (3-step work chain) for a clean, deterministic value path.
#[tokio::test]
async fn snapshot_round_trip_continues_to_correct_done() {
    let mut rt = FlowRuntime::new(NwOuter { v: 5 }).unwrap();

    // Run the first step (outer work: NwOuter → NwInner).
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    // Capture and round-trip the snapshot through JSON serialization.
    let snapshot: FlowSnapshot = rt.snapshot();
    let json_bytes = serde_json::to_vec(&snapshot).expect("snapshot serialization failed");
    let restored: FlowSnapshot =
        serde_json::from_slice(&json_bytes).expect("snapshot deserialization failed");

    // Rebuild from the restored snapshot and run to completion.
    let mut rt2 = FlowRuntime::<NwOuter>::from_snapshot(restored).unwrap();
    // (5 * 2 + 10) + 1 = 21; first step already ran, so we continue from NwInner
    assert_eq!(run_to_done!(rt2), NwFinal { v: 21 });
}

/// Snapshot taken at the very start (before any step) recreates the identical initial state.
#[tokio::test]
async fn snapshot_from_initial_state_produces_same_result() {
    let mut rt_original = FlowRuntime::new(NwOuter { v: 2 }).unwrap();
    let snapshot = rt_original.snapshot();

    let json_bytes = serde_json::to_vec(&snapshot).unwrap();
    let restored: FlowSnapshot = serde_json::from_slice(&json_bytes).unwrap();
    let mut rt_restored = FlowRuntime::<NwOuter>::from_snapshot(restored).unwrap();

    let direct = run_to_done!(rt_original);
    let via_snapshot = run_to_done!(rt_restored);
    // (2 * 2 + 10) + 1 = 15
    assert_eq!(direct, NwFinal { v: 15 });
    assert_eq!(via_snapshot, NwFinal { v: 15 });
}

/// Snapshot taken after Done still lets the restored runtime return Done immediately.
#[tokio::test]
async fn snapshot_after_done_restored_runtime_returns_done() {
    let mut rt = FlowRuntime::new(NwOuter { v: 1 }).unwrap();
    run_to_done!(rt);

    let snapshot = rt.snapshot();
    let json_bytes = serde_json::to_vec(&snapshot).unwrap();
    let restored: FlowSnapshot = serde_json::from_slice(&json_bytes).unwrap();
    let mut rt2 = FlowRuntime::<NwOuter>::from_snapshot(restored).unwrap();

    // (1 * 2 + 10) + 1 = 13
    assert!(matches!(rt2.next(ctx()).await.unwrap(), RunOut::Done(_)));
}

/// Multiple independent snapshots from separate runtimes don't share state.
#[tokio::test]
async fn snapshots_are_independent_between_runtimes() {
    let mut rt_a = FlowRuntime::new(NwOuter { v: 10 }).unwrap();
    let mut rt_b = FlowRuntime::new(NwOuter { v: 20 }).unwrap();

    rt_a.next(ctx()).await.unwrap();
    rt_b.next(ctx()).await.unwrap();

    let snap_a = serde_json::to_vec(&rt_a.snapshot()).unwrap();
    let snap_b = serde_json::to_vec(&rt_b.snapshot()).unwrap();

    let mut rta2 =
        FlowRuntime::<NwOuter>::from_snapshot(serde_json::from_slice(&snap_a).unwrap()).unwrap();
    let mut rtb2 =
        FlowRuntime::<NwOuter>::from_snapshot(serde_json::from_slice(&snap_b).unwrap()).unwrap();

    let out_a = run_to_done!(rta2);
    let out_b = run_to_done!(rtb2);

    // a: (10*2+10)+1 = 31, b: (20*2+10)+1 = 51
    assert_eq!(out_a, NwFinal { v: 31 });
    assert_eq!(out_b, NwFinal { v: 51 });
}
