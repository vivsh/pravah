//! Integration tests for `pravah` using the public API only.
//!
//! Covers:
//! - Suspend / resume — simple (single agent) and nested (agent inside a multi-node flow)
//! - Fork + join — convergence, correct merged value, no dangling intermediate nodes
//! - Either — correct branch taken, other branch never executes
//! - No dangling nodes — after `Done`, calling `next()` again returns `Done` (not `Continue`)

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use either::Either;
use pravah::clients::{
    Client, ClientError, ClientFactory, ClientOptions, ClientOutput, ClientResponse, Provider,
    ToolCall,
};
use pravah::commons::Agent;
use pravah::flows::{AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, RunOut};
use pravah::tools::{Tool, ToolBox, ToolError};
use pravah::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ── Mock client infrastructure ────────────────────────────────────────────────

struct MockHandle {
    responses: Arc<Mutex<VecDeque<ClientResponse>>>,
}

#[async_trait]
impl Client for MockHandle {
    async fn execute(&self, _messages: &[pravah::clients::Message]) -> Result<ClientResponse, ClientError> {
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

fn call(name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: format!("id-{name}"),
        name: name.to_string(),
        args,
        thought_signatures: None,
    }
}

fn ctx() -> Context {
    Context::new(pravah::FlowConf {
        working_dir: Some(std::env::temp_dir()),
        ..Default::default()
    })
}

/// Drive the runtime until `Done`, panic on `Suspend`.
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

// ── Domain types — suspend/resume simple ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SimpleIn {
    task: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct SimpleOut {
    signed_by: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GateTool {
    reason: String,
}

impl Tool for GateTool {
    type Output = serde_json::Value;
    fn name() -> &'static str {
        "gate"
    }
    fn description() -> &'static str {
        "Block until an external reviewer approves"
    }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        // Always suspends — the caller is responsible for resuming.
        Err(ToolError::suspend(json!({"reason": self.reason})))
    }
}

impl Agent for SimpleIn {
    type Output = SimpleOut;
    fn build() -> AgentConfig {
        AgentConfig::new("Approval agent", "openai://test-model")
            .with_tools(ToolBox::builder().tool::<GateTool>().build())
    }
}

impl Flow for SimpleIn {
    type Output = SimpleOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<SimpleIn>().build()
    }
}

// ── Domain types — suspend/resume nested ─────────────────────────────────────

// A flow: Work → Agent (may suspend) → Work → Done
//   SetupIn (work) → TaskIn (agent) → ReviewIn (work) → ReviewOut (terminal)

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SetupIn {
    seed: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TaskIn {
    prompt: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ReviewIn {
    result: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct ReviewOut {
    summary: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CheckTool {
    note: String,
}
impl Tool for CheckTool {
    type Output = serde_json::Value;
    fn name() -> &'static str {
        "check"
    }
    fn description() -> &'static str {
        "Check item and suspend"
    }
    async fn call(self, _ctx: Context) -> Result<Self::Output, ToolError> {
        Err(ToolError::suspend(json!({"note": self.note})))
    }
}

impl Agent for TaskIn {
    type Output = ReviewIn;
    fn build() -> AgentConfig {
        AgentConfig::new("task agent", "openai://test-model")
            .with_tools(ToolBox::builder().tool::<CheckTool>().build())
    }
}

impl Flow for SetupIn {
    type Output = ReviewOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work::<SetupIn, TaskIn, _, _>(|inp, _| async move {
                Ok(TaskIn {
                    prompt: format!("process seed={}", inp.seed),
                })
            })
            .agent::<TaskIn>()
            .work::<ReviewIn, ReviewOut, _, _>(|r, _| async move {
                Ok(ReviewOut {
                    summary: r.result.to_uppercase(),
                })
            })
            .build()
    }
}

// ── Domain types — fork + join ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SplitIn {
    n: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct BranchAlpha {
    n: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct BranchBeta {
    n: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ProcessedBeta {
    n: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct MergedOut {
    total: i32,
}

impl Flow for SplitIn {
    type Output = MergedOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork::<SplitIn, BranchAlpha, BranchBeta, _>(|inp, _| {
                Ok((BranchAlpha { n: inp.n }, BranchBeta { n: inp.n * 3 }))
            })
            // Process beta branch before the join
            .work::<BranchBeta, ProcessedBeta, _, _>(|b, _| async move {
                Ok(ProcessedBeta { n: b.n + 1 })
            })
            .join::<BranchAlpha, ProcessedBeta, MergedOut, _>(|a, pb, _| {
                Ok(MergedOut { total: a.n + pb.n })
            })
            .build()
    }
}

// ── Domain types — either ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RouteIn {
    go_left: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct LeftPath {
    tag: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RightPath {
    tag: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct RouteOut {
    tag: String,
    from_left: bool,
}

impl Flow for RouteIn {
    type Output = RouteOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .either::<RouteIn, LeftPath, RightPath, _>(|inp, _| {
                if inp.go_left {
                    Ok(Either::Left(LeftPath { tag: "L".into() }))
                } else {
                    Ok(Either::Right(RightPath { tag: "R".into() }))
                }
            })
            .work::<LeftPath, RouteOut, _, _>(|l, _| async move {
                Ok(RouteOut { tag: l.tag, from_left: true })
            })
            .work::<RightPath, RouteOut, _, _>(|r, _| async move {
                Ok(RouteOut { tag: r.tag, from_left: false })
            })
            .build()
    }
}

// ── Tests: simple suspend / resume ───────────────────────────────────────────

#[tokio::test]
async fn simple_suspend_and_resume_completes() {
    // LLM call 1: requests the gate tool → suspends
    // After resume: LLM call 2: calls submit → agent exits
    let factory = MockFactory::new(vec![
        tool_resp(vec![call("gate", json!({"reason": "needs approval"}))]),
        tool_resp(vec![call("submit", json!({"signed_by": "alice"}))]),
    ]);
    let mut rt = FlowRuntime::new(SimpleIn { task: "deploy".into() })
        .unwrap()
        .with_factory(factory);

    // Step 1: agent runs → tool suspends
    let out = rt.next(ctx()).await.expect("next() failed");
    let tool_id = match out {
        RunOut::Suspend { tool_id, .. } => tool_id,
        other => panic!("expected Suspend, got {other:?}"),
    };
    assert!(tool_id.contains("gate"), "tool_id should mention 'gate': {tool_id}");

    // Resume with reviewer payload
    let after_resume = rt
        .resume(ctx(), (tool_id, json!({"approved": true})))
        .await
        .expect("resume() failed");
    // Tool batch finishes (Complete) → Continue; agent needs another LLM turn
    assert!(
        matches!(after_resume, RunOut::Continue),
        "expected Continue after resume, got {after_resume:?}"
    );

    // Step 2: LLM call 2 → submit → agent exits → Continue (exit state set)
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    // Terminal → Done
    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out.signed_by, "alice"),
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn simple_suspend_next_without_resume_errors() {
    let factory = MockFactory::new(vec![
        tool_resp(vec![call("gate", json!({"reason": "gate"}))]),
    ]);
    let mut rt = FlowRuntime::new(SimpleIn { task: "test".into() })
        .unwrap()
        .with_factory(factory);

    rt.next(ctx()).await.unwrap(); // → Suspend

    let err = rt.next(ctx()).await.unwrap_err();
    assert!(
        matches!(err, FlowError::ResumeRequired(_)),
        "expected ResumeRequired, got {err:?}"
    );
}

#[tokio::test]
async fn simple_resume_with_wrong_tool_id_errors() {
    let factory = MockFactory::new(vec![
        tool_resp(vec![call("gate", json!({"reason": "gate"}))]),
    ]);
    let mut rt = FlowRuntime::new(SimpleIn { task: "test".into() })
        .unwrap()
        .with_factory(factory);

    rt.next(ctx()).await.unwrap(); // → Suspend

    let err = rt
        .resume(ctx(), ("WrongAgent::wrong_tool".into(), json!({})))
        .await
        .unwrap_err();
    assert!(
        matches!(err, FlowError::ResumeMismatchError(_)),
        "expected ResumeMismatchError, got {err:?}"
    );
}

#[tokio::test]
async fn simple_resume_when_not_suspended_errors() {
    // Agent completes without suspending (structured output)
    let factory = MockFactory::new(vec![resp(json!({"signed_by": "bot"}))]);
    let mut rt = FlowRuntime::new(SimpleIn { task: "test".into() })
        .unwrap()
        .with_factory(factory);

    rt.next(ctx()).await.unwrap(); // → Continue (agent exited normally)

    let err = rt
        .resume(ctx(), ("SimpleIn::gate".into(), json!({})))
        .await
        .unwrap_err();
    assert!(
        matches!(err, FlowError::UnexpectedResumption(_)),
        "expected UnexpectedResumption, got {err:?}"
    );
}

// ── Tests: nested suspend / resume ───────────────────────────────────────────

#[tokio::test]
async fn nested_suspend_and_resume_completes() {
    // Flow: SetupIn(work) → TaskIn(agent, suspends) → ReviewIn(work) → ReviewOut
    // LLM call 1: check tool → suspends
    // Resume → LLM call 2: submit → agent exits
    let factory = MockFactory::new(vec![
        tool_resp(vec![call("check", json!({"note": "needs review"}))]),
        tool_resp(vec![call("submit", json!({"result": "ok"}))]),
    ]);
    let mut rt = FlowRuntime::new(SetupIn { seed: 7 })
        .unwrap()
        .with_factory(factory);

    // Step 1: work node (SetupIn→TaskIn) → Continue
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    // Step 2: agent runs → check tool → suspends
    let out = rt.next(ctx()).await.expect("step 2 failed");
    let tool_id = match out {
        RunOut::Suspend { tool_id, .. } => tool_id,
        other => panic!("expected Suspend after agent step, got {other:?}"),
    };
    assert!(tool_id.contains("check"), "tool_id should contain 'check': {tool_id}");

    // Resume
    let after_resume = rt
        .resume(ctx(), (tool_id, json!({"approved": true})))
        .await
        .expect("resume failed");
    assert!(
        matches!(after_resume, RunOut::Continue),
        "expected Continue after resume, got {after_resume:?}"
    );

    // Step 3: LLM call 2 → submit → agent exits → Continue
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    // Step 4: ReviewIn work node runs → Continue
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    // Terminal → Done (ReviewOut.summary is uppercased ReviewIn.result)
    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out.summary, "OK"),
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn nested_suspend_state_is_preserved_through_resume() {
    // Verify the work done before the suspension (SetupIn → TaskIn) is not lost
    // after resume. The agent's prompt should still reflect the seed value.
    // We test this indirectly: the submit output carries the result string "ok",
    // and the post-agent work uppercases it. If state were lost, the work step
    // would fail to deserialize ReviewIn.
    let factory = MockFactory::new(vec![
        tool_resp(vec![call("check", json!({"note": "verify"}) )]),
        tool_resp(vec![call("submit", json!({"result": "completed"}))]),
    ]);
    let mut rt = FlowRuntime::new(SetupIn { seed: 42 })
        .unwrap()
        .with_factory(factory);

    rt.next(ctx()).await.unwrap(); // work
    let step2 = rt.next(ctx()).await.unwrap(); // agent → suspend
    let tool_id = match step2 {
        RunOut::Suspend { tool_id, .. } => tool_id,
        other => panic!("{other:?}"),
    };
    rt.resume(ctx(), (tool_id, json!(null))).await.unwrap();
    rt.next(ctx()).await.unwrap(); // agent second LLM turn → exit → Continue
    rt.next(ctx()).await.unwrap(); // ReviewIn work → Continue

    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out.summary, "COMPLETED"),
        other => panic!("expected Done, got {other:?}"),
    }
}

// ── Tests: fork + join ────────────────────────────────────────────────────────

#[tokio::test]
async fn fork_join_produces_correct_merged_value() {
    // SplitIn(n=4):
    //   fork → BranchAlpha(4) + BranchBeta(12)
    //   work(BranchBeta→ProcessedBeta): 12+1 = 13
    //   join(BranchAlpha, ProcessedBeta): 4+13 = 17
    let mut rt = FlowRuntime::new(SplitIn { n: 4 }).unwrap();

    // fork fires
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // BranchBeta work fires (BranchAlpha join not ready yet)
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // join fires (both parents present)
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out, MergedOut { total: 17 }),
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn fork_join_no_dangling_nodes_after_done() {
    // After Done, calling next() again must return Done (not Continue).
    // This proves no intermediate fork-branch states remain active.
    let mut rt = FlowRuntime::new(SplitIn { n: 2 }).unwrap();

    rt.next(ctx()).await.unwrap(); // fork
    rt.next(ctx()).await.unwrap(); // work on beta
    rt.next(ctx()).await.unwrap(); // join

    let first_done = rt.next(ctx()).await.unwrap();
    assert!(
        matches!(first_done, RunOut::Done(_)),
        "expected Done, got {first_done:?}"
    );

    // Second call must also return Done — no dangling work left
    let second = rt.next(ctx()).await.unwrap();
    assert!(
        matches!(second, RunOut::Done(_)),
        "expected Done on second call (no dangling nodes), got {second:?}"
    );
}

#[tokio::test]
async fn join_does_not_fire_until_both_branches_present() {
    // After fork, BranchAlpha is in state but ProcessedBeta is not yet.
    // The join for BranchAlpha should be skipped, not fired early.
    // Next step should be the work node for BranchBeta, not the join.
    let mut rt = FlowRuntime::new(SplitIn { n: 1 }).unwrap();

    // fork produces BranchAlpha(1) and BranchBeta(3)
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    // At this point: alpha join cannot fire (no ProcessedBeta yet);
    // beta work can fire → produces ProcessedBeta(4)
    let step2 = rt.next(ctx()).await.unwrap();
    assert!(
        matches!(step2, RunOut::Continue),
        "expected Continue (beta work, join not ready), got {step2:?}"
    );

    // Now both parents present → join fires
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => assert_eq!(out.total, 1 + 3 + 1), // alpha(1) + beta(3)+1 = 5
        other => panic!("expected Done, got {other:?}"),
    }
}

// ── Tests: either ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn either_left_branch_taken_and_completed() {
    let mut rt = FlowRuntime::new(RouteIn { go_left: true }).unwrap();

    // either fires → routes to LeftPath
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    // LeftPath work fires → RouteOut(tag="L", from_left=true)
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => {
            assert_eq!(out.tag, "L");
            assert!(out.from_left);
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn either_right_branch_taken_and_completed() {
    let mut rt = FlowRuntime::new(RouteIn { go_left: false }).unwrap();

    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));
    assert!(matches!(rt.next(ctx()).await.unwrap(), RunOut::Continue));

    match rt.next(ctx()).await.unwrap() {
        RunOut::Done(out) => {
            assert_eq!(out.tag, "R");
            assert!(!out.from_left);
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn either_no_dangling_nodes_after_done() {
    // After Done via the left branch, right-branch work should never have run.
    // Verify: after Done, the next call returns Done (not Continue), proving
    // no right-branch state is sitting unprocessed.
    let mut rt = FlowRuntime::new(RouteIn { go_left: true }).unwrap();

    run_to_done!(rt);

    // Second call: only the terminal state remains — must be Done
    let second = rt.next(ctx()).await.unwrap();
    assert!(
        matches!(second, RunOut::Done(_)),
        "expected Done on second call (no dangling right-branch state), got {second:?}"
    );
}

#[tokio::test]
async fn either_left_and_right_are_independent() {
    // Running the same flow twice with different routing must yield independent results.
    let out_left = {
        let mut rt = FlowRuntime::new(RouteIn { go_left: true }).unwrap();
        run_to_done!(rt)
    };
    let out_right = {
        let mut rt = FlowRuntime::new(RouteIn { go_left: false }).unwrap();
        run_to_done!(rt)
    };
    assert!(out_left.from_left);
    assert!(!out_right.from_left);
    assert_ne!(out_left.tag, out_right.tag);
}
