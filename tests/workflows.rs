//! Integration tests for pure workflows — no real LLM calls.
//!
//! Every flow here uses only `work`, `fork`, `join`, `either`, and `flow` nodes.
//! The `testing` feature is NOT required because no agent nodes are exercised.

use either::Either;
use pravah::flows::{Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── shared helper ─────────────────────────────────────────────────────────────

fn ctx() -> Context {
    Context::new(FlowConf::default())
}

/// Drive a runtime to completion and return the output.
/// Panics if the flow suspends or never finishes within 100 steps.
async fn run_to_done<I: Flow>(mut rt: FlowRuntime<I>) -> I::Output {
    for _ in 0..100 {
        match rt.next(ctx()).await.expect("step failed") {
            FlowStep::Continue => {}
            FlowStep::Done(v) => return v,
            FlowStep::Suspend(_) => panic!("unexpected suspension"),
        }
    }
    panic!("flow did not finish within 100 steps")
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1: simple work chains
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Num(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NumPlus1(i64);

async fn add1(n: Num, _c: Context) -> Result<NumPlus1, FlowError> {
    Ok(NumPlus1(n.0 + 1))
}

impl Flow for Num {
    type Output = NumPlus1;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().work(add1).build()
    }
}

/// Single work node — flow goes straight through one transformation.
#[tokio::test]
async fn test_single_work_node() {
    let out = run_to_done(FlowRuntime::new(Num(5)).unwrap()).await;
    assert_eq!(out.0, 6);
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Chain3In(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Chain3Mid1(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Chain3Mid2(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Chain3Out(i64);

async fn chain3_step1(n: Chain3In, _c: Context) -> Result<Chain3Mid1, FlowError> {
    Ok(Chain3Mid1(n.0 + 10))
}
async fn chain3_step2(n: Chain3Mid1, _c: Context) -> Result<Chain3Mid2, FlowError> {
    Ok(Chain3Mid2(n.0 * 3))
}
async fn chain3_step3(n: Chain3Mid2, _c: Context) -> Result<Chain3Out, FlowError> {
    Ok(Chain3Out(n.0 - 5))
}

impl Flow for Chain3In {
    type Output = Chain3Out;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(chain3_step1)
            .work(chain3_step2)
            .work(chain3_step3)
            .build()
    }
}

/// Three chained work nodes — (n+10)*3-5.
#[tokio::test]
async fn test_three_chained_work_nodes() {
    let out = run_to_done(FlowRuntime::new(Chain3In(0)).unwrap()).await;
    assert_eq!(out.0, 25); // (0+10)*3-5
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2: either / branching
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EitherIn {
    value: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EvenBranch(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct OddBranch(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EitherOut(String);

async fn even_to_out(n: EvenBranch, _c: Context) -> Result<EitherOut, FlowError> {
    Ok(EitherOut(format!("even:{}", n.0)))
}
async fn odd_to_out(n: OddBranch, _c: Context) -> Result<EitherOut, FlowError> {
    Ok(EitherOut(format!("odd:{}", n.0)))
}
fn route_even_odd(i: EitherIn, _c: Context) -> Result<Either<EvenBranch, OddBranch>, FlowError> {
    if i.value % 2 == 0 {
        Ok(Either::Left(EvenBranch(i.value)))
    } else {
        Ok(Either::Right(OddBranch(i.value)))
    }
}

impl Flow for EitherIn {
    type Output = EitherOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .either(route_even_odd)
            .work(even_to_out)
            .work(odd_to_out)
            .build()
    }
}

/// Either node routes even numbers left, odd numbers right.
#[tokio::test]
async fn test_either_left_branch() {
    let out = run_to_done(FlowRuntime::new(EitherIn { value: 4 }).unwrap()).await;
    assert_eq!(out.0, "even:4");
}

#[tokio::test]
async fn test_either_right_branch() {
    let out = run_to_done(FlowRuntime::new(EitherIn { value: 7 }).unwrap()).await;
    assert_eq!(out.0, "odd:7");
}

/// Zero is even — tests the boundary of the routing predicate.
#[tokio::test]
async fn test_either_zero_is_even() {
    let out = run_to_done(FlowRuntime::new(EitherIn { value: 0 }).unwrap()).await;
    assert_eq!(out.0, "even:0");
}

// ─── work before routing ──────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct PreRouteIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct PreRouteNorm(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct PosBranch(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NegBranch(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct PreRouteOut(String);

async fn normalize(n: PreRouteIn, _c: Context) -> Result<PreRouteNorm, FlowError> {
    Ok(PreRouteNorm(n.0 * 2))
}
fn route_sign(n: PreRouteNorm, _c: Context) -> Result<Either<PosBranch, NegBranch>, FlowError> {
    if n.0 >= 0 {
        Ok(Either::Left(PosBranch(n.0)))
    } else {
        Ok(Either::Right(NegBranch(n.0)))
    }
}
async fn pos_label(n: PosBranch, _c: Context) -> Result<PreRouteOut, FlowError> {
    Ok(PreRouteOut(format!("pos:{}", n.0)))
}
async fn neg_label(n: NegBranch, _c: Context) -> Result<PreRouteOut, FlowError> {
    Ok(PreRouteOut(format!("neg:{}", n.0)))
}

impl Flow for PreRouteIn {
    type Output = PreRouteOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(normalize)
            .either(route_sign)
            .work(pos_label)
            .work(neg_label)
            .build()
    }
}

/// Work node runs before either, routing positive and negative paths.
#[tokio::test]
async fn test_work_then_either_positive() {
    let out = run_to_done(FlowRuntime::new(PreRouteIn(3)).unwrap()).await;
    assert_eq!(out.0, "pos:6");
}

#[tokio::test]
async fn test_work_then_either_negative() {
    let out = run_to_done(FlowRuntime::new(PreRouteIn(-5)).unwrap()).await;
    assert_eq!(out.0, "neg:-10");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3: fork + join
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkJoinIn {
    x: i64,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkLeft(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkRight(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkJoinOut {
    sum: i64,
}

fn split(i: ForkJoinIn, _c: Context) -> Result<(ForkLeft, ForkRight), FlowError> {
    Ok((ForkLeft(i.x + 1), ForkRight(i.x * 2)))
}
fn merge(l: ForkLeft, r: ForkRight, _c: Context) -> Result<ForkJoinOut, FlowError> {
    Ok(ForkJoinOut { sum: l.0 + r.0 })
}

impl Flow for ForkJoinIn {
    type Output = ForkJoinOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().fork(split).join(merge).build()
    }
}

/// Fork splits input, join combines both branches: (x+1) + (x*2).
#[tokio::test]
async fn test_fork_join_basic() {
    let out = run_to_done(FlowRuntime::new(ForkJoinIn { x: 4 }).unwrap()).await;
    assert_eq!(out.sum, 5 + 8);
}

// ─── fork → work on each branch → join ───────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FWJIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FWJLeft(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FWJRight(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FWJLeftDone(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FWJRightDone(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FWJOut(i64);

fn fwj_split(n: FWJIn, _c: Context) -> Result<(FWJLeft, FWJRight), FlowError> {
    Ok((FWJLeft(n.0), FWJRight(n.0)))
}
async fn fwj_left_work(n: FWJLeft, _c: Context) -> Result<FWJLeftDone, FlowError> {
    Ok(FWJLeftDone(n.0 + 100))
}
async fn fwj_right_work(n: FWJRight, _c: Context) -> Result<FWJRightDone, FlowError> {
    Ok(FWJRightDone(n.0 * 100))
}
fn fwj_join(l: FWJLeftDone, r: FWJRightDone, _c: Context) -> Result<FWJOut, FlowError> {
    Ok(FWJOut(l.0 - r.0))
}

impl Flow for FWJIn {
    type Output = FWJOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork(fwj_split)
            .work(fwj_left_work)
            .work(fwj_right_work)
            .join(fwj_join)
            .build()
    }
}

/// Independent work nodes on each branch before joining: (n+100) - (n*100).
#[tokio::test]
async fn test_fork_work_work_join() {
    let out = run_to_done(FlowRuntime::new(FWJIn(5)).unwrap()).await;
    assert_eq!(out.0, 105 - 500);
}

// ─── work → fork → join → work ───────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WFJWIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WFJWNorm(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WFJWL(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WFJWR(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WFJWMid(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WFJWOut(i64);

async fn wfjw_norm(n: WFJWIn, _c: Context) -> Result<WFJWNorm, FlowError> {
    Ok(WFJWNorm(n.0 + 1))
}
fn wfjw_split(n: WFJWNorm, _c: Context) -> Result<(WFJWL, WFJWR), FlowError> {
    Ok((WFJWL(n.0 * 2), WFJWR(n.0 * 3)))
}
fn wfjw_join(l: WFJWL, r: WFJWR, _c: Context) -> Result<WFJWMid, FlowError> {
    Ok(WFJWMid(l.0 + r.0))
}
async fn wfjw_final(n: WFJWMid, _c: Context) -> Result<WFJWOut, FlowError> {
    Ok(WFJWOut(n.0 * 10))
}

impl Flow for WFJWIn {
    type Output = WFJWOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(wfjw_norm)
            .fork(wfjw_split)
            .join(wfjw_join)
            .work(wfjw_final)
            .build()
    }
}

/// work → fork → join → work: ((x+1)*2 + (x+1)*3) * 10.
#[tokio::test]
async fn test_work_fork_join_work() {
    let out = run_to_done(FlowRuntime::new(WFJWIn(2)).unwrap()).await;
    assert_eq!(out.0, 150);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4: nested flows
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct InnerIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct InnerOut(i64);

async fn inner_double(n: InnerIn, _c: Context) -> Result<InnerOut, FlowError> {
    Ok(InnerOut(n.0 * 2))
}

impl Flow for InnerIn {
    type Output = InnerOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().work(inner_double).build()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct OuterIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct OuterOut(i64);

async fn outer_prep(n: OuterIn, _c: Context) -> Result<InnerIn, FlowError> {
    Ok(InnerIn(n.0 + 5))
}
async fn outer_post(n: InnerOut, _c: Context) -> Result<OuterOut, FlowError> {
    Ok(OuterOut(n.0 + 1))
}

impl Flow for OuterIn {
    type Output = OuterOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(outer_prep)
            .flow::<InnerIn>()
            .work(outer_post)
            .build()
    }
}

/// Nested flow: outer prepares, inner runs, outer finalises.
/// x=3 → inner gets 8 → doubled to 16 → outer adds 1 → 17.
#[tokio::test]
async fn test_nested_flow_basic() {
    let out = run_to_done(FlowRuntime::new(OuterIn(3)).unwrap()).await;
    assert_eq!(out.0, 17);
}

// ─── three levels of nesting ─────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct L2In(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct L2Out(i64);

async fn l2_work(n: L2In, _c: Context) -> Result<L2Out, FlowError> {
    Ok(L2Out(n.0 + 100))
}

impl Flow for L2In {
    type Output = L2Out;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().work(l2_work).build()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct L1In(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct L1Out(i64);

async fn l1_prep(n: L1In, _c: Context) -> Result<L2In, FlowError> {
    Ok(L2In(n.0 * 2))
}
async fn l1_post(n: L2Out, _c: Context) -> Result<L1Out, FlowError> {
    Ok(L1Out(n.0 - 10))
}

impl Flow for L1In {
    type Output = L1Out;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(l1_prep)
            .flow::<L2In>()
            .work(l1_post)
            .build()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RootIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RootOut(i64);

async fn root_prep(n: RootIn, _c: Context) -> Result<L1In, FlowError> {
    Ok(L1In(n.0 + 1))
}
async fn root_post(n: L1Out, _c: Context) -> Result<RootOut, FlowError> {
    Ok(RootOut(n.0))
}

impl Flow for RootIn {
    type Output = RootOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(root_prep)
            .flow::<L1In>()
            .work(root_post)
            .build()
    }
}

/// Three levels of nesting: (((x+1)*2)+100)-10.
#[tokio::test]
async fn test_doubly_nested_flow() {
    let out = run_to_done(FlowRuntime::new(RootIn(5)).unwrap()).await;
    assert_eq!(out.0, ((5 + 1) * 2 + 100) - 10); // 102
}

// ─── nested flow wrapping an inner fork/join ──────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NForkIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NForkA(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NForkB(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NForkADone(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NForkBDone(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NForkOut(i64);

fn nfork_split(n: NForkIn, _c: Context) -> Result<(NForkA, NForkB), FlowError> {
    Ok((NForkA(n.0), NForkB(n.0)))
}
async fn nfork_a_work(n: NForkA, _c: Context) -> Result<NForkADone, FlowError> {
    Ok(NForkADone(n.0 * 10))
}
async fn nfork_b_work(n: NForkB, _c: Context) -> Result<NForkBDone, FlowError> {
    Ok(NForkBDone(n.0 + 1000))
}
fn nfork_join(a: NForkADone, b: NForkBDone, _c: Context) -> Result<NForkOut, FlowError> {
    Ok(NForkOut(a.0 + b.0))
}

impl Flow for NForkIn {
    type Output = NForkOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork(nfork_split)
            .work(nfork_a_work)
            .work(nfork_b_work)
            .join(nfork_join)
            .build()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NFWrap(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NFWrapOut(i64);

async fn nf_prep(n: NFWrap, _c: Context) -> Result<NForkIn, FlowError> {
    Ok(NForkIn(n.0))
}
async fn nf_post(n: NForkOut, _c: Context) -> Result<NFWrapOut, FlowError> {
    Ok(NFWrapOut(n.0))
}

impl Flow for NFWrap {
    type Output = NFWrapOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(nf_prep)
            .flow::<NForkIn>()
            .work(nf_post)
            .build()
    }
}

/// Outer flow wraps an inner fork/join. x=3 → 3*10 + (3+1000) = 1033.
#[tokio::test]
async fn test_nested_flow_containing_fork_join() {
    let out = run_to_done(FlowRuntime::new(NFWrap(3)).unwrap()).await;
    assert_eq!(out.0, 1033);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5: error behaviour
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrKindIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrKindOut(i64);

async fn always_err(_n: ErrKindIn, _c: Context) -> Result<ErrKindOut, FlowError> {
    Err(FlowError::Internal { handler: "test_suspend", detail: "deliberate failure".into() })
}

impl Flow for ErrKindIn {
    type Output = ErrKindOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().work(always_err).build()
    }
}

/// A work node returning an error never produces Suspend or Done.
#[tokio::test]
async fn test_work_error_is_not_suspend() {
    let mut rt = FlowRuntime::new(ErrKindIn(0)).unwrap();
    let result = rt.next(ctx()).await;
    assert!(result.is_err());
    match result {
        Err(_) => {}
        Ok(FlowStep::Suspend(_)) => panic!("error should not surface as Suspend"),
        Ok(_) => panic!("error should not surface as Done/Continue"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 6: snapshot / restore
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SnapIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SnapMid(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SnapOut(i64);

async fn snap_step1(n: SnapIn, _c: Context) -> Result<SnapMid, FlowError> {
    Ok(SnapMid(n.0 + 1))
}
async fn snap_step2(n: SnapMid, _c: Context) -> Result<SnapOut, FlowError> {
    Ok(SnapOut(n.0 * 2))
}

impl Flow for SnapIn {
    type Output = SnapOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().work(snap_step1).work(snap_step2).build()
    }
}

/// Snapshot mid-flow; restored runtime produces the same result. Also verifies
/// the snapshot round-trips cleanly through JSON.
#[tokio::test]
async fn test_snapshot_restore_mid_flow() {
    let mut rt = FlowRuntime::new(SnapIn(9)).unwrap();
    assert!(matches!(rt.next(ctx()).await.unwrap(), FlowStep::Continue));

    let snap = rt.snapshot();
    let json = serde_json::to_string(&snap).unwrap();
    let snap2: pravah::flows::FlowSnapshot = serde_json::from_str(&json).unwrap();

    let rt2 = FlowRuntime::<SnapIn>::from_snapshot(snap2).unwrap();
    let out = run_to_done(rt2).await;
    assert_eq!(out.0, 20); // (9+1)*2
}

/// Calling next() on a completed (Done) snapshot returns an error.
#[tokio::test]
async fn test_snapshot_after_done_errors_on_next() {
    let mut rt = FlowRuntime::new(SnapIn(0)).unwrap();
    loop {
        match rt.next(ctx()).await.unwrap() {
            FlowStep::Done(_) => break,
            FlowStep::Continue => {}
            FlowStep::Suspend(_) => panic!("unexpected"),
        }
    }
    let mut rt2 = FlowRuntime::<SnapIn>::from_snapshot(rt.snapshot()).unwrap();
    assert!(rt2.next(ctx()).await.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 7: build-time validation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ValIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ValOut(i64);

/// Building a flow with no nodes fails — entry type is not registered.
#[tokio::test]
async fn test_empty_flow_build_fails() {
    impl Flow for ValIn {
        type Output = ValOut;
        fn build() -> Result<FlowGraph, FlowError> {
            FlowGraph::builder().build()
        }
    }
    assert!(FlowRuntime::new(ValIn(0)).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 8: complex composed flows
// ═══════════════════════════════════════════════════════════════════════════════

// ─── either inside a nested flow ─────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EFIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EFPos(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EFNeg(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EFOut(String);

fn ef_route(n: EFIn, _c: Context) -> Result<Either<EFPos, EFNeg>, FlowError> {
    if n.0 >= 0 {
        Ok(Either::Left(EFPos(n.0)))
    } else {
        Ok(Either::Right(EFNeg(n.0)))
    }
}
async fn ef_pos_work(n: EFPos, _c: Context) -> Result<EFOut, FlowError> {
    Ok(EFOut(format!("+{}", n.0)))
}
async fn ef_neg_work(n: EFNeg, _c: Context) -> Result<EFOut, FlowError> {
    Ok(EFOut(format!("{}", n.0)))
}

impl Flow for EFIn {
    type Output = EFOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .either(ef_route)
            .work(ef_pos_work)
            .work(ef_neg_work)
            .build()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EFWrap(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EFWrapOut(String);

async fn efw_prep(n: EFWrap, _c: Context) -> Result<EFIn, FlowError> {
    Ok(EFIn(n.0))
}
async fn efw_post(n: EFOut, _c: Context) -> Result<EFWrapOut, FlowError> {
    Ok(EFWrapOut(format!("val={}", n.0)))
}

impl Flow for EFWrap {
    type Output = EFWrapOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(efw_prep)
            .flow::<EFIn>()
            .work(efw_post)
            .build()
    }
}

/// Outer flow wraps inner flow containing either — both branches reachable.
#[tokio::test]
async fn test_outer_flow_inner_either_positive() {
    let out = run_to_done(FlowRuntime::new(EFWrap(42)).unwrap()).await;
    assert_eq!(out.0, "val=+42");
}

#[tokio::test]
async fn test_outer_flow_inner_either_negative() {
    let out = run_to_done(FlowRuntime::new(EFWrap(-7)).unwrap()).await;
    assert_eq!(out.0, "val=-7");
}

// ─── fork → either on left branch → join ─────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJLeft(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJRight(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJLeftA(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJLeftB(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJLeftDone(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJRightDone(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FEJOut(i64);

fn fej_split(n: FEJIn, _c: Context) -> Result<(FEJLeft, FEJRight), FlowError> {
    Ok((FEJLeft(n.0), FEJRight(n.0)))
}
fn fej_route(n: FEJLeft, _c: Context) -> Result<Either<FEJLeftA, FEJLeftB>, FlowError> {
    if n.0 >= 0 {
        Ok(Either::Left(FEJLeftA(n.0 + 1)))
    } else {
        Ok(Either::Right(FEJLeftB(n.0 - 1)))
    }
}
async fn fej_la_work(n: FEJLeftA, _c: Context) -> Result<FEJLeftDone, FlowError> {
    Ok(FEJLeftDone(n.0 * 10))
}
async fn fej_lb_work(n: FEJLeftB, _c: Context) -> Result<FEJLeftDone, FlowError> {
    Ok(FEJLeftDone(n.0 * 10))
}
async fn fej_right_work(n: FEJRight, _c: Context) -> Result<FEJRightDone, FlowError> {
    Ok(FEJRightDone(n.0 + 5))
}
fn fej_join(l: FEJLeftDone, r: FEJRightDone, _c: Context) -> Result<FEJOut, FlowError> {
    Ok(FEJOut(l.0 + r.0))
}

impl Flow for FEJIn {
    type Output = FEJOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork(fej_split)
            .either(fej_route)
            .work(fej_la_work)
            .work(fej_lb_work)
            .work(fej_right_work)
            .join(fej_join)
            .build()
    }
}

/// Fork, left branch has either, positive: left_a(4)→40, right=8, total=48.
#[tokio::test]
async fn test_fork_either_join_positive() {
    let out = run_to_done(FlowRuntime::new(FEJIn(3)).unwrap()).await;
    assert_eq!(out.0, 40 + 8);
}

/// Same topology, negative: left_b(-3)→-30, right=3, total=-27.
#[tokio::test]
async fn test_fork_either_join_negative() {
    let out = run_to_done(FlowRuntime::new(FEJIn(-2)).unwrap()).await;
    assert_eq!(out.0, -30 + 3);
}

// ─── step count ───────────────────────────────────────────────────────────────

/// Three chained work nodes produce exactly 2 Continue calls before Done.
#[tokio::test]
async fn test_step_count_three_work() {
    let mut rt = FlowRuntime::new(Chain3In(0)).unwrap();
    let mut continues = 0;
    loop {
        match rt.next(ctx()).await.unwrap() {
            FlowStep::Continue => continues += 1,
            FlowStep::Done(_) => break,
            FlowStep::Suspend(_) => panic!(),
        }
    }
    assert_eq!(continues, 2);
}

// ─── error propagation ───────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrIn(bool);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrOut(i64);

async fn failing_work(n: ErrIn, _c: Context) -> Result<ErrOut, FlowError> {
    if n.0 {
        Err(FlowError::Internal { handler: "test", detail: "intentional failure".into() })
    } else {
        Ok(ErrOut(0))
    }
}

impl Flow for ErrIn {
    type Output = ErrOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().work(failing_work).build()
    }
}

/// Work node returning FlowError propagates out of next().
#[tokio::test]
async fn test_work_error_propagates() {
    let mut rt = FlowRuntime::new(ErrIn(true)).unwrap();
    assert!(rt.next(ctx()).await.is_err());
}

// ─── routing is pure (not stateful) ──────────────────────────────────────────

/// Running the same either flow across a range of signed inputs confirms pure routing.
#[tokio::test]
async fn test_either_routing_is_stateless() {
    for i in -3i64..=3 {
        let out = run_to_done(FlowRuntime::new(EitherIn { value: i }).unwrap()).await;
        if i % 2 == 0 {
            assert_eq!(out.0, format!("even:{i}"));
        } else {
            assert_eq!(out.0, format!("odd:{i}"));
        }
    }
}

// ─── snapshot before fork ────────────────────────────────────────────────────

/// Snapshot at the start of a fork/join flow, restore, run — result equals fresh run.
#[tokio::test]
async fn test_snapshot_before_fork() {
    let fresh_out = run_to_done(FlowRuntime::new(ForkJoinIn { x: 7 }).unwrap()).await;

    let snap = FlowRuntime::new(ForkJoinIn { x: 7 }).unwrap().snapshot();
    let json = serde_json::to_string(&snap).unwrap();
    let snap2: pravah::flows::FlowSnapshot = serde_json::from_str(&json).unwrap();
    let out2 = run_to_done(FlowRuntime::<ForkJoinIn>::from_snapshot(snap2).unwrap()).await;

    assert_eq!(fresh_out.sum, out2.sum);
}

// ─── either routing errors ───────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrRouteIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrLeft(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrRight(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ErrRouteOut(i64);

fn err_route(n: ErrRouteIn, _c: Context) -> Result<Either<ErrLeft, ErrRight>, FlowError> {
    if n.0 == 999 {
        Err(FlowError::Internal { handler: "test", detail: "route error".into() })
    } else if n.0 >= 0 {
        Ok(Either::Left(ErrLeft(n.0)))
    } else {
        Ok(Either::Right(ErrRight(n.0)))
    }
}
async fn err_left(n: ErrLeft, _c: Context) -> Result<ErrRouteOut, FlowError> {
    Ok(ErrRouteOut(n.0))
}
async fn err_right(n: ErrRight, _c: Context) -> Result<ErrRouteOut, FlowError> {
    Ok(ErrRouteOut(n.0))
}

impl Flow for ErrRouteIn {
    type Output = ErrRouteOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .either(err_route)
            .work(err_left)
            .work(err_right)
            .build()
    }
}

/// Either routing closure returning an error propagates out of next().
#[tokio::test]
async fn test_either_routing_error_propagates() {
    let mut rt = FlowRuntime::new(ErrRouteIn(999)).unwrap();
    assert!(rt.next(ctx()).await.is_err());
}

/// Either routing that succeeds completes normally.
#[tokio::test]
async fn test_either_routing_success() {
    let out = run_to_done(FlowRuntime::new(ErrRouteIn(5)).unwrap()).await;
    assert_eq!(out.0, 5);
}

// ─── work after fork/join receives the joined value ──────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FJWIn(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FJWLeft(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FJWRight(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FJWJoined(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FJWOut(i64);

fn fjw_split(n: FJWIn, _c: Context) -> Result<(FJWLeft, FJWRight), FlowError> {
    Ok((FJWLeft(n.0), FJWRight(n.0)))
}
fn fjw_join(l: FJWLeft, r: FJWRight, _c: Context) -> Result<FJWJoined, FlowError> {
    Ok(FJWJoined(l.0 * r.0))
}
async fn fjw_post(n: FJWJoined, _c: Context) -> Result<FJWOut, FlowError> {
    Ok(FJWOut(n.0 + 1000))
}

impl Flow for FJWIn {
    type Output = FJWOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork(fjw_split)
            .join(fjw_join)
            .work(fjw_post)
            .build()
    }
}

/// Work node after join receives the joined value correctly: x*x + 1000.
#[tokio::test]
async fn test_work_after_fork_join_receives_correct_value() {
    let out = run_to_done(FlowRuntime::new(FJWIn(6)).unwrap()).await;
    assert_eq!(out.0, 36 + 1000);
}

// ─── snapshot mid fork-work-join ─────────────────────────────────────────────

/// Advance one step into a fork/work/join, snapshot+restore, finish — same result.
#[tokio::test]
async fn test_snapshot_mid_fork_work_join() {
    let fresh = run_to_done(FlowRuntime::new(FWJIn(4)).unwrap()).await;

    let mut rt = FlowRuntime::new(FWJIn(4)).unwrap();
    rt.next(ctx()).await.unwrap(); // fork step
    let snap = rt.snapshot();
    let json = serde_json::to_string(&snap).unwrap();
    let snap2: pravah::flows::FlowSnapshot = serde_json::from_str(&json).unwrap();
    let out = run_to_done(FlowRuntime::<FWJIn>::from_snapshot(snap2).unwrap()).await;
    assert_eq!(fresh.0, out.0);
}

// ─── snapshot of deeply nested flow ──────────────────────────────────────────

/// Snapshot a three-level nested flow at the start; restore and complete correctly.
#[tokio::test]
async fn test_deep_nested_flow_snapshot_restore() {
    let fresh = run_to_done(FlowRuntime::new(RootIn(3)).unwrap()).await;
    let snap = FlowRuntime::new(RootIn(3)).unwrap().snapshot();
    let json = serde_json::to_string(&snap).unwrap();
    let snap2: pravah::flows::FlowSnapshot = serde_json::from_str(&json).unwrap();
    let out = run_to_done(FlowRuntime::<RootIn>::from_snapshot(snap2).unwrap()).await;
    assert_eq!(fresh.0, out.0);
}

// ─── fork / join error propagation ───────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkErrIn(bool);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkErrL(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkErrR(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForkErrOut(i64);

fn fork_err_split(n: ForkErrIn, _c: Context) -> Result<(ForkErrL, ForkErrR), FlowError> {
    if n.0 {
        Err(FlowError::Internal { handler: "test", detail: "fork error".into() })
    } else {
        Ok((ForkErrL(1), ForkErrR(2)))
    }
}
fn fork_err_join(l: ForkErrL, r: ForkErrR, _c: Context) -> Result<ForkErrOut, FlowError> {
    Ok(ForkErrOut(l.0 + r.0))
}

impl Flow for ForkErrIn {
    type Output = ForkErrOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().fork(fork_err_split).join(fork_err_join).build()
    }
}

/// Fork closure returning an error propagates out of next().
#[tokio::test]
async fn test_fork_error_propagates() {
    let mut rt = FlowRuntime::new(ForkErrIn(true)).unwrap();
    assert!(rt.next(ctx()).await.is_err());
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct JoinErrIn {
    fail: bool,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct JoinErrL(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct JoinErrR(i64);
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct JoinErrOut(i64);

fn join_split(n: JoinErrIn, _c: Context) -> Result<(JoinErrL, JoinErrR), FlowError> {
    Ok((JoinErrL(if n.fail { 0 } else { 1 }), JoinErrR(2)))
}
fn join_err_merge(l: JoinErrL, r: JoinErrR, _c: Context) -> Result<JoinErrOut, FlowError> {
    if l.0 == 0 {
        Err(FlowError::Internal { handler: "test", detail: "join error".into() })
    } else {
        Ok(JoinErrOut(l.0 + r.0))
    }
}

impl Flow for JoinErrIn {
    type Output = JoinErrOut;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().fork(join_split).join(join_err_merge).build()
    }
}

/// Join closure returning an error propagates out of next().
#[tokio::test]
async fn test_join_error_propagates() {
    let mut rt = FlowRuntime::new(JoinErrIn { fail: true }).unwrap();
    rt.next(ctx()).await.unwrap(); // fork step
    let result = loop {
        match rt.next(ctx()).await {
            Ok(FlowStep::Continue) => {}
            Ok(FlowStep::Done(_)) => break Ok(()),
            Ok(FlowStep::Suspend(_)) => panic!(),
            Err(e) => break Err(e),
        }
    };
    assert!(result.is_err());
}

// ─── large values ────────────────────────────────────────────────────────────

/// Large i64 values flow through the chain without truncation.
#[tokio::test]
async fn test_work_large_value() {
    let out = run_to_done(FlowRuntime::new(Chain3In(i64::MAX / 4)).unwrap()).await;
    let expected = ((i64::MAX / 4 + 10) * 3) - 5;
    assert_eq!(out.0, expected);
}
