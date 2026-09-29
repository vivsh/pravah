use super::*;
use pravah::clients::{Role, TokenUsage, ToolCall};
use pravah::graph::{HistoryChange, from_value};
use pravah::{CompactionResult, HistoryEntry};

#[derive(Default)]
struct HistoryBatch;

impl ContinuationHandler for HistoryBatch {
    fn start<'a>(
        &'a self,
        payload: &'a Value,
        _: Option<Value>,
        _: Vec<Value>,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        from_value(payload.clone()).map_err(codec)
    }
    fn advance<'a>(
        &'a self,
        _: &'a Value,
        _: Value,
        _: ContinuationEvent,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        Err(GraphError::Invalid(
            "unexpected history-batch advance".into(),
        ))
    }
}

/// Uses the same deterministic append identity as the modern runtime.
fn entry(position: u64, session: &str, message: Message) -> HistoryEntry {
    HistoryEntry {
        id: Uuid::new_v5(
            &Uuid::from_u128(7),
            format!("pravah.history.v1:{position}").as_bytes(),
        ),
        position,
        session_id: session.into(),
        agent_id: "agent".into(),
        evicted: false,
        message,
    }
}

/// Runs an authored history transition through the real validation/commit boundary.
fn runtime(
    history: Vec<HistoryChange>,
    outputs: Vec<Value>,
) -> Result<pravah::Runtime, GraphError> {
    let builder = TypedGraphBuilder::<()>::new();
    let output = builder.continuation::<(), (), HistoryBatch, _>(
        builder.root(),
        ContinuationTransition {
            history,
            outputs,
            ..Default::default()
        },
    );
    builder.finish(output)?.start((), Uuid::from_u128(7))
}

fn compact(session: &str, indices: Vec<usize>) -> HistoryChange {
    HistoryChange::Compact {
        session_id: session.into(),
        decision: CompactionResult {
            evict_indices: indices,
            summary: Some("Summary".into()),
        },
    }
}

fn answer() -> Message {
    let mut usage = TokenUsage::default();
    usage.input = Some(10);
    usage.output = Some(2);
    Message::assistant("answer").with_usage(usage)
}

#[derive(Default)]
struct AppendAgain;
impl ContinuationHandler for AppendAgain {
    fn start<'a>(
        &'a self,
        payload: &'a Value,
        _: Option<Value>,
        _: Vec<Value>,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let size = payload
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| GraphError::Invalid("invalid fixture size".into()))?;
        Ok(ContinuationTransition {
            checkpoint: Some(true.into()),
            history: vec![HistoryChange::Append(vec![entry(
                0,
                "a",
                Message::user("x".repeat(size)),
            )])],
            ..Default::default()
        })
    }
    fn advance<'a>(
        &'a self,
        _: &'a Value,
        _: Value,
        _: ContinuationEvent,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            outputs: vec![Value::NULL],
            history: vec![HistoryChange::Append(vec![entry(1, "a", answer())])],
            ..Default::default()
        })
    }
}

/// Acknowledging a new batch does not copy payloads already owned by committed history.
#[test]
fn append_allocations_do_not_scale_with_retained_history() -> Result<(), GraphError> {
    let mut counts = Vec::new();
    for size in [1_u64, 1_000_000] {
        let builder = TypedGraphBuilder::<()>::new();
        let output = builder.continuation::<(), (), AppendAgain, _>(builder.root(), size);
        let mut runtime = builder.finish(output)?.start((), Uuid::from_u128(7))?;
        runtime.next()?;
        let mut outcome = Ok(Step::Continue);
        let allocations = allocation_counter::measure(|| {
            outcome = runtime.next();
        });
        outcome?;
        counts.push((allocations.count_total, allocations.bytes_total));
        assert_eq!(runtime.snapshot()?.history().entries().len(), 2);
    }
    assert_eq!(counts[0], counts[1]);
    Ok(())
}

/// Sequential append/compact changes preserve session isolation, positions and usage accounting.
#[test]
fn sequential_deltas_share_one_atomic_commit() -> Result<(), GraphError> {
    let changes = vec![
        HistoryChange::Append(vec![
            entry(0, "a", Message::user("one")),
            entry(1, "a", answer()),
            entry(2, "b", Message::user("other")),
            entry(3, "b", answer()),
            entry(4, "a", Message::user("two")),
        ]),
        compact("a", vec![0, 1]),
        HistoryChange::Append(vec![
            entry(5, "a", answer()),
            entry(6, "a", Message::user("three")),
        ]),
        compact("a", vec![0, 1, 2]),
    ];
    let mut runtime = runtime(changes, vec![Value::NULL])?;
    runtime.next()?;
    let snapshot = cbor_roundtrip(json_roundtrip(runtime.snapshot()?)?)?;
    let history = snapshot.history();
    assert_eq!(
        history
            .entries()
            .iter()
            .map(|e| e.position)
            .collect::<Vec<_>>(),
        [0, 2, 3, 6]
    );
    assert_eq!(history.session_entries("b").len(), 2);
    assert_eq!(history.total_input(), Some(30));
    assert_eq!(history.total_output(), Some(6));
    assert!(matches!(history.entries()[0].message.role, Role::System));
    assert_eq!(
        history.entries()[0].id,
        Uuid::new_v5(&Uuid::from_u128(7), b"pravah.summary.v1:a:7")
    );
    Ok(())
}

/// Later invalid appends, protected-prefix edits and VM output errors cannot commit earlier deltas.
#[test]
fn invalid_later_change_leaves_runtime_unchanged() -> Result<(), GraphError> {
    let initial = vec![
        entry(0, "a", Message::user("first")),
        entry(1, "a", answer()),
        entry(2, "a", Message::user("pending")),
    ];
    let mut wrong_id = entry(3, "a", answer());
    wrong_id.id = Uuid::nil();
    for change in [
        HistoryChange::Append(vec![entry(2, "a", answer())]),
        HistoryChange::Append(vec![wrong_id]),
        compact("a", vec![0, 1, 2]),
        compact("a", vec![1]),
        compact("a", vec![0]),
    ] {
        let mut runtime = runtime(
            vec![HistoryChange::Append(initial.clone()), change],
            vec![Value::NULL],
        )?;
        let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
        assert!(runtime.next().is_err());
        assert_eq!(
            serde_json::to_value(runtime.snapshot()?).map_err(codec)?,
            before
        );
    }
    let mut runtime = runtime(
        vec![HistoryChange::Append(initial)],
        vec![Value::NULL, Value::NULL],
    )?;
    let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    assert!(runtime.next().is_err());
    assert_eq!(
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?,
        before
    );
    Ok(())
}

/// Reclamation keeps complete provider tool groups; partial groups remain invalid.
#[test]
fn compaction_preserves_tool_groups() -> Result<(), GraphError> {
    let call = ToolCall::new("call".into(), "tool".into(), serde_json::json!({}));
    let entries = vec![
        entry(0, "a", Message::user("first")),
        entry(
            1,
            "a",
            Message::new(
                Role::AssistantToolCalls { calls: vec![call] },
                String::new(),
            ),
        ),
        entry(2, "a", Message::tool_output("call".into(), "result")),
        entry(3, "a", answer()),
        entry(4, "a", Message::user("pending")),
    ];
    for count in [2, 3, 4] {
        let mut runtime = runtime(
            vec![
                HistoryChange::Append(entries.clone()),
                compact("a", (0..count).collect()),
            ],
            vec![Value::NULL],
        )?;
        if count == 4 {
            runtime.next()?;
            assert_eq!(runtime.snapshot()?.history().entries().len(), 2);
        } else {
            assert!(runtime.next().is_err());
            assert!(runtime.snapshot()?.history().entries().is_empty());
        }
    }
    Ok(())
}
