use super::*;

fn proposal(call_id: &str) -> AgentToolProposal {
    AgentToolProposal::new(
        call_id.into(),
        "search".into(),
        Value::object([("query", Value::from("pravah"))]).expect("test arguments should be valid"),
    )
}

fn result(call_id: &str) -> AgentToolResult {
    AgentToolResult::new(
        call_id.into(),
        "search".into(),
        proposal(call_id).arguments().clone(),
        Value::from("found"),
        false,
    )
}

/// Verifies repetition ignores provider call identities but retains semantic values.
#[test]
fn repetition_metrics_use_canonical_semantic_batches() {
    let mut metrics = AgentLoopMetrics::default();
    metrics
        .record_proposal(&[proposal("provider-a")], None)
        .expect("first proposal should record");
    metrics
        .record_proposal(&[proposal("provider-b")], None)
        .expect("equivalent proposal should record");
    metrics
        .record_results(&[result("provider-a")])
        .expect("first result should record");
    metrics
        .record_results(&[result("provider-b")])
        .expect("equivalent result should record");

    assert_eq!(metrics.repeated_proposals(), 2);
    assert_eq!(metrics.repeated_results(), 2);
    assert_eq!(metrics.calls_for("search"), 2);
}

/// Verifies a final output counts as a model turn and ends the tool-round streak.
#[test]
fn output_metrics_reset_consecutive_tool_rounds() {
    let mut metrics = AgentLoopMetrics::default();
    metrics
        .record_proposal(&[proposal("provider-a")], None)
        .expect("proposal should record");
    metrics
        .record_output(Some({
            let mut usage = TokenUsage::default();
            usage.input = Some(5);
            usage.output = Some(3);
            usage
        }))
        .expect("output should record");

    assert_eq!(metrics.model_turns(), 2);
    assert_eq!(metrics.consecutive_tool_rounds(), 0);
    assert_eq!(metrics.input_tokens(), 5);
    assert_eq!(metrics.output_tokens(), 3);
}
