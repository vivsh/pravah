use super::*;

/// Builds an ordinary frame with no agent metadata so ownership validation remains independently testable.
fn runtime() -> Result<Runtime, GraphError> {
    let flow = crate::graph::compile(|root: crate::graph::Flow<String>| root)?;
    flow.start("input".into(), Uuid::nil())
}

/// Restored ownership must be ordered, nonempty, unkeyed and exclusive across frames.
#[test]
fn malformed_owner_lists_are_rejected() -> Result<(), GraphError> {
    for sessions in [vec![""], vec!["key:one"], vec!["a", "a"], vec!["b", "a"]] {
        let mut runtime = runtime()?;
        runtime.frame_mut(0)?.unkeyed_conversations =
            sessions.into_iter().map(str::to_owned).collect();
        assert!(matches!(
            validate_conversation_owners(&runtime.callables, &runtime.state, &runtime.history),
            Err(GraphError::SnapshotValidation(_))
        ));
    }
    let mut runtime = runtime()?;
    runtime
        .frame_mut(0)?
        .unkeyed_conversations
        .push("shared".into());
    let frame = runtime.frame(0)?.clone();
    runtime.state.frames.push(frame);
    assert!(
        validate_conversation_owners(&runtime.callables, &runtime.state, &runtime.history).is_err()
    );
    Ok(())
}

/// Imported unkeyed messages acquire the root lifetime; later local messages cannot restore without an owner.
#[test]
fn imported_and_local_history_require_exact_ownership() -> Result<(), GraphError> {
    let mut imported = MessageHistory::new();
    imported.push(
        "imported",
        "old-agent",
        crate::clients::Message::assistant("old"),
    );
    let flow = crate::graph::compile(|root: crate::graph::Flow<String>| root)?;
    let mut runtime = flow.start_with_history("input".into(), Uuid::nil(), imported)?;
    assert!(runtime.conversation_is_active("imported"));
    validate_conversation_owners(&runtime.callables, &runtime.state, &runtime.history)?;
    let entries = runtime.history.stage_entries(
        runtime.state.execution_id,
        "local",
        "custom",
        vec![crate::clients::Message::assistant("new")],
    )?;
    for entry in entries {
        runtime.history.commit_entry(entry);
    }
    assert!(
        validate_conversation_owners(&runtime.callables, &runtime.state, &runtime.history).is_err()
    );
    runtime
        .frame_mut(0)?
        .unkeyed_conversations
        .push("local".into());
    validate_conversation_owners(&runtime.callables, &runtime.state, &runtime.history)?;
    Ok(())
}

/// Emptying working rows never rewinds IDs, cumulative usage or persistence acknowledgements.
#[test]
fn drop_preserves_history_progress_and_accounting() -> Result<(), GraphError> {
    let mut runtime = runtime()?;
    let mut message = crate::clients::Message::assistant("reply");
    message.usage = Some(
        crate::clients::TokenUsage::new()
            .with_input(11)
            .with_output(7),
    );
    runtime.history.push("key:thread", "agent", message);
    runtime.state.persisted_history_position = 0;
    runtime
        .state
        .loaded_conversation_keys
        .insert("thread".into());
    let position = runtime.history.next_position();
    let usage = runtime.history.total_usage();
    let last = runtime
        .history
        .last_usage()
        .map(|usage| (usage.input, usage.output));
    runtime.drop_conversation("key:thread")?;
    assert!(runtime.history.is_empty());
    assert_eq!(runtime.history.next_position(), position);
    assert_eq!(runtime.history.total_usage(), usage);
    assert_eq!(
        runtime
            .history
            .last_usage()
            .map(|usage| (usage.input, usage.output)),
        last
    );
    assert_eq!(runtime.state.persisted_history_position, 0);
    assert!(!runtime.state.loaded_conversation_keys.contains("thread"));
    runtime.drop_conversation("key:thread")?;
    Ok(())
}

/// A frame retains its session even if compaction removed every working row; foreign frames cannot append to it.
#[test]
fn ownership_survives_empty_history_and_rejects_foreign_writers() -> Result<(), GraphError> {
    let mut runtime = runtime()?;
    runtime
        .frame_mut(0)?
        .unkeyed_conversations
        .push("session".into());
    assert!(runtime.conversation_is_active("session"));
    assert!(matches!(
        runtime.drop_conversation("session"),
        Err(GraphError::AgentConversationBusy)
    ));
    let mut frame = runtime.frame(0)?.clone();
    frame.unkeyed_conversations.clear();
    runtime.state.frames.push(frame);
    let entries = runtime.history.stage_entries(
        runtime.state.execution_id,
        "session",
        "custom",
        vec![crate::clients::Message::user("new")],
    )?;
    assert!(matches!(
        runtime.validate_conversation_append(1, &entries),
        Err(GraphError::HistoryValidation(_))
    ));
    assert!(runtime.history.is_empty());
    Ok(())
}

/// Keyed inspection and missing-ID removal allocate no scratch state or cached lifecycle metadata.
#[test]
fn inspection_and_missing_drop_allocate_nothing() -> Result<(), GraphError> {
    let mut runtime = runtime()?;
    runtime.history.push(
        "key:thread",
        "agent",
        crate::clients::Message::assistant("reply"),
    );
    let allocations = allocation_counter::measure(|| {
        assert!(!runtime.conversation_is_active("key:thread"));
        assert!(!runtime.conversation_is_active("absent"));
        assert!(runtime.drop_conversation("absent").is_ok());
    });
    assert_eq!(allocations.count_total, 0);
    Ok(())
}

/// A failed parent write cannot retire a child frame or erase its working conversation.
#[test]
fn failed_return_keeps_frame_and_history_retryable() -> Result<(), GraphError> {
    let mut runtime = runtime()?;
    runtime.frame_mut(0)?.write_epoch = u64::MAX;
    let mut child = runtime.frame(0)?.clone();
    child.write_epoch = 1;
    let exit = runtime.callables[0].graph.exit;
    child.return_target = Some(ReturnTarget::Edge { parent_edge: exit });
    child.unkeyed_conversations.push("child".into());
    runtime.state.frames.push(child);
    runtime.history.push(
        "child",
        "custom",
        crate::clients::Message::assistant("answer"),
    );
    let before = serde_json::to_value(runtime.snapshot()?)
        .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
    assert!(runtime.try_exit_frames().is_err());
    assert_eq!(runtime.state.frames.len(), 2);
    assert!(runtime.conversation_is_active("child"));
    assert_eq!(
        serde_json::to_value(runtime.snapshot()?)
            .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?,
        before
    );
    Ok(())
}

/// A summary retains the same frame owner after its source rows are removed, and disappears at frame exit.
#[test]
fn summaries_follow_frame_ownership_and_missing_owner_is_rejected() -> Result<(), GraphError> {
    let mut runtime = runtime()?;
    let entries = runtime.history.stage_entries(
        runtime.state.execution_id,
        "session",
        "custom",
        vec![
            crate::clients::Message::user("one"),
            crate::clients::Message::assistant("one"),
            crate::clients::Message::user("two"),
            crate::clients::Message::assistant("two"),
        ],
    )?;
    runtime.register_frame_conversations(0, &entries)?;
    for entry in entries {
        runtime.history.commit_entry(entry);
    }
    runtime.compact_history(
        "session",
        crate::CompactionResult {
            evict_indices: vec![0, 1],
            summary: Some("prior exchange".into()),
        },
    )?;
    assert_eq!(runtime.history.entries().len(), 3);
    assert!(runtime.conversation_is_active("session"));
    validate_conversation_owners(&runtime.callables, &runtime.state, &runtime.history)?;
    runtime.frame_mut(0)?.unkeyed_conversations.clear();
    assert!(
        validate_conversation_owners(&runtime.callables, &runtime.state, &runtime.history).is_err()
    );
    runtime
        .frame_mut(0)?
        .unkeyed_conversations
        .push("session".into());
    assert!(matches!(runtime.try_exit_frames()?, Step::Done(_)));
    assert!(runtime.history.is_empty());
    assert_eq!(runtime.history.next_position(), 4);
    Ok(())
}

/// The old format cannot silently restore without exact frame ownership.
#[test]
fn previous_snapshot_format_is_rejected() -> Result<(), GraphError> {
    let runtime = runtime()?;
    let mut snapshot = runtime.snapshot()?;
    snapshot.snapshot_version = 14;
    let flow = crate::graph::compile(|root: crate::graph::Flow<String>| root)?;
    assert!(matches!(
        flow.restore(snapshot),
        Err(GraphError::SnapshotVersion {
            got: 14,
            expected: 15
        })
    ));
    Ok(())
}

/// Arbitrary response data named `key` is not a conversation selection outside configuration.
#[test]
fn domain_result_keys_are_not_conversation_references() -> Result<(), GraphError> {
    let mut runtime = runtime()?;
    let request = AgentRequest::new(Uuid::nil(), AgentOperation::PersistHistory);
    let response = AgentResponse::new(
        request.id(),
        Ok(crate::graph::to_value(serde_json::json!({"key": "thread"}))
            .map_err(|error| GraphError::SnapshotValidation(error.to_string()))?),
    );
    runtime
        .frame_mut(0)?
        .continuation_inboxes
        .push(vec![ContinuationInput::Agent { request, response }]);
    assert!(!runtime.conversation_is_active("key:thread"));
    Ok(())
}
