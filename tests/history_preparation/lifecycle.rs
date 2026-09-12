use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::*;
use pravah::Snapshot;
use pravah::testing::CapturingHistoryStore;

#[derive(Clone, Default)]
struct CountPreparation(Arc<AtomicUsize>);

impl HistoryPreparer for CountPreparation {
    type Error = std::convert::Infallible;

    async fn prepare(
        &self,
        _request: HistoryPreparation<'_>,
    ) -> Result<HistoryReplacement, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(HistoryReplacement::default())
    }
}

/// No policy preserves ordinary chat history and all appended store records.
#[tokio::test]
async fn no_policy_keeps_ordinary_chat_behavior() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!({"text":"a"}))
        .then_output(serde_json::json!({"text":"b"}));
    let store = CapturingHistoryStore::new();
    let mut chat = Chat::new(
        tutor,
        Context::default().with_client_factory(factory.clone()),
    )
    .with_store(store.clone());
    for text in ["a", "b"] {
        chat.send(Question { text: text.into() }).await?;
    }
    assert_eq!(
        factory
            .calls()
            .iter()
            .map(|(_, messages)| messages.len())
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert_eq!(chat.snapshot()?.history().entries().len(), 4);
    assert_eq!(store.record_count(), 4);
    Ok(())
}

/// Each actual model dispatch has one policy invocation and final output has none.
#[tokio::test]
async fn final_output_does_not_prepare_again() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!({"text":"a"}));
    let policy = CountPreparation::default();
    let mut chat = Chat::new(
        tutor,
        Context::default().with_client_factory(factory.clone()),
    )
    .with_history_preparer(policy.clone());
    chat.send(Question { text: "a".into() }).await?;
    assert_eq!(factory.calls().len(), 1);
    assert_eq!(policy.0.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Repeated consolidation bounds physical history while the audit store receives every original row.
#[tokio::test]
async fn repeated_summaries_bound_snapshots_without_rewriting_store() -> Result<(), GraphError> {
    let mut factory = ScriptedFactory::new();
    for _ in 0..20 {
        factory = factory.then_output(serde_json::json!({"text":"answer"}));
    }
    let store = CapturingHistoryStore::new();
    let mut chat = Chat::new(tutor, Context::default().with_client_factory(factory))
        .with_history_preparer(Summarize)
        .with_store(store.clone());
    let mut sizes = Vec::new();
    for _ in 0..20 {
        chat.send(Question {
            text: "question".repeat(30),
        })
        .await?;
        let snapshot = chat.snapshot()?;
        assert!(snapshot.history().entries().len() <= 3);
        assert!(snapshot.history().entries().iter().all(|e| !e.evicted));
        sizes.push(
            serde_json::to_vec(snapshot.history())
                .expect("history JSON")
                .len(),
        );
    }
    assert!(sizes[19] <= sizes[1] + 10);
    assert_eq!(store.record_count(), 40);
    let entries = store.all_entries();
    assert!(
        entries
            .windows(2)
            .all(|pair| pair[1].position == pair[0].position + 1)
    );
    assert!(
        entries
            .iter()
            .all(|e| e.agent_id != "__summary__" && !e.evicted)
    );
    Ok(())
}

/// JSON and CBOR snapshots restore with fresh policy and client dependencies and no format change.
#[tokio::test]
async fn snapshots_restore_with_fresh_preparer() -> Result<(), GraphError> {
    let original = CountPreparation::default();
    let mut chat = Chat::new(
        tutor,
        Context::default().with_client_factory(
            ScriptedFactory::new().then_output(serde_json::json!({"text":"a"})),
        ),
    )
    .with_history_preparer(original.clone());
    chat.send(Question { text: "a".into() }).await?;
    let snapshot = chat.snapshot()?;
    let json = serde_json::to_vec(&snapshot).expect("JSON snapshot");
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor).expect("CBOR snapshot");
    let snapshots: [Snapshot; 2] = [
        serde_json::from_slice(&json).expect("JSON restore"),
        ciborium::from_reader(cbor.as_slice()).expect("CBOR restore"),
    ];
    for snapshot in snapshots {
        let fresh = CountPreparation::default();
        let mut restored = Chat::from_snapshot(
            tutor,
            snapshot,
            Context::default().with_client_factory(
                ScriptedFactory::new().then_output(serde_json::json!({"text":"b"})),
            ),
        )?
        .with_history_preparer(fresh.clone());
        restored.send(Question { text: "b".into() }).await?;
        assert_eq!(fresh.0.load(Ordering::SeqCst), 1);
    }
    assert_eq!(original.0.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Physically consolidated history, including its system summary, survives both snapshot codecs.
#[tokio::test]
async fn summary_snapshots_round_trip_and_continue() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!({"text":"a"}))
        .then_output(serde_json::json!({"text":"b"}));
    let mut chat = Chat::new(tutor, Context::default().with_client_factory(factory))
        .with_history_preparer(Summarize);
    for text in ["a", "b"] {
        chat.send(Question { text: text.into() }).await?;
    }
    let snapshot = chat.snapshot()?;
    let json = serde_json::to_vec(&snapshot).expect("JSON snapshot");
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor).expect("CBOR snapshot");
    let copies: [Snapshot; 2] = [
        serde_json::from_slice(&json).expect("JSON restore"),
        ciborium::from_reader(cbor.as_slice()).expect("CBOR restore"),
    ];
    for copy in copies {
        assert_eq!(
            serde_json::to_value(&copy).expect("copy"),
            serde_json::to_value(&snapshot).expect("snapshot")
        );
        let client = ScriptedFactory::new().then_output(serde_json::json!({"text":"c"}));
        let mut restored = Chat::from_snapshot(
            tutor,
            copy,
            Context::default().with_client_factory(client.clone()),
        )?
        .with_history_preparer(Summarize);
        restored.send(Question { text: "c".into() }).await?;
        assert_eq!(restored.snapshot()?.history().entries().len(), 3);
        assert!(matches!(client.calls()[0].1[0].role, Role::System));
    }
    Ok(())
}
