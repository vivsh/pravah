use super::*;
use crate::GraphError;
use crate::clients::Message;

/// Generation exposes routing directly while keeping sensitive payloads out of Debug.
#[test]
fn generation_routing_is_borrowed_and_round_trips() -> Result<(), GraphError> {
    let request = AgentRequest {
        version: AGENT_REQUEST_VERSION,
        id: Uuid::nil(),
        persist: None,
        operation: Arc::new(AgentOperation::Generate {
            session_id: "key:tenant/conversation".into(),
            model: "openai:///model".into(),
            options: Value::NULL,
            entries: Vec::new(),
            guidance: vec![Message::user("private input")],
            budget_conclusion: false,
            compact: false,
            last_usage: None,
            total_input: None,
            total_output: None,
        }),
    };
    assert_eq!(request.kind(), "generate");
    assert_eq!(request.provider(), Some("openai"));
    assert_eq!(request.model(), Some("openai:///model"));
    assert_eq!(request.conversation_key(), Some("tenant/conversation"));
    let debug = format!("{request:?}");
    assert!(!debug.contains("private input"));
    assert!(!debug.contains("openai"));
    let bytes = serde_json::to_vec(&request).map_err(codec)?;
    let restored: AgentRequest = serde_json::from_slice(&bytes).map_err(codec)?;
    assert_eq!(restored.provider(), request.provider());
    let mut bytes = Vec::new();
    ciborium::into_writer(&request, &mut bytes).map_err(codec)?;
    let restored: AgentRequest = ciborium::from_reader(bytes.as_slice()).map_err(codec)?;
    assert_eq!(restored.conversation_key(), request.conversation_key());
    Ok(())
}

/// Operations without a resolved model do not invent a provider for task routing.
#[test]
fn history_routing_needs_no_runtime() -> Result<(), GraphError> {
    let request = AgentRequest {
        version: AGENT_REQUEST_VERSION,
        id: Uuid::nil(),
        persist: None,
        operation: Arc::new(AgentOperation::PersistHistory),
    };
    assert_eq!(request.kind(), "persist_history");
    assert_eq!(request.conversation_key(), None);
    assert_eq!(request.provider(), None);
    assert_eq!(request.handler(), None);
    Ok(())
}

/// Cloning requests and inspecting routing metadata never copies the owned operation.
#[test]
fn shared_request_routing_allocates_nothing() {
    let request = AgentRequest {
        version: AGENT_REQUEST_VERSION,
        id: Uuid::nil(),
        persist: Some(
            vec![HistoryEntry {
                id: Uuid::nil(),
                position: 0,
                session_id: "key:conversation".into(),
                agent_id: "agent".into(),
                evicted: false,
                message: Message::user("archived input".repeat(4096)),
            }]
            .into(),
        ),
        operation: Arc::new(AgentOperation::Generate {
            session_id: "key:conversation".into(),
            model: "openrouter:///provider/model".into(),
            options: Value::NULL,
            entries: Vec::new(),
            guidance: vec![Message::user("large input".repeat(4096))],
            budget_conclusion: false,
            compact: false,
            last_usage: None,
            total_input: None,
            total_output: None,
        }),
    };
    let allocations = allocation_counter::measure(|| {
        for _ in 0..100 {
            let copy = std::hint::black_box(request.clone());
            std::hint::black_box(copy.kind());
            std::hint::black_box(copy.provider());
            std::hint::black_box(copy.model());
            std::hint::black_box(copy.handler());
            std::hint::black_box(copy.conversation_key());
            assert!(Arc::ptr_eq(&copy.operation, &request.operation));
            assert!(Arc::ptr_eq(
                copy.persist.as_ref().expect("batch"),
                request.persist.as_ref().expect("batch")
            ));
        }
    });
    assert_eq!(allocations.count_total, 0);
}

/// Unknown versions and fields fail at the external protocol boundary.
#[test]
fn malformed_request_is_rejected() {
    let request = serde_json::json!({"version": 2, "id": Uuid::nil(), "operation": "load_history", "key": "one"});
    assert!(serde_json::from_value::<AgentRequest>(request).is_err());
    let request = serde_json::json!({"version": 1, "id": Uuid::nil(), "operation": "load_history", "key": "one", "extra": true});
    assert!(serde_json::from_value::<AgentRequest>(request).is_err());
}

fn codec(error: impl fmt::Display) -> GraphError {
    GraphError::Invalid(error.to_string())
}
