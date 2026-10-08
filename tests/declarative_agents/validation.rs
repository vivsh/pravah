use super::*;
use pravah::clients::Message;
use pravah::{AgentConfig, Chat, ChatBuilder};

fn invalid(root: Agent<Question>) -> Agent<Notes> {
    root.model("test:///model").turn_budget(0).build()
}

fn invalid_flow(root: Flow<Question>) -> Flow<Notes> {
    root.agent(invalid)
}

async fn configure(question: Question, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///model",
        "Answer",
        Message::user(question.topic),
    ))
}

fn conflicting(root: Agent<Question>) -> Agent<Notes> {
    root.instructions("Fixed").configure(configure)
}

fn conflicting_flow(root: Flow<Question>) -> Flow<Notes> {
    root.agent(conflicting)
}

/// Public graph compilation rejects invalid declarations and mixed configuration modes before execution.
#[test]
fn compilation_rejects_declaration_errors() -> Result<(), GraphError> {
    for (flow, expected) in [
        (
            invalid_flow as fn(Flow<Question>) -> Flow<Notes>,
            "turn budget",
        ),
        (conflicting_flow, "cannot be combined"),
    ] {
        let Err(GraphError::Invalid(reason)) = compile(flow) else {
            return Err(GraphError::Invalid(format!(
                "missing declaration error for {expected}"
            )));
        };
        assert!(reason.contains(expected), "{reason}");
    }
    Ok(())
}

/// Chat's terminal boundary uses the same validation while retaining its fallible configuration error.
#[test]
fn chat_and_agent_share_validation() {
    let cases: Vec<ChatBuilder<String, String>> = vec![
        Chat::builder(),
        Chat::builder().model(" "),
        Chat::builder().model("test:///model").key(" "),
        Chat::builder().model("test:///model").turn_budget(0),
        Chat::builder()
            .model("test:///model")
            .turn_budget(2)
            .turn_budget(3),
        Chat::builder()
            .model("test:///model")
            .tool_budget_named("missing", 1),
        Chat::builder().model("test:///model").max_output_tokens(0),
    ];
    for builder in cases {
        assert!(matches!(
            builder.build(Context::default()),
            Err(GraphError::AgentConfigValidation(_))
        ));
    }
}

/// The shared schema has a new identity while retaining the existing settings wire fields.
#[test]
fn graph_settings_schema_is_explicit() -> Result<(), GraphError> {
    let flow = compile(research)?;
    let encoded = serde_json::to_value(flow.graph()).map_err(codec)?;
    let text = serde_json::to_string(&encoded).map_err(codec)?;
    assert!(text.contains("AgentSettings"));
    assert!(!text.contains("ChatSettings"));
    let graph = serde_json::from_value(encoded).map_err(codec)?;
    let prepared = pravah::graph::PreparedGraph::new(graph, flow.registry().clone())?;
    assert_eq!(prepared.fingerprint(), flow.prepared().fingerprint());
    Ok(())
}
