use super::*;
use pravah::{Chat, ChatBuilder, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

pub(super) fn builder() -> ChatBuilder<String, String> {
    Chat::builder().model("test:///stream").key("conversation")
}

fn controlled_agent(root: Agent<String>) -> Agent<String> {
    root.control(control).configure(configure)
}

async fn configure(input: String, _: Context) -> Result<pravah::AgentConfig, GraphError> {
    Ok(pravah::AgentConfig::new(
        "test:///stream",
        "Answer.",
        pravah::clients::Message::user(input),
    )
    .turn_budget(1))
}

async fn control(
    loop_: pravah::AgentLoop<String>,
    _: Context,
) -> Result<pravah::AgentDecision, GraphError> {
    assert_eq!(loop_.input(), "question");
    assert_eq!(loop_.point(), pravah::AgentInterventionPoint::BeforeModel);
    assert_eq!(loop_.turns_remaining(), Some(1));
    Ok(pravah::AgentDecision::continue_())
}

/// Function-defined Chat retains typed control, plain-text configuration and last-turn completion.
#[tokio::test]
async fn function_defined_chat_keeps_controller_and_configuration() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = Chat::new(controlled_agent, context(Mode::Complete, stats.clone())?)?;
    let mut progress = 0;
    let turn = chat
        .send_stream("question", |_, _| {
            progress += 1;
            std::future::ready(())
        })
        .await?;
    assert_eq!(turn.output, "authoritative answer");
    assert_eq!(progress, 2);
    assert_eq!(
        chat.snapshot()?.history().entries()[0].message.content,
        "question"
    );
    assert_eq!(stats.starts.load(Ordering::SeqCst), 1);
    assert_eq!(stats.ordinary.load(Ordering::SeqCst), 0);
    Ok(())
}

/// Keyed streaming commits only authoritative output, and ordinary send stays non-streaming.
#[tokio::test]
async fn keyed_stream_then_ordinary_turn_preserves_state() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = builder()
        .state(vec!["private state".to_owned()])
        .build(context(Mode::Complete, stats.clone())?)?;
    let mut events = Vec::new();
    let reply = chat
        .send_stream_with_key("question", "message-42", |id, event| {
            events.push((id, event));
            std::future::ready(())
        })
        .await?;
    assert_eq!(reply.output, "authoritative answer");
    assert_eq!(events.len(), 2);
    assert!(
        events
            .iter()
            .all(|(id, event)| *id == events[0].0 && matches!(event, LlmEvent::TextDelta { .. }))
    );
    let snapshot = chat.snapshot()?;
    let entries = snapshot.history().entries();
    assert_eq!(entries[0].message.key.as_deref(), Some("message-42"));
    assert!(entries[1].message.key.is_none());
    assert_eq!(entries[1].message.content, "\"authoritative answer\"");
    assert_eq!(
        snapshot.history().last_usage().and_then(|u| u.total()),
        Some(13)
    );
    assert_eq!(chat.get()?, ["private state"]);
    assert_eq!(chat.send("ordinary question").await?.output, reply.output);
    assert_eq!(stats.starts.load(Ordering::SeqCst), 1);
    assert_eq!(stats.ordinary.load(Ordering::SeqCst), 1);
    Ok(())
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Lookup {
    query: String,
}

fn tools(tools: Toolset) -> Toolset {
    tools.tool(lookup)
}

async fn lookup(request: Lookup, _: Context) -> Result<String, pravah::tools::ToolError> {
    Ok(format!("found {}", request.query))
}

/// Fragmentary arguments stay provisional; completed calls execute before a second correlated stream.
#[tokio::test]
async fn tools_execute_only_from_complete_proposal() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = builder()
        .tools(tools)
        .turn_budget(3)
        .tool_budget::<Lookup>(1)
        .build(context(Mode::ToolRound, stats.clone())?)?;
    let mut ids = Vec::new();
    let mut tools_seen = 0;
    let turn = chat
        .send_stream("research", |id, event| {
            ids.push(id);
            if let LlmEvent::ToolCallDelta {
                index,
                arguments_delta,
                ..
            } = event
            {
                assert_eq!(index, 7);
                assert_eq!(arguments_delta, "{");
                assert_eq!(stats.tool_results_seen.load(Ordering::SeqCst), 0);
                tools_seen += 1;
            }
            std::future::ready(())
        })
        .await?;
    assert_eq!(turn.output, "authoritative answer");
    assert_eq!(tools_seen, 1);
    assert_ne!(ids.first(), ids.last());
    assert_eq!(stats.starts.load(Ordering::SeqCst), 2);
    assert_eq!(stats.tool_results_seen.load(Ordering::SeqCst), 1);
    let snapshot = chat.snapshot()?;
    let entries = snapshot.history().entries();
    assert_eq!(entries.len(), 4);
    assert!(
        matches!(&entries[1].message.role, pravah::clients::Role::AssistantToolCalls { calls } if calls.len() == 1)
    );
    assert!(
        matches!(&entries[2].message.role, pravah::clients::Role::Tool { call_id } if call_id == "call-1")
    );
    assert_eq!(stats.ordinary.load(Ordering::SeqCst), 0);
    Ok(())
}
