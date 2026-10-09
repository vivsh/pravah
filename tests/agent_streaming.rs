//! Deterministic streaming workers preserve the synchronous VM's completion boundary.

use pravah::clients::{LlmEvent, Provider};
use pravah::{
    Agent, AgentExecutor, AgentRequest, Context, Flow, GraphError, Runtime, Step, compile,
};
use std::sync::Arc;
use uuid::Uuid;

#[path = "agent_streaming/fixtures.rs"]
mod fixtures;
use fixtures::{Mode, Stats, context};

#[path = "agent_streaming/chat.rs"]
mod chat;
#[path = "agent_streaming/failures.rs"]
mod failures;
#[path = "agent_streaming/history.rs"]
mod history;
#[path = "agent_streaming/lifecycle.rs"]
mod lifecycle;

fn assistant(root: Agent<String>) -> Agent<String> {
    root.model("test:///stream").key("conversation").build()
}

fn workflow(root: Flow<String>) -> Flow<String> {
    root.agent(assistant)
}

/// Stops after configuration without running a model or introducing a hidden test retry.
async fn generation(
    runtime: &mut Runtime,
    executor: &AgentExecutor,
) -> Result<AgentRequest, GraphError> {
    for _ in 0..100 {
        if let Step::Agent(request) = runtime.next()? {
            if request.kind() == "generate" {
                return Ok(request);
            }
            runtime.resume_agent(executor.execute(&request).await)?;
        }
    }
    Err(GraphError::Invalid("missing generation".into()))
}

/// A JSON-restored pending request accepts one streamed response, never provisional text.
#[tokio::test]
async fn executor_stream_completes_restored_graph() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let flow = compile(workflow)?;
    let executor = flow
        .prepared()
        .executor(context(Mode::Complete, stats.clone())?);
    let mut runtime = flow.start("question".into(), Uuid::nil())?;
    let request = generation(&mut runtime, &executor).await?;
    let encoded = serde_json::to_vec(&runtime.snapshot()?).map_err(codec)?;
    let mut restored = flow.restore(serde_json::from_slice(&encoded).map_err(codec)?)?;
    let mut progress = Vec::new();
    let response = executor
        .execute_stream(&request, |id, event| {
            assert_eq!(id, request.id());
            progress.push(event);
            std::future::ready(())
        })
        .await;
    assert!(response.outcome().is_ok());
    assert_eq!(
        progress
            .iter()
            .filter_map(|event| match event {
                LlmEvent::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        ["provisional ", "text"]
    );
    assert_eq!(runtime.history().entries().len(), 1);
    restored.resume_agent(response)?;
    complete(&mut restored, &flow)?;
    assert_eq!(stats.starts.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(stats.ordinary.load(std::sync::atomic::Ordering::SeqCst), 0);
    Ok(())
}

/// Interprets an accepted completion without redispatching any external operation.
fn complete(
    runtime: &mut Runtime,
    flow: &pravah::CompiledFlow<String, String>,
) -> Result<(), GraphError> {
    for _ in 0..100 {
        if let Step::Done(value) = runtime.next()? {
            assert_eq!(flow.decode_output(value)?, "authoritative answer");
            assert_eq!(
                runtime
                    .history()
                    .entries()
                    .last()
                    .map(|e| e.message.content.as_str()),
                Some("\"authoritative answer\"")
            );
            return Ok(());
        }
    }
    Err(GraphError::Invalid("flow did not complete".into()))
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::Invalid(error.to_string())
}
