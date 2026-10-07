//! End-to-end durable agent work, independent of runtime borrowing.

use pravah::clients::Message;
use pravah::testing::{ScriptedFactory, providers};
use pravah::{Agent, AgentConfig, Context, Flow, GraphError, Step, compile};
use serde_json::json;
use uuid::Uuid;

fn workflow(root: Flow<String>) -> Flow<String> {
    root.agent(assistant)
}
fn assistant(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}
async fn configure(input: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new("test:///model", "Answer.", Message::user(input)).key("thread"))
}

/// A recorded generation completion survives restoration and is never redispatched.
#[tokio::test]
async fn worker_completion_restores_without_reexecution() -> Result<(), GraphError> {
    let script = ScriptedFactory::new().then_output(json!("answer"));
    let flow = compile(workflow)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(providers(script.clone())?));
    let mut runtime = flow.start("question".into(), Uuid::nil())?;
    let request = match runtime.next()? {
        Step::Agent(request) => request,
        _ => return Err(GraphError::Invalid("expected configure".into())),
    };
    runtime.resume_agent(executor.execute(&request).await)?;
    let request = loop {
        if let Step::Agent(request) = runtime.next()? {
            break request;
        }
    };
    assert_eq!(request.model(), Some("test:///model"));
    assert_eq!(request.provider(), Some("test"));
    assert_eq!(request.conversation_key(), Some("thread"));
    let snapshot = runtime.snapshot()?;
    let encoded = serde_json::to_vec(&snapshot).map_err(codec)?;
    let mut restored = flow.restore(serde_json::from_slice(&encoded).map_err(codec)?)?;
    let response = executor.execute(&request).await;
    restored.resume_agent(response.clone())?;
    assert!(restored.resume_agent(response).is_err());
    let snapshot = restored.snapshot()?;
    let mut restored = flow.restore(snapshot)?;
    loop {
        match restored.next()? {
            Step::Continue => {}
            Step::Done(value) => {
                assert_eq!(flow.decode_output(value)?, "answer");
                break;
            }
            _ => {
                return Err(GraphError::Invalid(
                    "completed work was dispatched again".into(),
                ));
            }
        }
    }
    assert_eq!(script.calls().len(), 1);
    assert_eq!(restored.history().entries().len(), 2);
    Ok(())
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::Invalid(error.to_string())
}
