//! Declarative agents use the same typed graph and durable worker boundary as configured agents.

use pravah::testing::{ScriptedFactory, providers};
use pravah::{Agent, Context, Flow, GraphError, Step, compile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[path = "support/host.rs"]
mod host;

#[path = "declarative_agents/composition.rs"]
mod composition;
#[path = "declarative_agents/restore.rs"]
mod restore;
#[path = "declarative_agents/validation.rs"]
mod validation;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Question {
    topic: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Notes {
    finding: String,
}

fn researcher(root: Agent<Question>) -> Agent<Notes> {
    root.model("test:///model")
        .instructions("Research carefully.")
        .key("research")
        .turn_budget(3)
        .max_output_tokens(500)
        .build()
}

fn research(root: Flow<Question>) -> Flow<Notes> {
    root.agent(researcher)
}

/// A non-Clone typed input renders once and a pending generation restores without reconfiguration.
#[tokio::test]
async fn typed_declaration_executes_after_restore() -> Result<(), GraphError> {
    let script = ScriptedFactory::new().then_output(serde_json::json!({"finding":"verified"}));
    let flow = compile(research)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(providers(script.clone())?));
    let mut runtime = flow.start(
        Question {
            topic: "durability".into(),
        },
        Uuid::nil(),
    )?;
    assert!(script.calls().is_empty());
    let request = pending_generation(&mut runtime, &executor).await?;
    let snapshot = runtime.snapshot()?;
    let encoded = serde_json::to_vec(&snapshot).map_err(codec)?;
    let mut restored = flow.restore(serde_json::from_slice(&encoded).map_err(codec)?)?;
    assert_eq!(
        restored.pending_agent().map(|pending| pending.id()),
        Some(request.id())
    );
    restored.resume_agent(executor.execute(&request).await)?;
    let Step::Done(value) = host::finish(&mut restored, &executor).await? else {
        return Err(GraphError::Invalid("expected completed research".into()));
    };
    assert_eq!(
        flow.decode_output(value)?,
        Notes {
            finding: "verified".into()
        }
    );
    let calls = script.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        runtime.history().entries()[0].message.content,
        "{\"topic\":\"durability\"}"
    );
    Ok(())
}

/// Stops at the generation boundary after committing declarative configuration.
async fn pending_generation(
    runtime: &mut pravah::Runtime,
    executor: &pravah::AgentExecutor,
) -> Result<pravah::AgentRequest, GraphError> {
    for _ in 0..100 {
        if let Step::Agent(request) = runtime.next()? {
            if request.model().is_some() {
                return Ok(request);
            }
            runtime.resume_agent(executor.execute(&request).await)?;
        }
    }
    Err(GraphError::Invalid("generation request missing".into()))
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::Invalid(error.to_string())
}
