use super::*;
use pravah::{AgentRequest, Runtime, Snapshot};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn twice(root: Flow<Question>) -> Flow<Notes> {
    root.agent(researcher)
        .map(|notes| Question {
            topic: notes.finding,
        })
        .agent(researcher)
}

fn question() -> Question {
    Question {
        topic: "durability".into(),
    }
}

/// Covers one unchanged Flow, an unresolved configure request, and the next keyed invocation.
#[tokio::test]
async fn flow_resolves_once_per_invocation_across_restore() -> Result<(), GraphError> {
    let flow = compile(twice)?;
    let mut original = flow.start(question(), Uuid::nil())?;
    let request = next_request(&mut original)?;
    assert_eq!(request.kind(), "configure");
    assert_eq!(request.model(), None);

    for cbor in [false, true] {
        let script = ScriptedFactory::new()
            .then_output(serde_json::json!({"finding":"first"}))
            .then_output(serde_json::json!({"finding":"second"}));
        let registry = providers(script.clone())?
            .register("alternate", script.clone())
            .map_err(codec)?;
        let ctx = Context::default().with_providers(registry);
        let mut runtime = flow.restore(roundtrip(original.snapshot()?, cbor)?)?;
        let first = ctx
            .clone()
            .with_model_resolver(|_| Some("alternate:///first".into()));
        let response = flow.prepared().executor(first).execute(&request).await;
        runtime.resume_agent(response)?;
        let (next, resolutions) = next_model_context(ctx);
        let executor = flow.prepared().executor(next);

        let mut runtime = flow.restore(roundtrip(runtime.snapshot()?, cbor)?)?;
        let generation = next_request(&mut runtime)?;
        assert_eq!(generation.model(), Some("alternate:///first"));
        assert_eq!(generation.provider(), Some("alternate"));
        let mut runtime = flow.restore(roundtrip(runtime.snapshot()?, cbor)?)?;
        assert_eq!(
            runtime.pending_agent().map(AgentRequest::id),
            Some(generation.id())
        );
        assert_eq!(resolutions.load(Ordering::Relaxed), 0);

        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Done(_)
        ));
        assert_eq!(resolutions.load(Ordering::Relaxed), 1);
        assert_keyed_model_switch(&runtime, &script);
    }
    Ok(())
}

/// Checks that both recorded provider calls share one preserved conversation.
fn assert_keyed_model_switch(runtime: &Runtime, script: &ScriptedFactory) {
    let calls = script.calls();
    let models: Vec<_> = calls.iter().map(|(model, _)| model.as_str()).collect();
    assert_eq!(models, ["alternate:///first", "test:///next"]);
    let entries = runtime.history().entries();
    assert_eq!(entries.len(), 4);
    assert!(
        entries
            .iter()
            .all(|entry| entry.session_id == "key:research")
    );
}

fn next_model_context(ctx: Context) -> (Context, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&count);
    let ctx = ctx.with_model_resolver(move |declared| {
        assert_eq!(declared, "test:///model");
        observed.fetch_add(1, Ordering::Relaxed);
        Some("test:///next".into())
    });
    (ctx, count)
}

/// Invalid overrides fail normally and never invoke the declared default as a fallback.
#[tokio::test]
async fn invalid_replacements_do_not_fall_back() -> Result<(), GraphError> {
    let flow = compile(research)?;
    for replacement in [" ", "not-a-url"] {
        let script = ScriptedFactory::new().then_output(serde_json::json!({"finding":"fallback"}));
        let ctx = Context::default()
            .with_model_resolver(|_| Some("test:///unused".into()))
            .with_model_resolver(move |_| Some(replacement.into()))
            .with_providers(providers(script.clone())?);
        let executor = flow.prepared().executor(ctx);
        let mut runtime = flow.start(question(), Uuid::nil())?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await,
            Err(GraphError::AgentFailed { .. })
        ));
        assert!(script.calls().is_empty());
    }
    Ok(())
}

fn next_request(runtime: &mut Runtime) -> Result<AgentRequest, GraphError> {
    for _ in 0..100 {
        match runtime.next()? {
            Step::Agent(request) => return Ok(request),
            Step::Continue => {}
            _ => break,
        }
    }
    Err(GraphError::Invalid("expected agent request".into()))
}

fn roundtrip(snapshot: Snapshot, cbor: bool) -> Result<Snapshot, GraphError> {
    if cbor {
        let mut encoded = Vec::new();
        ciborium::into_writer(&snapshot, &mut encoded).map_err(codec)?;
        ciborium::from_reader(encoded.as_slice()).map_err(codec)
    } else {
        let encoded = serde_json::to_vec(&snapshot).map_err(codec)?;
        serde_json::from_slice(&encoded).map_err(codec)
    }
}
