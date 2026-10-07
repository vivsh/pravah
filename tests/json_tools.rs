use pravah::clients::{Role, ToolCall};
use pravah::testing::ScriptedFactory;
use pravah::{AgentResponse, ChatRequest, ChatStep, GraphError, Snapshot};
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};

#[path = "json_tools/definitions.rs"]
mod definitions;
#[path = "json_tools/fixtures.rs"]
mod fixtures;
#[path = "json_tools/restore.rs"]
mod restore;
#[path = "json_tools/snapshots.rs"]
mod snapshots;
use fixtures::*;

/// Two catalogue schemas/aliases use one Context proxy and preserve canonical provider metadata.
#[tokio::test]
async fn two_catalogue_tools_share_a_context_proxy() -> Result<(), TestError> {
    let catalogue = catalogue();
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            ToolCall::new(
                "staff".into(),
                "find_staff_member".into(),
                json!({"staff_id":42}),
            ),
            ToolCall::new(
                "visits".into(),
                "list_staff_visits".into(),
                json!({"staff_id":42}),
            ),
        ])
        .then_output(json!("Found staff and visits."));
    let proxy = Arc::new(Proxy::default());
    let ctx = inspected_context(&factory, &catalogue, &proxy)?;
    let mut chat = builder(catalogue.clone()).build(ctx)?;
    assert_eq!(
        chat.send("Find staff 42 and their visits").await?.output,
        "Found staff and visits."
    );
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 2);
    let snapshot = chat.snapshot()?;
    let outputs = tool_outputs(&snapshot);
    assert_eq!(
        outputs,
        vec![
            json!({"staff_id":42,"email":null}),
            json!([{"status":"arrived"}])
        ]
    );
    assert_eq!(factory.calls().len(), 2);
    assert!(
        builder(catalogue)
            .restore::<()>(snapshot, context(&ScriptedFactory::new(), &proxy)?)
            .is_ok()
    );
    Ok(())
}

/// Incorrect arguments consume the alias budget, return a correctable error and never call the proxy.
#[tokio::test]
async fn invalid_arguments_do_not_execute() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![ToolCall::new(
            "bad".into(),
            "find_staff_member".into(),
            json!({"staff_id":0,"extra":true}),
        )])
        .then_output(json!("Please provide a valid identifier."));
    let proxy = Arc::new(Proxy::default());
    let mut chat = builder(catalogue())
        .tool_budget_named("find_staff_member", 1)
        .build(context(&factory, &proxy)?)?;
    chat.send("Find staff").await?;
    let outputs = tool_outputs(&chat.snapshot()?);
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 0);
    assert_eq!(outputs[0]["error_kind"], "Validation");
    assert_eq!(outputs[0]["tool"], "find_staff_member");
    Ok(())
}

/// An invalid success fails after one proxy call and its accepted failure survives snapshot restore.
#[tokio::test]
async fn invalid_success_is_fatal_without_replay() -> Result<(), TestError> {
    let mut catalogue = catalogue();
    catalogue[0]["output_schema"] = json!({"type":"integer"});
    let factory = ScriptedFactory::new().then_tool_calls(vec![staff_call()]);
    let proxy = Arc::new(Proxy::default());
    let ctx = context(&factory, &proxy)?;
    let mut chat = builder(catalogue.clone()).build(ctx.clone())?;
    let error = chat
        .send("Find staff 42")
        .await
        .expect_err("invalid response");
    assert!(matches!(error, GraphError::AgentFailed { .. }));
    let snapshot = chat.snapshot()?;
    assert!(tool_outputs(&snapshot).is_empty());
    let mut restored = builder(catalogue).restore::<()>(snapshot, ctx)?;
    assert!(matches!(
        restored.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// A caller-delivered success cannot bypass canonical output validation or generate another HTTP call.
#[tokio::test]
async fn external_success_is_validated_before_history() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_tool_calls(vec![staff_call()]);
    let proxy = Arc::new(Proxy::default());
    let mut chat = builder(catalogue()).build(context(&factory, &proxy)?)?;
    chat.submit("Find staff 42")?;
    let request = advance_to_tool(&mut chat).await?;
    chat.resume_agent(AgentResponse::new(
        request.id(),
        Ok(pravah::graph::to_value(
            json!({"kind":"success","value":{"staff_id":"invalid","email":null}}),
        )?),
    ))?;
    assert!(chat.next().is_err());
    assert!(tool_outputs(&chat.snapshot()?).is_empty());
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

/// Checkpointed visibility does not bypass the proxy's current-user permission decision.
#[tokio::test]
async fn resumed_tool_checks_current_permissions() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![staff_call()])
        .then_output(json!("Denied."));
    let proxy = Arc::new(Proxy::default());
    let ctx = context(&factory, &proxy)?;
    let mut chat = builder(catalogue()).build(ctx.clone())?;
    chat.submit(ChatRequest::from("Find staff 42".to_owned()).tools(["find_staff_member"]))?;
    advance_to_tool(&mut chat).await?;
    let snapshot = chat.snapshot()?;
    proxy.denied.store(true, Ordering::SeqCst);
    let mut restored = builder(catalogue()).restore::<()>(snapshot, ctx)?;
    let request = restored
        .pending_agent()
        .ok_or(TestError::Missing("pending tool"))?
        .clone();
    restored.resume_agent(restored.executor().execute(&request).await)?;
    finish(&mut restored).await?;
    let outputs = tool_outputs(&restored.snapshot()?);
    assert_eq!(outputs[0]["error_kind"], "Security");
    assert_eq!(outputs[0]["tool"], "find_staff_member");
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

/// A missing shared Context service fails durably without producing a success or invoking another tool.
#[tokio::test]
async fn missing_context_service_is_fatal() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_tool_calls(vec![staff_call()]);
    let ctx = pravah::Context::default().with_providers(pravah::testing::providers(factory)?);
    let mut chat = builder(catalogue()).build(ctx.clone())?;
    assert!(matches!(
        chat.send("Find staff 42").await,
        Err(GraphError::AgentFailed { .. })
    ));
    let snapshot = chat.snapshot()?;
    assert!(tool_outputs(&snapshot).is_empty());
    let mut restored = builder(catalogue()).restore::<()>(snapshot, ctx)?;
    assert!(matches!(
        restored.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    Ok(())
}
