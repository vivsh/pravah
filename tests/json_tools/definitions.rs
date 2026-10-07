use super::*;
use pravah::clients::Message;
use pravah::tools::ToolError;
use pravah::{Agent, AgentConfig, Context, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct FindStaffMember {
    staff_id: u32,
}

async fn typed(input: FindStaffMember, _: Context) -> Result<u32, ToolError> {
    Ok(input.staff_id)
}

fn typed_tools(tools: Toolset) -> Toolset {
    tools.tool(typed)
}

/// Typed handlers and catalogue aliases share existing dispatch without changing typed result rendering.
#[tokio::test]
async fn typed_and_json_tools_can_be_mixed() -> Result<(), TestError> {
    let mut catalogue = catalogue();
    catalogue[0]["alias"] = json!("lookup_staff");
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            ToolCall::new("json".into(), "lookup_staff".into(), json!({"staff_id":42})),
            staff_call(),
        ])
        .then_output(json!("Done"));
    let proxy = Arc::new(Proxy::default());
    let mut chat = builder(catalogue)
        .tools(typed_tools)
        .build(context(&factory, &proxy)?)?;
    assert_eq!(chat.send("Find staff 42").await?.output, "Done");
    assert_eq!(
        tool_outputs(&chat.snapshot()?),
        vec![json!({"staff_id":42,"email":null}), json!(42)]
    );
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Duplicate names, reserved identities and invalid named budgets fail before callbacks run.
#[test]
fn invalid_definitions_fail_build() -> Result<(), TestError> {
    let factory = ScriptedFactory::new();
    let proxy = Arc::new(Proxy::default());
    let ctx = context(&factory, &proxy)?;
    let mut duplicate = catalogue();
    duplicate[1]["alias"] = duplicate[0]["alias"].clone();
    assert!(builder(duplicate).build(ctx.clone()).is_err());
    for name in ["", "__rath_final_output"] {
        let mut invalid = catalogue();
        invalid[0]["alias"] = json!(name);
        assert!(builder(invalid).build(ctx.clone()).is_err());
    }
    let mut permitted = catalogue();
    permitted[0]["alias"] = json!("__invalid_tool__");
    assert!(builder(permitted).build(ctx.clone()).is_ok());
    assert!(
        builder(catalogue())
            .tools(typed_tools)
            .build(ctx.clone())
            .is_err()
    );
    assert!(
        builder(catalogue())
            .tool_budget_named("missing", 1)
            .build(ctx.clone())
            .is_err()
    );
    assert!(
        builder(catalogue())
            .tool_budget_named("find_staff_member", 0)
            .build(ctx.clone())
            .is_err()
    );
    assert!(
        builder(catalogue())
            .tool_budget_named("find_staff_member", 1)
            .tool_budget_named("find_staff_member", 2)
            .build(ctx.clone())
            .is_err()
    );
    assert!(
        builder(catalogue())
            .tool_budget::<FindStaffMember>(1)
            .tool_budget_named("find_staff_member", 1)
            .build(ctx)
            .is_err()
    );
    assert!(factory.calls().is_empty());
    Ok(())
}

/// AgentConfig named budgets use the existing typed budget identity and validation boundary.
#[tokio::test]
async fn configuration_budget_names_share_typed_identity() -> Result<(), TestError> {
    async fn configure(_: String, _: Context) -> Result<AgentConfig, GraphError> {
        Ok(
            AgentConfig::new("test:///test", "assist", Message::user("question"))
                .tool_budget::<FindStaffMember>(1)
                .tool_budget_named("find_staff_member", 1),
        )
    }
    fn agent(input: Agent<String>) -> Agent<String> {
        input.tools(typed_tools).configure(configure)
    }
    let factory = ScriptedFactory::new();
    let proxy = Arc::new(Proxy::default());
    let mut chat = pravah::Chat::new(agent, context(&factory, &proxy)?)?;
    assert!(chat.send("question").await.is_err());
    assert!(factory.calls().is_empty());
    Ok(())
}
