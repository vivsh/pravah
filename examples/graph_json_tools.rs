//! Register catalogue-shaped tool definitions without endpoint-specific Rust types.
//!
//! Run with `cargo run --example graph_json_tools --features testing`; no credentials needed.

mod support;

use pravah::clients::{Role, ToolCall, ToolDefinition};
use pravah::deps::Deps;
use pravah::testing::{ScriptedFactory, providers};
use pravah::tools::ToolError;
use pravah::{Chat, Context};
use serde_json::{Value, json};
use std::sync::Arc;
use support::ExampleError;

/// Application-owned service shared by every catalogue operation.
struct CatalogueProxy;

impl CatalogueProxy {
    /// Demonstrates trusted routing; a real proxy checks current permissions before HTTP dispatch.
    async fn execute(&self, operation: &str, args: Value) -> Result<Value, ToolError> {
        match operation {
            "staff.retrieve" => Ok(json!({"staff_id":args["staff_id"],"email":null})),
            "visits.list" => Ok(json!([{"status":"arrived"}])),
            _ => Err(ToolError::Security("operation is not allowlisted".into())),
        }
    }
}

/// Supplies the same definition/schema/binding values an application could load from a catalogue.
fn catalogue() -> Vec<(&'static str, ToolDefinition, Value)> {
    let input = json!({"type":"object","properties":{"staff_id":{"type":"integer","minimum":1}},
        "required":["staff_id"],"additionalProperties":false});
    vec![
        (
            "staff.retrieve",
            ToolDefinition::new(
                "find_staff_member".into(),
                "Find staff".into(),
                input.clone(),
            ),
            json!({"type":"object","required":["staff_id","email"],"additionalProperties":false,
                "properties":{"staff_id":{"type":"integer"},"email":{"type":["string","null"],"format":"email"}}}),
        ),
        (
            "visits.list",
            ToolDefinition::new("list_staff_visits".into(), "List visits".into(), input),
            json!({"type":"array","items":{"type":"object","required":["status"],
                "properties":{"status":{"enum":["arrived","left"]}},"additionalProperties":false}}),
        ),
    ]
}

/// Runs two aliases through one Context service and prints their exact successful JSON values.
#[tokio::main]
async fn main() -> Result<(), ExampleError> {
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
        .then_output(json!("Staff and visits found."));
    let mut deps = Deps::default();
    deps.insert(Arc::new(CatalogueProxy));
    let ctx = Context::default()
        .with_deps(deps)
        .with_providers(providers(factory)?);
    let records = catalogue();
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .instructions("Assist staff")
        .tool_budget_named("find_staff_member", 1)
        .tools(move |mut tools| {
            for (operation, definition, output) in records {
                tools = tools.json(definition, Some(output), move |args, ctx| async move {
                    ctx.require::<CatalogueProxy>()?
                        .execute(operation, args)
                        .await
                });
            }
            tools
        })
        .build(ctx)?;
    println!(
        "{}",
        chat.send("Find staff 42 and their visits").await?.output
    );
    for entry in chat.snapshot()?.history().entries() {
        if matches!(entry.message.role, Role::Tool { .. }) {
            println!("{}", entry.message.content);
        }
    }
    Ok(())
}
