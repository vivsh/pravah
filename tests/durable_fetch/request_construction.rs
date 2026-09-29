use super::*;
use pravah::clients::{ToolChoice, ToolDefinition};
use pravah::graph::{FetchBody, fetch::rath::RathRequest, from_value};
use pravah::tools::ToolError;
use pravah::{Agent, AgentConfig, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}

async fn search(_: Search, _: Context) -> Result<String, ToolError> {
    Ok("found".into())
}

fn tools(root: Toolset) -> Toolset {
    root.tool(search)
}

fn agent(root: Agent<String>) -> Agent<String> {
    root.tools(tools).configure(configure)
}

/// Supplies opaque nested configuration to expose any repeated tree construction.
async fn configure(memory: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///recorded",
        "Instructions",
        Message::user("question"),
    )
    .memory(memory)
    .provider_config(serde_json::json!({"opaque": vec!["large".repeat(1000); 32]}))
    .max_output_tokens(123))
}

fn flow(root: Flow<String>) -> Flow<String> {
    root.agent(agent)
}

fn body(body: Option<&FetchBody>) -> Result<&Value, GraphError> {
    match body {
        Some(FetchBody::Value(value)) => Ok(value),
        _ => Err(codec("missing body")),
    }
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, GraphError> {
    value.get(key).ok_or_else(|| codec(key))
}

/// Construction shares authored/configured trees and retains ordinary Rath option encoding.
#[tokio::test]
async fn generation_construction_shares_immutable_values() -> Result<(), GraphError> {
    let workflow = compile(flow)?;
    let executor = pravah::FetchExecutor::new(Context::default())
        .with_registry(Arc::new(workflow.registry().clone()));
    for memory in ["", "  Unicode \u{1f30a}\n "] {
        let mut runtime = workflow.start(memory.into(), Uuid::nil())?;
        let configure = next_fetch(&mut runtime)?;
        let payload = field(body(configure.request().body_ref())?, "payload")?.clone();
        let configured = executor.execute(&configure).await?;
        let resolved = field(body(configured.body_ref())?, "resolved")?.clone();
        runtime.resume_fetch(configure.id(), Ok(configured))?;
        let preparation = loop {
            let fetch = next_fetch(&mut runtime)?;
            if fetch.request().url() == "pravah://prepare" {
                break fetch;
            }
            runtime.resume_fetch(fetch.id(), Ok(executor.execute(&fetch).await?))?;
        };
        let request = field(body(preparation.request().body_ref())?, "request")?;
        check_request(request, &payload, &resolved, memory)?;
    }
    Ok(())
}

/// Compares the shared encoder against the original public Rath constructor and codec.
fn check_request(
    request: &Value,
    payload: &Value,
    resolved: &Value,
    memory: &str,
) -> Result<(), GraphError> {
    let options = field(request, "options")?;
    let schema = field(field(options, "response_format")?, "schema")?;
    let provider = field(options, "provider_config")?;
    assert!(std::ptr::eq(
        field(schema, "type")?,
        field(field(payload, "output_schema")?, "type")?
    ));
    assert!(std::ptr::eq(
        field(provider, "opaque")?,
        field(field(resolved, "provider_config")?, "opaque")?
    ));
    let tool = field(payload, "tools")?
        .as_array()
        .and_then(|v| v.first())
        .ok_or_else(|| codec("tool"))?;
    let definition = ToolDefinition::new(
        from_value(field(tool, "name")?.clone()).map_err(codec)?,
        from_value(field(tool, "description")?.clone()).map_err(codec)?,
        from_value(field(tool, "parameters")?.clone()).map_err(codec)?,
    );
    let expected = ClientOptions::default()
        .with_name(
            field(payload, "agent_id")?
                .as_str()
                .ok_or_else(|| codec("identity"))?,
        )
        .with_preamble(format!("Instructions\n\n<memory>\n{memory}\n</memory>"))
        .with_output_schema(from_value(field(payload, "output_schema")?.clone()).map_err(codec)?)
        .with_tools(vec![definition])
        .with_tool_choice(ToolChoice::Auto)
        .with_max_output_tokens(123)
        .with_provider_config(from_value::<serde_json::Value>(provider.clone()).map_err(codec)?);
    let expected = RathRequest::new("openai:///recorded", expected, Vec::new());
    assert_eq!(request, &pravah::graph::to_value(expected).map_err(codec)?);
    Ok(())
}
