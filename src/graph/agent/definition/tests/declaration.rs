use super::*;
use crate::graph::{AgentDecision, AgentLoop};
use crate::tools::ToolError;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}

fn base() -> Agent<String> {
    Agent::root().model("test:///model").instructions("Answer.")
}

async fn configure(input: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///model",
        "Dynamic.",
        Message::user(input),
    ))
}

async fn control(_: AgentLoop<String>, _: Context) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::continue_())
}

async fn search(_: Search, _: Context) -> Result<String, ToolError> {
    Ok("result".into())
}

/// Invalid declarations accumulate errors without accepting conflicting or repeated terminal methods.
#[test]
fn rejects_invalid_declarations() {
    let reference = McpResourceRef::new("docs", "docs://guide");
    let cases: Vec<(&str, Agent<String>)> = vec![
        ("missing model", Agent::<String>::root().build()),
        ("empty model", base().model(" ").build()),
        ("empty key", base().key(" ").build()),
        ("zero output", base().max_output_tokens(0).build()),
        ("zero turns", base().turn_budget(0).build()),
        (
            "repeated turns",
            base().turn_budget(2).turn_budget(3).build(),
        ),
        ("unknown tool", base().tool_budget::<Search>(1).build()),
        ("zero calls", base().tool_budget::<Search>(0).build()),
        (
            "repeated calls",
            base()
                .tool_budget::<Search>(1)
                .tool_budget_named("search", 2)
                .build(),
        ),
        (
            "bad resource",
            base()
                .resources([McpResourceRef::new("", "docs://guide")])
                .build(),
        ),
        (
            "duplicate resource",
            base().resources([reference.clone(), reference]).build(),
        ),
    ];
    for (name, agent) in cases {
        assert!(!agent.definition.errors.is_empty(), "{name}");
    }
}

/// Conflicting terminal methods and operations after finalization are never silently accepted.
#[test]
fn rejects_conflicting_terminal_methods() {
    let cases: Vec<(&str, Agent<String>)> = vec![
        ("settings then configure", base().configure(configure)),
        (
            "configure then settings",
            Agent::root().configure(configure).model("test:///other"),
        ),
        ("build twice", base().build::<String>().build()),
        (
            "build then configure",
            base().build::<String>().configure(configure),
        ),
        (
            "configure then build",
            Agent::root().configure::<String, _, _>(configure).build(),
        ),
        (
            "setter after build",
            base().build::<String>().instructions("Late"),
        ),
        (
            "tools after build",
            base().build::<String>().tools(|tools| tools.tool(search)),
        ),
        (
            "control after build",
            base().build::<String>().control(control),
        ),
        (
            "repeated control",
            base().control(control).control(control).build(),
        ),
    ];
    for (name, agent) in cases {
        assert!(!agent.definition.errors.is_empty(), "{name}");
    }
}

/// Scalar replacement, budgets and reference ordering use one payload and no retained declaration copy.
#[tokio::test]
async fn finalized_settings_have_one_owner() -> Result<(), GraphError> {
    let references = [
        McpResourceRef::new("docs", "docs://second"),
        McpResourceRef::new("docs", "docs://first"),
    ];
    let agent = configured_agent(references.clone())?;
    assert!(agent.definition.settings.is_none());
    assert!(agent.definition.instructions.is_none());
    assert!(agent.definition.controller.is_some());
    let data = agent
        .definition
        .configuration
        .as_ref()
        .ok_or_else(|| GraphError::Invalid("data missing".into()))?;
    let handler = agent
        .definition
        .configure
        .as_ref()
        .ok_or_else(|| GraphError::Invalid("handler missing".into()))?;
    handler.validate_data(Some(data))?;
    let config = handler
        .configure(
            Value::from("question"),
            Some(&data.value),
            Uuid::nil(),
            Context::default(),
        )
        .await?;
    assert_eq!(config.message.content, "\"question\"");
    assert_eq!(config.model, "test:///replacement");
    assert_eq!(config.instructions, "Replacement.");
    assert_eq!(config.key.as_deref(), Some("thread"));
    assert_eq!(config.max_output_tokens, Some(512));
    assert_eq!(config.turn_budget, Some(3));
    assert_eq!(config.tool_budgets[0].limit, 2);
    assert_eq!(config.resources, references);
    assert_eq!(
        config.provider_config,
        Some(serde_json::json!({"temperature":0.2}))
    );
    Ok(())
}

/// Exercises every shared setting with the same tool and controller definition.
fn configured_agent(references: [McpResourceRef; 2]) -> Result<Agent<String>, GraphError> {
    base()
        .model("test:///replacement")
        .instructions("Replacement.")
        .key("thread")
        .provider_config(serde_json::json!({"temperature":0.2}))
        .max_output_tokens(0)
        .max_output_tokens(512)
        .resources(references)
        .tools(|tools| tools.tool(search))
        .control(control)
        .turn_budget(3)
        .tool_budget::<Search>(2)
        .build_checked()
}

/// Instructions alone are declarative, and cannot silently mix with custom configure.
#[test]
fn rejects_instruction_only_conflict() {
    let agent: Agent<String> = Agent::root().instructions("").configure(configure);
    assert!(!agent.definition.errors.is_empty());
}
