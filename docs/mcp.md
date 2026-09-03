# MCP Resources and Agent Tool Filters

Pravah agents can receive text resources from Streamable HTTP MCP servers and
select a subset of their prepared tools for each invocation. Resources provide
context; tools provide actions. They are configured independently.

Enable MCP support:

```toml
[dependencies]
pravah = { version = "0.4.9", features = ["mcp"] }
```

## Register an MCP Server

Server locations and credentials belong to the runtime `Context`:

```rust
use pravah::graph::McpServer;
use pravah::Context;

let ctx = Context::default().with_mcp_server(
    McpServer::new("handbook", "https://mcp.example.com")
        .bearer_token(token)
        .header("x-tenant", tenant_id),
);
```

The identifier `handbook` is application-defined. Agent configuration uses it
to select resources without storing the server URL or credentials in the graph
or snapshot. Register the server again when restoring in another process.

## Inspect and Select Resources

The configuration function can inspect the server's sorted resource and
resource-template catalog before selecting context for one invocation:

```rust
use pravah::clients::Message;
use pravah::graph::{AgentConfig, GraphError, McpResourceRef};
use pravah::Context;

async fn configure_reviewer(
    request: ReviewRequest,
    ctx: Context,
) -> Result<AgentConfig, GraphError> {
    let catalog = ctx
        .mcp_resources("handbook")
        .await
        .map_err(|error| GraphError::McpResource(error.to_string()))?;

    let resource = catalog
        .iter()
        .find(|item| item.uri() == request.policy_uri && !item.is_template())
        .ok_or_else(|| {
            GraphError::McpResource(format!(
                "policy resource '{}' is unavailable",
                request.policy_uri,
            ))
        })?;

    Ok(AgentConfig::new(
        "openai:///gpt-5",
        "Review the request using the selected policy.",
        Message::user(request.text),
    )
    .resources([McpResourceRef::new("handbook", resource.uri())]))
}
```

Use `McpResourceRef::template` when the catalog entry is a URI template:

```rust
use std::collections::BTreeMap;

let arguments = BTreeMap::from([
    ("team".to_string(), request.team),
    ("region".to_string(), request.region),
]);

let resource = McpResourceRef::template(
    "handbook",
    "policy://{team}/{region}",
    arguments,
);
```

Selected resources retain their declared order. Pravah accepts text resources,
rejects duplicate references and blob content, and resolves all references
before the first model dispatch.

## Declare and Filter Tools

Declare the complete candidate toolset in the agent definition:

```rust
use pravah::graph::{Agent, Toolset};

fn review_tools(tools: Toolset) -> Toolset {
    tools
        .tool::<ReadFile>()
        .tool_handler(search_knowledge)
        .flow(verify_claim)
}

fn reviewer(root: Agent<ReviewRequest>) -> Agent<Review> {
    root.tools(review_tools).configure(configure_reviewer)
}
```

Then select a subset while configuring a particular invocation:

```rust
use pravah::graph::ToolFilter;

let allow_search = request.allow_external_search;

let config = AgentConfig::new(
    "openai:///gpt-5",
    "Review the request.",
    Message::user(request.text),
)
.tool_filter(ToolFilter::new(move |tool| {
    allow_search || tool.name() != "search_knowledge"
}))
.resources([resource]);
```

The filter may capture invocation-specific values. It receives read-only
`ToolInfo`, including the tool name, description, and input schema. It cannot
add undeclared tools or alter their prepared order.

An agent controller may further change the visible subset between model turns:

```rust
let decision = AgentDecision::redirect()
    .guidance("Use only policy resources for the next step.")
    .tools(ToolFilter::new(|tool| tool.name() == "search_policy"));
```

This selection is limited to tools accepted by the original invocation
configuration and remains active until another redirect changes it. MCP text
already resolved for the invocation remains available as system context;
changing tool visibility neither rereads resources nor rewrites conversation
history.

Tool names are derived from their Rust input types. For example,
`SearchKnowledge` becomes `search_knowledge`.

## Restoration

The resolved model configuration, selected tool identities, resource text, and
resource provenance are checkpointed. Restoring a snapshot does not rerun the
configuration function or reread MCP resources. Live server registration and
model clients remain runtime services and must be supplied again for future
agent invocations.

See [`graph_agent_mcp`](../examples/graph_agent_mcp.rs) for the complete
runnable example.
