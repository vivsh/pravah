//! Graph agent with MCP text resources and invocation-specific tool filtering.
//!
//! Requires the `mcp` feature, a Streamable HTTP MCP resource server, and
//! credentials for the model selected by `PRAVAH_MODEL_URL`.
//!
//! Set `PRAVAH_MCP_URL` and `PRAVAH_MCP_RESOURCE_URI` before running.

#[cfg(feature = "mcp")]
mod support;

#[cfg(feature = "mcp")]
mod enabled {
    use std::env;
    use std::sync::Arc;

    use pravah::clients::Message;
    use pravah::deps::Deps;
    use pravah::graph::{
        self, Agent, AgentConfig, Flow, GraphError, McpResourceRef, McpServer, Step, ToolFilter,
        Toolset,
    };
    use pravah::tools::ToolError;
    use pravah::{Context, FlowConf};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::support::ExampleError;

    const MCP_SERVER: &str = "knowledge";

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct Question {
        text: String,
        model: String,
        resource_uri: String,
        allow_search: bool,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Answer {
        text: String,
        sources_used: Vec<String>,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct SearchKnowledge {
        query: String,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct SearchResult {
        matches: Vec<String>,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct AddNumbers {
        left: f64,
        right: f64,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Sum {
        value: f64,
    }

    struct SearchIndex(Vec<String>);

    fn knowledge_tools(tools: Toolset) -> Toolset {
        tools.tool(search_knowledge).tool(add_numbers)
    }

    async fn search_knowledge(
        input: SearchKnowledge,
        ctx: Context,
    ) -> Result<SearchResult, ToolError> {
        let query = input.query.to_lowercase();
        let index = ctx.require::<SearchIndex>()?;
        let matches = index
            .0
            .iter()
            .filter(|entry| entry.to_lowercase().contains(&query))
            .cloned()
            .collect();
        Ok(SearchResult { matches })
    }

    async fn add_numbers(input: AddNumbers, _ctx: Context) -> Result<Sum, ToolError> {
        Ok(Sum {
            value: input.left + input.right,
        })
    }

    fn knowledge_agent(root: Agent<Question>) -> Agent<Answer> {
        root.tools(knowledge_tools).configure(configure_agent)
    }

    async fn configure_agent(question: Question, ctx: Context) -> Result<AgentConfig, GraphError> {
        let resource = select_resource(&ctx, &question.resource_uri).await?;
        let allow_search = question.allow_search;
        Ok(AgentConfig::new(
            question.model,
            "Answer from the selected MCP context. Cite the resource URI in sources_used.",
            Message::user(question.text),
        )
        .tool_filter(ToolFilter::new(move |tool| {
            allow_search || tool.name() != "search_knowledge"
        }))
        .resources([resource]))
    }

    async fn select_resource(
        ctx: &Context,
        requested_uri: &str,
    ) -> Result<McpResourceRef, GraphError> {
        let catalog = ctx
            .mcp_resources(MCP_SERVER)
            .await
            .map_err(|error| GraphError::McpResource(error.to_string()))?;
        let selected = catalog
            .iter()
            .find(|resource| resource.uri() == requested_uri && !resource.is_template())
            .ok_or_else(|| {
                GraphError::McpResource(format!(
                    "resource '{requested_uri}' is not a concrete resource on '{MCP_SERVER}'"
                ))
            })?;
        Ok(McpResourceRef::new(MCP_SERVER, selected.uri()))
    }

    fn answer_question(root: Flow<Question>) -> Flow<Answer> {
        root.agent(knowledge_agent)
    }

    fn runtime_context() -> Result<Context, ExampleError> {
        let url = required_env("PRAVAH_MCP_URL")?;
        let mut server = McpServer::new(MCP_SERVER, url);
        if let Ok(token) = env::var("PRAVAH_MCP_BEARER_TOKEN") {
            server = server.bearer_token(token);
        }
        if let Ok(tenant) = env::var("PRAVAH_MCP_TENANT") {
            server = server.header("x-tenant", tenant);
        }
        let mut deps = Deps::default();
        deps.insert(Arc::new(SearchIndex(vec![
            "refund requests require an approval record".into(),
            "security incidents require immediate escalation".into(),
        ])));
        Ok(Context::new(FlowConf::default())
            .with_deps(deps)
            .with_mcp_server(server))
    }

    fn required_env(name: &str) -> Result<String, ExampleError> {
        env::var(name).map_err(|_| ExampleError::unexpected(format!("set {name} before running")))
    }

    /// Runs the configured graph until the agent produces its structured answer.
    pub async fn run() -> Result<(), ExampleError> {
        dotenvy::dotenv().ok();
        let ctx = runtime_context()?;
        let question = Question {
            text: env::args()
                .nth(1)
                .unwrap_or_else(|| "What does the selected policy say?".into()),
            model: env::var("PRAVAH_MODEL_URL").unwrap_or_else(|_| "openai:///gpt-5-mini".into()),
            resource_uri: required_env("PRAVAH_MCP_RESOURCE_URI")?,
            allow_search: env::var("PRAVAH_ALLOW_SEARCH").is_ok_and(|value| value == "1"),
        };
        let flow = graph::compile(answer_question)?;
        let mut runtime = flow.runtime(question)?;
        loop {
            match runtime.next(ctx.clone()).await? {
                Step::Continue => {}
                Step::Done(value) => {
                    println!("{:#?}", flow.decode_output(value)?);
                    return Ok(());
                }
                Step::Suspend(_) => {
                    return Err(ExampleError::unexpected(
                        "MCP resource example does not suspend",
                    ));
                }
            }
        }
    }
}

#[cfg(feature = "mcp")]
#[tokio::main]
async fn main() -> Result<(), support::ExampleError> {
    enabled::run().await
}

#[cfg(not(feature = "mcp"))]
fn main() {
    eprintln!("run with `cargo run --example graph_agent_mcp --features mcp`");
}
