//! Deterministic graph-agent turn and tool budgets without a custom controller.
//!
//! Run with `cargo run --example graph_agent_budgets --features testing`.

#[cfg(feature = "testing")]
mod support;

#[cfg(feature = "testing")]
mod example {
    use pravah::Context;
    use pravah::clients::Message;
    use pravah::graph::{Agent, AgentConfig, Flow, Step, Toolset, compile};
    use pravah::testing::{ScriptedFactory, mock_tool_call};
    use pravah::tools::{ToolError, ToolOutput};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use super::support::ExampleError;

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct ResearchRequest {
        question: String,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct ResearchAnswer {
        answer: String,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct SearchRequest {
        query: String,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct SearchResult {
        evidence: String,
    }

    impl ToolOutput for SearchResult {}

    fn research_tools(tools: Toolset) -> Toolset {
        tools.tool_handler(search)
    }

    async fn search(input: SearchRequest, _ctx: Context) -> Result<SearchResult, ToolError> {
        Ok(SearchResult {
            evidence: format!("Evidence for {}", input.query),
        })
    }

    fn researcher(root: Agent<ResearchRequest>) -> Agent<ResearchAnswer> {
        root.tools(research_tools).configure(configure_researcher)
    }

    /// Configures one invocation with compact declarative budgets.
    async fn configure_researcher(
        request: ResearchRequest,
        _ctx: Context,
    ) -> Result<AgentConfig, pravah::graph::GraphError> {
        Ok(AgentConfig::new(
            "openai:///scripted",
            "Use available evidence, then return the structured answer.",
            Message::user(request.question),
        )
        .turn_budget(1)
        .tool_budget::<SearchRequest>(1))
    }

    fn research(root: Flow<ResearchRequest>) -> Flow<ResearchAnswer> {
        root.agent(researcher)
    }

    /// Runs an exact tool budget followed by Pravah's forced conclusion turn.
    pub(super) async fn run() -> Result<(), ExampleError> {
        let factory = ScriptedFactory::new()
            .then_tool_calls(vec![
                mock_tool_call("search-1", "search_request", json!({"query": "Pravah"})),
                mock_tool_call("search-2", "search_request", json!({"query": "budgets"})),
            ])
            .then_output(json!({"answer": "One search ran; the budget then forced conclusion."}));
        let flow = compile(research)?;
        let mut runtime = flow.runtime(ResearchRequest {
            question: "Explain the result succinctly.".into(),
        })?;
        let ctx = Context::default().with_client_factory(factory);
        drive(&flow, &mut runtime, ctx).await
    }

    /// Drives the stepwise runtime until the structured agent output is ready.
    async fn drive(
        flow: &pravah::graph::CompiledFlow<ResearchRequest, ResearchAnswer>,
        runtime: &mut pravah::graph::Runtime,
        ctx: Context,
    ) -> Result<(), ExampleError> {
        loop {
            match runtime.next(ctx.clone()).await? {
                Step::Continue => {}
                Step::Done(value) => {
                    println!("{:#?}", flow.decode_output(value)?);
                    return Ok(());
                }
                Step::Suspend(_) => {
                    return Err(ExampleError::unexpected(
                        "budgeted agent suspended unexpectedly",
                    ));
                }
            }
        }
    }
}

#[cfg(feature = "testing")]
#[tokio::main]
async fn main() -> Result<(), support::ExampleError> {
    example::run().await
}

#[cfg(not(feature = "testing"))]
fn main() {
    eprintln!("enable the 'testing' feature to run this deterministic example");
}
