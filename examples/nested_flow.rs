//! Compatibility-only legacy example where an outer flow delegates to an inner flow.
//!
//! Requires `GEMINI_API_KEY`.

mod support;

use pravah::legacy::{Agent, AgentConfig, Flow, FlowError, FlowRuntime, FlowStep, Node};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use support::ExampleError;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchQuery {
    query: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchResult {
    findings: String,
}

impl Agent for ResearchQuery {
    type Output = ResearchResult;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a research assistant. Answer the user's query with a concise \
             paragraph of factual findings.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

impl Flow for ResearchQuery {
    type Output = ResearchResult;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BlogRequest {
    topic: String,
    audience: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalArticle {
    title: String,
    body: String,
}

impl Agent for ResearchResult {
    type Output = FinalArticle;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a blog writer. Using the research findings provided, write a \
             short, engaging blog post with a title and two paragraphs.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

async fn derive_query(req: BlogRequest, _ctx: Context) -> Result<ResearchQuery, FlowError> {
    Ok(ResearchQuery {
        query: format!("{} (for {})", req.topic, req.audience),
    })
}

impl Flow for BlogRequest {
    type Output = FinalArticle;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.work(derive_query).flow().agent()
    }
}

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let input = BlogRequest {
        topic: "Why Rust's ownership model matters".to_string(),
        audience: "developers new to systems programming".to_string(),
    };

    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(article) => {
                println!("# {}\n\n{}", article.title, article.body);
                break;
            }
            FlowStep::Suspend(_) => {
                return Err(ExampleError::unexpected(
                    "nested flow unexpectedly suspended",
                ));
            }
        }
    }

    Ok(())
}
