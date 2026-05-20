//! Nested flow example where an outer flow delegates research to an inner flow.

use pravah::flows::{Agent, AgentConfig, Flow, FlowBuilder, FlowError, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};


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

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a research assistant. Answer the user's query with a concise \
             paragraph of factual findings.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

impl Flow for ResearchQuery {
    type Output = ResearchResult;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<ResearchQuery>()
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

    fn build() -> AgentConfig {
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

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .work(derive_query)
            .flow::<ResearchQuery>()
            .agent::<ResearchResult>()
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
                eprintln!("Unexpected suspension");
                break;
            }
        }
    }

    Ok(())
}
