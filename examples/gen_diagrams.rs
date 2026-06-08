//! Diagram generation example that exercises the main flow node types.

use either::Either;
use pravah::flows::{Agent, AgentConfig, Flow, FlowBuilder, FlowError, FlowRuntime, FlowStep, Node};
use pravah::flows::FlowGraphDiagram;
use pravah::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ArticleRequest {
    topic: String,
    format: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchTask { topic: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AudienceTask { topic: String, format: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchNotes { notes: String, sources: Vec<String> }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AudienceProfile { tone: String, reading_level: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ContentBrief { notes: String, tone: String, format: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct OutlineRequest { brief: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Outline { sections: Vec<String>, format: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct QuickDraft { content: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LongDraft { sections: Vec<String> }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReviewedDraft { content: String, notes: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalArticle { title: String, body: String }


impl Agent for OutlineRequest {
    type Output = Outline;
    fn configure() -> AgentConfig { AgentConfig::new("Generate a structured outline with sections.", "gemini:///gemini-2.5-flash-lite") }
}

impl Flow for OutlineRequest {
    type Output = Outline;
    fn define(root: Node<Self>) -> FlowBuilder {
        root.agent().finalize()
    }
}


impl Agent for LongDraft {
    type Output = ReviewedDraft;
    fn configure() -> AgentConfig { AgentConfig::new("Review the draft for quality, accuracy, and structure.", "gemini:///gemini-2.5-flash-lite") }
}

impl Flow for LongDraft {
    type Output = ReviewedDraft;
    fn define(root: Node<Self>) -> FlowBuilder {
        root.agent().finalize()
    }
}


impl Agent for ResearchTask {
    type Output = ResearchNotes;
    fn configure() -> AgentConfig { AgentConfig::new("Research the topic and gather key facts and sources.", "gemini:///gemini-2.5-flash-lite") }
}

impl Agent for AudienceTask {
    type Output = AudienceProfile;
    fn configure() -> AgentConfig { AgentConfig::new("Analyse the target audience and determine tone and reading level.", "gemini:///gemini-2.5-flash-lite") }
}

impl Agent for QuickDraft {
    type Output = FinalArticle;
    fn configure() -> AgentConfig { AgentConfig::new("Write a concise, punchy article from the quick draft.", "gemini:///gemini-2.5-flash-lite") }
}

impl Agent for ReviewedDraft {
    type Output = FinalArticle;
    fn configure() -> AgentConfig { AgentConfig::new("Polish the reviewed draft into a publication-ready article.", "gemini:///gemini-2.5-flash-lite") }
}


fn split_request(req: ArticleRequest) -> (ResearchTask, AudienceTask) {
    (
        ResearchTask { topic: req.topic.clone() },
        AudienceTask { topic: req.topic, format: req.format },
    )
}

fn merge_findings(
    research: ResearchNotes,
    audience: AudienceProfile,
) -> ContentBrief {
    ContentBrief { notes: research.notes, tone: audience.tone, format: audience.reading_level }
}

async fn prepare_outline(brief: ContentBrief, _ctx: Context) -> Result<OutlineRequest, FlowError> {
    Ok(OutlineRequest {
        brief: format!("notes={} tone={} format={}", brief.notes, brief.tone, brief.format),
    })
}

async fn run_outline_flow(req: OutlineRequest, ctx: Context) -> Result<Outline, FlowError> {
    let mut rt = FlowRuntime::new(req)?;
    loop {
        match rt.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(result) => return Ok(result),
            FlowStep::Suspend(_) => {
                return Err(FlowError::Internal { handler: "gen_diagrams", detail: "outline flow suspended unexpectedly".into() });
            }
        }
    }
}

fn route_draft(outline: Outline) -> Either<QuickDraft, LongDraft> {
    if outline.format == "quick" || outline.sections.len() <= 3 {
        Either::Left(QuickDraft { content: outline.sections.join("\n") })
    } else {
        Either::Right(LongDraft { sections: outline.sections })
    }
}

async fn run_review_flow(draft: LongDraft, ctx: Context) -> Result<ReviewedDraft, FlowError> {
    let mut rt = FlowRuntime::new(draft)?;
    loop {
        match rt.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(result) => return Ok(result),
            FlowStep::Suspend(_) => {
                return Err(FlowError::Internal { handler: "gen_diagrams", detail: "review flow suspended unexpectedly".into() });
            }
        }
    }
}


impl Flow for ArticleRequest {
    type Output = FinalArticle;

    fn define(root: Node<Self>) -> FlowBuilder {
        let (research, audience) = root.split(split_request);

        research.agent()
            .merge(audience.agent(), |(research, audience)| merge_findings(research, audience))
            .work(prepare_outline)
            .work(run_outline_flow)
            .either(route_draft)
            .branch(
                |quick| quick.agent(),
                |long| long.work(run_review_flow).agent(),
            )
            .finalize()
    }
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let diagram = FlowGraphDiagram::from_flow::<ArticleRequest>()?;

    println!("=== TREE ===");
    println!("{}", diagram.render_tree());
    println!("=== MERMAID ===");
    println!("{}", diagram.mermaid());
    println!("=== DOT ===");
    println!("{}", diagram.dot());

    Ok(())
}

