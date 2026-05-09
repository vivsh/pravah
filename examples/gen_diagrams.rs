//! Diagram generator — Article Production Pipeline.
//!
//! A realistic multi-stage content pipeline that exercises every node type:
//!
//! ```text
//! ArticleRequest
//!   ├─fork─┬─ ResearchTask  ──agent──► ResearchNotes  ─┐
//!          └─ AudienceTask  ──agent──► AudienceProfile ─┤join
//!                                                        ▼
//!                                                  ContentBrief
//!                                                  │work
//!                                            OutlineRequest
//!                                     (nested OutlineFlow) │work
//!                                                          ▼
//!                                                       Outline
//!                                                 ┌──either──┐
//!                                           QuickDraft     LongDraft
//!                                           (agent)   (nested ReviewFlow)
//!                                               │              │work
//!                                               │         ReviewedDraft
//!                                               │           (agent)
//!                                               └──────────────┘
//!                                                      ▼
//!                                               FinalArticle ◉
//! ```
//!
//! Inner flows:
//! - **OutlineFlow**: `OutlineRequest` → agent → `Outline`
//! - **ReviewFlow**: `LongDraft` → agent → `ReviewedDraft`
//!
//! ```shell
//! cargo run --features diagram-text --example gen_diagrams
//! ```

use either::Either;
use pravah::flows::{Agent, Flow, FlowError, FlowGraph, FlowRuntime, RunOut};
use pravah::flows::FlowGraphDiagram;
use pravah::tools::ToolBox;
use pravah::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Domain types ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ArticleRequest {
    topic: String,
    format: String, // "quick" | "detailed"
}

// Fork outputs
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchTask { topic: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AudienceTask { topic: String, format: String }

// Agent outputs from parallel branches
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchNotes { notes: String, sources: Vec<String> }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AudienceProfile { tone: String, reading_level: String }

// Join output
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ContentBrief { notes: String, tone: String, format: String }

// Nested OutlineFlow entry
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct OutlineRequest { brief: String }

// Outer either-node input + OutlineFlow terminal
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Outline { sections: Vec<String>, format: String }

// Either branches
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct QuickDraft { content: String }

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LongDraft { sections: Vec<String> }

// Nested ReviewFlow output
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReviewedDraft { content: String, notes: String }

// Terminal (both paths converge here)
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalArticle { title: String, body: String }

// ── Inner flow 1: OutlineFlow (OutlineRequest → agent → Outline) ───────────────

impl Agent for OutlineRequest {
    type Output = Outline;
    fn preamble() -> String { "Generate a structured outline with sections.".into() }
    fn model_url() -> String { "gemini://gemini-2.5-flash-lite".into() }
    fn tool_box() -> ToolBox { ToolBox::builder().build() }
}

impl Flow for OutlineRequest {
    type Output = Outline;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<OutlineRequest>().build()
    }
}

// ── Inner flow 2: ReviewFlow (LongDraft → agent → ReviewedDraft) ───────────────

impl Agent for LongDraft {
    type Output = ReviewedDraft;
    fn preamble() -> String { "Review the draft for quality, accuracy, and structure.".into() }
    fn model_url() -> String { "gemini://gemini-2.5-flash-lite".into() }
    fn tool_box() -> ToolBox { ToolBox::builder().build() }
}

impl Flow for LongDraft {
    type Output = ReviewedDraft;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<LongDraft>().build()
    }
}

// ── Outer flow agents ──────────────────────────────────────────────────────────

impl Agent for ResearchTask {
    type Output = ResearchNotes;
    fn preamble() -> String { "Research the topic and gather key facts and sources.".into() }
    fn model_url() -> String { "gemini://gemini-2.5-flash-lite".into() }
    fn tool_box() -> ToolBox { ToolBox::builder().build() }
}

impl Agent for AudienceTask {
    type Output = AudienceProfile;
    fn preamble() -> String { "Analyse the target audience and determine tone and reading level.".into() }
    fn model_url() -> String { "gemini://gemini-2.5-flash-lite".into() }
    fn tool_box() -> ToolBox { ToolBox::builder().build() }
}

impl Agent for QuickDraft {
    type Output = FinalArticle;
    fn preamble() -> String { "Write a concise, punchy article from the quick draft.".into() }
    fn model_url() -> String { "gemini://gemini-2.5-flash-lite".into() }
    fn tool_box() -> ToolBox { ToolBox::builder().build() }
}

impl Agent for ReviewedDraft {
    type Output = FinalArticle;
    fn preamble() -> String { "Polish the reviewed draft into a publication-ready article.".into() }
    fn model_url() -> String { "gemini://gemini-2.5-flash-lite".into() }
    fn tool_box() -> ToolBox { ToolBox::builder().build() }
}

// ── Outer flow node handlers ───────────────────────────────────────────────────

fn split_request(req: ArticleRequest, _ctx: Context) -> Result<(ResearchTask, AudienceTask), FlowError> {
    Ok((
        ResearchTask { topic: req.topic.clone() },
        AudienceTask { topic: req.topic, format: req.format },
    ))
}

fn merge_findings(
    research: ResearchNotes,
    audience: AudienceProfile,
    _ctx: Context,
) -> Result<ContentBrief, FlowError> {
    Ok(ContentBrief { notes: research.notes, tone: audience.tone, format: audience.reading_level })
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
            RunOut::Continue => {}
            RunOut::Done(result) => return Ok(result),
            RunOut::Suspend { value, tool_id } => {
                return Err(FlowError::AgentError(format!("outline flow suspended at {tool_id}: {value}")));
            }
        }
    }
}

fn route_draft(outline: Outline, _ctx: Context) -> Result<Either<QuickDraft, LongDraft>, FlowError> {
    if outline.format == "quick" || outline.sections.len() <= 3 {
        Ok(Either::Left(QuickDraft { content: outline.sections.join("\n") }))
    } else {
        Ok(Either::Right(LongDraft { sections: outline.sections }))
    }
}

async fn run_review_flow(draft: LongDraft, ctx: Context) -> Result<ReviewedDraft, FlowError> {
    let mut rt = FlowRuntime::new(draft)?;
    loop {
        match rt.next(ctx.clone()).await? {
            RunOut::Continue => {}
            RunOut::Done(result) => return Ok(result),
            RunOut::Suspend { value, tool_id } => {
                return Err(FlowError::AgentError(format!("review flow suspended at {tool_id}: {value}")));
            }
        }
    }
}

// ── Outer flow ─────────────────────────────────────────────────────────────────

impl Flow for ArticleRequest {
    type Output = FinalArticle;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork(split_request)        // ArticleRequest → (ResearchTask, AudienceTask)
            .agent::<ResearchTask>()    // ResearchTask   → ResearchNotes
            .agent::<AudienceTask>()    // AudienceTask   → AudienceProfile
            .join(merge_findings)       // (ResearchNotes, AudienceProfile) → ContentBrief
            .work(prepare_outline)      // ContentBrief   → OutlineRequest
            .work(run_outline_flow)     // OutlineRequest → Outline        (nested OutlineFlow)
            .either(route_draft)        // Outline        → QuickDraft | LongDraft
            .agent::<QuickDraft>()      // QuickDraft     → FinalArticle
            .work(run_review_flow)      // LongDraft      → ReviewedDraft  (nested ReviewFlow)
            .agent::<ReviewedDraft>()   // ReviewedDraft  → FinalArticle
            .build()
    }
}

// ── Main ───────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let diagram = FlowGraphDiagram::for_flow::<ArticleRequest>()?;

    println!("=== TREE ===");
    println!("{}", diagram.render_tree());
    println!("=== ASCII ===");
    println!("{}", diagram.render_ascii()?);
    println!("=== TEXT ===");
    println!("{}", diagram.render_text()?);
    println!("=== MERMAID ===");
    println!("{}", diagram.mermaid());
    println!("=== DOT ===");
    println!("{}", diagram.dot());

    Ok(())
}

