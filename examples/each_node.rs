//! Each-node example: classify the sentiment of every review in a list, then
//! summarise the counts.
//!
//! Sub-flow: `ReviewItem` → agent → `ReviewSentiment`
//! Outer flow: `ReviewBatch` → map → `Vec<ReviewItem>` → each::<ReviewItem>() → `Vec<ReviewSentiment>` → work → `SentimentReport`

use pravah::flows::{Agent, AgentConfig, Flow, FlowBuilder, FlowError, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReviewItem {
    text: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReviewSentiment {
    /// One of: positive, negative, neutral.
    label: String,
    reason: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ReviewBatch {
    items: Vec<ReviewItem>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SentimentReport {
    positive: usize,
    negative: usize,
    neutral: usize,
    summary: String,
}


impl Agent for ReviewItem {
    type Output = ReviewSentiment;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a sentiment classifier. Classify the product review as \
             positive, negative, or neutral. Return a label field containing \
             exactly one of those three words, and a one-sentence reason.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

/// Sub-flow: one review → one sentiment classification.
impl Flow for ReviewItem {
    type Output = ReviewSentiment;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder.agent::<ReviewItem>()
    }
}


async fn tally(results: Vec<ReviewSentiment>, _ctx: Context) -> Result<SentimentReport, FlowError> {
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut neutral = 0usize;

    for r in &results {
        match r.label.to_lowercase().as_str().trim() {
            "positive" => positive += 1,
            "negative" => negative += 1,
            _ => neutral += 1,
        }
    }

    let summary = format!(
        "{positive} positive, {negative} negative, {neutral} neutral \
         across {} reviews",
        results.len()
    );

    Ok(SentimentReport { positive, negative, neutral, summary })
}

fn unwrap_batch(batch: ReviewBatch) -> Vec<ReviewItem> {
    batch.items
}

/// Outer flow: unwrap the batch, fan out over every review, then aggregate.
impl Flow for ReviewBatch {
    type Output = SentimentReport;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .map(unwrap_batch)
            .each::<ReviewItem>()
            .work(tally)
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let batch = ReviewBatch {
        items: vec![
            ReviewItem { text: "Absolutely love this product! Best purchase I've made.".into() },
            ReviewItem { text: "Terrible quality, broke after one use. Very disappointed.".into() },
            ReviewItem { text: "It's okay, nothing special but does the job.".into() },
            ReviewItem { text: "Exceeded my expectations in every way, highly recommended!".into() },
            ReviewItem { text: "Packaging was damaged and customer support was unhelpful.".into() },
        ],
    };

    let mut runtime = FlowRuntime::new(batch)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(report) => {
                println!("{}", report.summary);
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
