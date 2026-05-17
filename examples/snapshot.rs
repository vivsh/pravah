//! Snapshot example showing save and restore mid-execution with work-only nodes.

use std::fs;
use std::path::PathBuf;

use pravah::flows::{Flow, FlowError, FlowGraph, FlowRuntime, FlowSnapshot, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RawRecord {
    id: u64,
    raw_value: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct NormalisedRecord {
    id: u64,
    value: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EnrichedRecord {
    id: u64,
    value: f64,
    category: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalRecord {
    id: u64,
    report: String,
}


async fn normalise(rec: RawRecord, _ctx: Context) -> Result<NormalisedRecord, FlowError> {
    let value: f64 = rec
        .raw_value
        .trim()
        .parse()
        .map_err(|_| FlowError::Internal { handler: "snapshot", detail: "invalid numeric string".into() })?;
    println!("[step 1] normalised {} → {value}", rec.raw_value);
    Ok(NormalisedRecord { id: rec.id, value })
}

async fn enrich(rec: NormalisedRecord, _ctx: Context) -> Result<EnrichedRecord, FlowError> {
    let category = if rec.value >= 100.0 { "high" } else if rec.value >= 50.0 { "medium" } else { "low" };
    println!("[step 2] enriched id={} → category={category}", rec.id);
    Ok(EnrichedRecord { id: rec.id, value: rec.value, category: category.to_string() })
}

async fn format_record(rec: EnrichedRecord, _ctx: Context) -> Result<FinalRecord, FlowError> {
    let report = format!("Record #{}: value={:.2}, category={}", rec.id, rec.value, rec.category);
    println!("[step 3] formatted report: {report}");
    Ok(FinalRecord { id: rec.id, report })
}


impl Flow for RawRecord {
    type Output = FinalRecord;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(normalise)
            .work(enrich)
            .work(format_record)
            .build()
    }
}


fn snapshot_path() -> PathBuf {
    std::env::temp_dir().join("pravah_snapshot_example.json")
}

fn save_snapshot(snap: &FlowSnapshot) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(snap)?;
    fs::write(snapshot_path(), json)?;
    println!("[snapshot] saved to {}", snapshot_path().display());
    Ok(())
}

fn load_snapshot() -> Result<FlowSnapshot, Box<dyn std::error::Error>> {
    let json = fs::read_to_string(snapshot_path())?;
    let snap: FlowSnapshot = serde_json::from_str(&json)?;
    println!("[snapshot] loaded from {}", snapshot_path().display());
    Ok(snap)
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let input = RawRecord { id: 42, raw_value: "73.5".to_string() };

    println!("\n=== Phase A: run step 1 then snapshot ===");
    let mut runtime = FlowRuntime::new(input)?;

    match runtime.next(ctx.clone()).await? {
        FlowStep::Continue => println!("[phase A] step 1 complete"),
        other => panic!("unexpected: {other:?}"),
    }

    save_snapshot(&runtime.snapshot())?;

    drop(runtime);

    println!("\n=== Phase B: restore and continue ===");
    let snap = load_snapshot()?;
    let mut restored = FlowRuntime::<RawRecord>::from_snapshot(snap)?;

    loop {
        match restored.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(record) => {
                println!("\n=== Done ===\n{}", record.report);

                let _ = fs::remove_file(snapshot_path());
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
