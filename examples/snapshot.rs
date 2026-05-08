//! # Example 4 — Snapshot, Save and Restore
//!
//! Demonstrates persisting a flow mid-execution and resuming it from a
//! saved snapshot — useful for long-running pipelines, crash recovery, and
//! distributed handoffs.
//!
//! ```text
//! RawRecord ──work──► NormalisedRecord ──work──► EnrichedRecord ──work──► FinalRecord (terminal)
//!                            ▲
//!                    snapshot taken here
//!                    (serialised to disk)
//!                            │
//!                    runtime restored here
//!                    and execution continues
//! ```
//!
//! This example uses `work`-only nodes so it runs without an API key.
//!
//! ## Running
//!
//! ```shell
//! cargo run --example snapshot
//! ```

use std::fs;
use std::path::PathBuf;

use pravah::flows::{Flow, FlowError, FlowGraph, FlowRuntime, FlowSnapshot, RunOut};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Types ────────────────────────────────────────────────────────────────────

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

// ── Flow ──────────────────────────────────────────────────────────────────────

impl Flow for RawRecord {
    type Output = FinalRecord;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            // Step 1 — parse and normalise the raw string value.
            .work::<RawRecord, NormalisedRecord, _, _>(|rec, _ctx| async move {
                let value: f64 = rec
                    .raw_value
                    .trim()
                    .parse()
                    .map_err(|_| FlowError::AgentError("invalid numeric string".into()))?;
                println!("[step 1] normalised {} → {value}", rec.raw_value);
                Ok(NormalisedRecord { id: rec.id, value })
            })
            // Step 2 — enrich with a derived category.
            .work::<NormalisedRecord, EnrichedRecord, _, _>(|rec, _ctx| async move {
                let category = if rec.value >= 100.0 {
                    "high"
                } else if rec.value >= 50.0 {
                    "medium"
                } else {
                    "low"
                };
                println!("[step 2] enriched id={} → category={category}", rec.id);
                Ok(EnrichedRecord { id: rec.id, value: rec.value, category: category.to_string() })
            })
            // Step 3 — format the final report.
            .work::<EnrichedRecord, FinalRecord, _, _>(|rec, _ctx| async move {
                let report = format!(
                    "Record #{}: value={:.2}, category={}",
                    rec.id, rec.value, rec.category
                );
                println!("[step 3] formatted report: {report}");
                Ok(FinalRecord { id: rec.id, report })
            })
            .build(RawRecord::node_id())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new(FlowConf::default());

    let input = RawRecord { id: 42, raw_value: "73.5".to_string() };

    // ── Phase A: run one step, snapshot, then drop the runtime ───────────────
    println!("\n=== Phase A: run step 1 then snapshot ===");
    let mut runtime = FlowRuntime::new(input)?;

    // Step 1: RawRecord → NormalisedRecord
    match runtime.next(ctx.clone()).await? {
        RunOut::Continue => println!("[phase A] step 1 complete"),
        other => panic!("unexpected: {other:?}"),
    }

    // Capture and persist the snapshot after step 1.
    save_snapshot(&runtime.snapshot())?;

    // Simulate a process restart — drop the original runtime entirely.
    drop(runtime);

    // ── Phase B: restore from snapshot and run to completion ─────────────────
    println!("\n=== Phase B: restore and continue ===");
    let snap = load_snapshot()?;
    let mut restored = FlowRuntime::<RawRecord>::from_snapshot(snap)?;

    loop {
        match restored.next(ctx.clone()).await? {
            RunOut::Continue => {}
            RunOut::Done(record) => {
                println!("\n=== Done ===\n{}", record.report);

                // Clean up the temporary snapshot file.
                let _ = fs::remove_file(snapshot_path());
                break;
            }
            RunOut::Suspend { value, tool_id } => {
                eprintln!("Unexpected suspension at '{tool_id}': {value}");
                break;
            }
        }
    }

    Ok(())
}
