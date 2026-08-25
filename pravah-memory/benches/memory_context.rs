use std::hint::black_box;
use std::time::Instant;

use pravah_memory::context::{ContextAssembler, ContextError, ContextOptions};
use pravah_memory::{EvidenceId, Memory, MemoryId, MemoryKind, SearchResult, TemporalMetadata};

#[cfg(debug_assertions)]
const ITERATIONS: usize = 100;
#[cfg(not(debug_assertions))]
const ITERATIONS: usize = 10_000;
const SAMPLES: usize = 20;

fn main() {
    if let Err(error) = run() {
        eprintln!("memory context benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), ContextError> {
    let results = fixture(50);
    let assembler = ContextAssembler::compact();
    let options = ContextOptions::default();
    let context = assembler.assemble(&results, options.clone())?;
    let allocations = allocation_counter::measure(|| {
        let assembled = assembler
            .assemble(black_box(&results), options.clone())
            .expect("benchmark fixture must assemble");
        black_box(assembled);
    });
    let mut samples = latency_samples(&assembler, &results, &options)?;
    samples.sort_by(f64::total_cmp);
    let p95 = samples[(samples.len() - 1) * 95 / 100];
    println!(
        "memory/context_50_to_8: {p95:.2} us/op p95; {} allocation(s), {} byte(s); {} selected",
        allocations.count_total,
        allocations.bytes_total,
        context.selected_memory_ids.len(),
    );
    Ok(())
}

fn latency_samples(
    assembler: &ContextAssembler,
    results: &[SearchResult],
    options: &ContextOptions,
) -> Result<Vec<f64>, ContextError> {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(assembler.assemble(black_box(results), options.clone())?);
        }
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64);
    }
    Ok(samples)
}

fn fixture(count: usize) -> Vec<SearchResult> {
    (0..count)
        .map(|position| SearchResult {
            memory: memory(position),
            evidence_key: format!("document:benchmark:{position}"),
            score: 1.0 / (position + 1) as f64,
            rerank_score: None,
            support_count: position as u32 % 3 + 1,
            conflicts: Vec::new(),
        })
        .collect()
}

fn memory(position: usize) -> Memory {
    Memory {
        id: MemoryId::new(),
        evidence_id: EvidenceId::new(),
        user_key: "benchmark-user".to_owned(),
        agent_key: "benchmark-agent".to_owned(),
        position: position as u32,
        text: format!(
            "Evidence-supported benchmark claim {position} with enough text to model context use."
        ),
        kind: MemoryKind::Fact,
        temporal: TemporalMetadata::default(),
        metadata: serde_json::json!({}),
        stale: false,
        current_for_retrieval: true,
    }
}
