//! Complete local turns with and without streaming, excluding initialization and warmup.

use super::*;

/// Compares identical bounded-history turns with a two-delta streaming backend.
pub(super) fn run(rt: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    for streaming in [false, true] {
        let context = Context::default().with_providers(pravah::testing::providers(Factory)?);
        let mut chat = Chat::new(agent, context)?.with_compactor(BoundHistory);
        for _ in 0..5 {
            rt.block_on(turn(&mut chat, streaming))?;
        }
        let mut result = Ok(());
        let allocations = allocation_counter::measure(|| {
            result = rt.block_on(turn(&mut chat, streaming));
        });
        result?;
        println!(
            "streaming/{streaming}: {} allocations / {} bytes",
            allocations.count_total, allocations.bytes_total
        );
        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            for _ in 0..iterations() {
                rt.block_on(turn(&mut chat, streaming))?;
            }
            samples.push(started.elapsed().as_nanos() / iterations());
        }
        println!("streaming/{streaming}: {samples:?} ns/turn");
    }
    Ok(())
}

/// Uses the same final output and compaction with optional operation-local event delivery.
async fn turn(chat: &mut Chat<String, String>, streaming: bool) -> Result<(), GraphError> {
    let reply = if streaming {
        chat.send_stream("question", |_, event| {
            std::hint::black_box(event);
            std::future::ready(())
        })
        .await?
    } else {
        chat.send("question").await?
    };
    std::hint::black_box(reply);
    Ok(())
}
