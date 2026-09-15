use super::*;
use pravah::{AgentDecision, AgentLoop, ChatRequest};
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

fn controlled(root: Agent<String>) -> Agent<String> {
    root.control(control).configure(configure)
}

async fn control(_: AgentLoop<String>, _: Context) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::continue_())
}

fn question() -> String {
    "question".into()
}

fn request() -> ChatRequest {
    ChatRequest::from("question").memory("Use concise answers.")
}

/// Covers controller, keyed, and callback-free Chat paths with the same local client.
pub(super) fn run(rt: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let context = || Context::default().with_client_factory(Factory);
    let mut chat = rt
        .block_on(Chat::new(controlled, context()))?
        .with_compactor(BoundHistory);
    measure(rt, "chat/controlled", &mut chat, question, false)?;
    let mut chat = rt
        .block_on(Chat::new(agent, context()))?
        .with_compactor(BoundHistory);
    measure(rt, "chat/keyed", &mut chat, question, true)?;
    let mut chat = rt.block_on(
        Chat::builder::<String>()
            .model("openai:///test")
            .instructions("Answer.")
            .compactor(BoundHistory)
            .build(context()),
    )?;
    measure(rt, "chat/builder", &mut chat, request, false)?;
    measure(rt, "chat/builder_keyed", &mut chat, request, true)
}

/// Measures initialized sends, including necessary input allocation and conversion.
fn measure<I>(
    rt: &tokio::runtime::Runtime,
    name: &str,
    chat: &mut Chat<I, String>,
    input: fn() -> I,
    keyed: bool,
) -> Result<(), GraphError>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    let mut send = || {
        if keyed {
            rt.block_on(chat.send_with_key(input(), "message-key"))
        } else {
            rt.block_on(chat.send(input()))
        }
    };
    for _ in 0..3 {
        send()?;
    }
    let mut result = Ok(());
    let allocations = allocation_counter::measure(|| {
        result = send().map(|_| ());
    });
    result?;
    println!(
        "{name}_allocations: {} / {} bytes",
        allocations.count_total, allocations.bytes_total
    );
    let mut samples = Vec::new();
    let count = iterations();
    for _ in 0..5 {
        let started = Instant::now();
        for _ in 0..count {
            std::hint::black_box(send()?);
        }
        samples.push(started.elapsed().as_nanos() / count);
    }
    samples.sort();
    println!("{name}: {samples:?} ns/turn");
    Ok(())
}
