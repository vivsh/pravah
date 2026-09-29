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

fn request() -> ChatRequest<String> {
    ChatRequest::from("question").memory("Use concise answers.")
}

/// Measures warmed construction separately from turn timing and process initialization.
pub(super) fn construction() -> Result<(), GraphError> {
    for (name, controlled) in [("builder", false), ("builder_controlled", true)] {
        let build = || {
            let builder = Chat::builder::<String, String>()
                .model("test:///test")
                .instructions("Answer.");
            let builder = if controlled {
                builder.control(control)
            } else {
                builder
            };
            builder.build(Context::default())
        };
        for _ in 0..5 {
            std::hint::black_box(build()?);
        }
        let mut result = Ok(());
        let allocations = allocation_counter::measure(|| result = build().map(|_| ()));
        result?;
        println!(
            "construction/{name}_allocations: {} / {} bytes",
            allocations.count_total, allocations.bytes_total
        );
        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            for _ in 0..100 {
                std::hint::black_box(build()?);
            }
            samples.push(started.elapsed().as_nanos() / 100);
        }
        println!("construction/{name}: {samples:?} ns/build");
    }
    Ok(())
}

/// Covers controller, keyed, and callback-free Chat paths with the same local client.
pub(super) fn run(rt: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let context = || -> Result<Context, GraphError> {
        Ok(Context::default().with_providers(pravah::testing::providers(Factory)?))
    };
    let mut chat = Chat::new(controlled, context()?)?.with_compactor(BoundHistory);
    measure(rt, "chat/controlled", &mut chat, question, false)?;
    let mut chat = Chat::new(agent, context()?)?.with_compactor(BoundHistory);
    measure(rt, "chat/keyed", &mut chat, question, true)?;
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .instructions("Answer.")
        .compactor(BoundHistory)
        .build(context()?)?;
    measure(rt, "chat/builder", &mut chat, request, false)?;
    measure(rt, "chat/builder_keyed", &mut chat, request, true)?;
    builder_controlled(rt)?;
    large_inputs(rt)
}

/// Separates callback-free controlled construction from steady-state controller dispatch.
fn builder_controlled(rt: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let started = Instant::now();
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .instructions("Answer.")
        .control(control)
        .compactor(BoundHistory)
        .build(Context::default().with_providers(pravah::testing::providers(Factory)?))?;
    println!(
        "chat/builder_controlled_construction: {} ns",
        started.elapsed().as_nanos()
    );
    measure(rt, "chat/builder_controlled", &mut chat, request, false)
}

fn large_question() -> String {
    "question".repeat(8192)
}

fn large_request() -> ChatRequest<String> {
    ChatRequest::from(large_question()).memory("Use concise answers.")
}

/// Records 64 KiB input costs independently of the retained-state benchmark.
fn large_inputs(rt: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let context = || -> Result<Context, GraphError> {
        Ok(Context::default().with_providers(pravah::testing::providers(Factory)?))
    };
    let mut chat = Chat::new(agent, context()?)?.with_compactor(BoundHistory);
    measure(rt, "chat/large_input", &mut chat, large_question, false)?;
    measure(
        rt,
        "chat/large_input_keyed",
        &mut chat,
        large_question,
        true,
    )?;
    let mut chat = Chat::new(controlled, context()?)?.with_compactor(BoundHistory);
    measure(
        rt,
        "chat/large_input_controlled",
        &mut chat,
        large_question,
        false,
    )?;
    let started = Instant::now();
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .instructions("Answer.")
        .compactor(BoundHistory)
        .build(context()?)?;
    println!(
        "chat/builder_construction: {} ns",
        started.elapsed().as_nanos()
    );
    measure(
        rt,
        "chat/large_input_builder",
        &mut chat,
        large_request,
        false,
    )?;
    measure(
        rt,
        "chat/large_input_builder_keyed",
        &mut chat,
        large_request,
        true,
    )
}

/// Measures initialized sends, including necessary input allocation and conversion.
fn measure<I, R>(
    rt: &tokio::runtime::Runtime,
    name: &str,
    chat: &mut Chat<I, String>,
    input: fn() -> R,
    keyed: bool,
) -> Result<(), GraphError>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    R: Into<ChatRequest<I>>,
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
