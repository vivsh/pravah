use std::time::Instant;

#[path = "chat/cases.rs"]
mod cases;

#[path = "chat/tool_loop.rs"]
mod tool_loop;

use pravah::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmBackend, Message,
    ModelUrl, Provider, ProviderFactory,
};
use pravah::{
    Agent, AgentConfig, Chat, CompactionRequest, CompactionResult, Compactor, Context, GraphError,
};

struct Factory;
struct Model {
    model: ModelUrl,
    options: ClientOptions,
}

impl ProviderFactory for Factory {
    async fn llm(&self, model: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        Ok(Client::from_backend(Model {
            model: model.clone(),
            options,
        }))
    }
}

impl LlmBackend for Model {
    fn model_url(&self) -> &ModelUrl {
        &self.model
    }
    fn options(&self) -> &ClientOptions {
        &self.options
    }
    async fn execute(&self, _messages: &[Message]) -> Result<ClientResponse, ClientError> {
        Ok(ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::Output(serde_json::json!("answer")),
        ))
    }
}

struct BoundHistory;
impl Compactor for BoundHistory {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        input: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        Ok(CompactionResult {
            evict_indices: (0..input.committed().len()).collect(),
            summary: None,
        })
    }
}

fn agent(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new("test:///test", "Answer.", Message::user(input)).key("conversation"))
}

/// Measures bounded-history steady-state sends, excluding initialization and warmup.
fn run(runtime: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let iterations = iterations();
    let mut samples = Vec::new();
    for _ in 0..5 {
        let context = Context::default().with_providers(pravah::testing::providers(Factory)?);
        let construction = Instant::now();
        let mut chat = Chat::new(agent, context)?.with_compactor(BoundHistory);
        println!(
            "chat/construction: {} ns",
            construction.elapsed().as_nanos()
        );
        for _ in 0..3 {
            runtime.block_on(chat.send("question"))?;
        }
        let mut result = Ok(());
        let allocations = allocation_counter::measure(|| {
            result = runtime.block_on(chat.send("question")).map(|_| ());
        });
        result?;
        println!(
            "chat/turn_allocations: {} / {} bytes",
            allocations.count_total, allocations.bytes_total
        );
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(runtime.block_on(chat.send("question"))?);
        }
        samples.push(started.elapsed().as_nanos() / iterations);
    }
    samples.sort();
    println!("chat/steady_turn: {:?} ns/turn", samples);
    measure_large_state(runtime)
}

/// Compares steady-state execution and checkpoint costs with a megabyte of application state.
fn measure_large_state(runtime: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let state = vec!["private-state".repeat(1000); 128];
    let context = Context::default().with_providers(pravah::testing::providers(Factory)?);
    let construction = Instant::now();
    let mut chat = Chat::with_state(agent, state, context)?.with_compactor(BoundHistory);
    println!(
        "chat/large_construction: {} ns",
        construction.elapsed().as_nanos()
    );
    for _ in 0..3 {
        runtime.block_on(chat.send("question"))?;
    }
    let mut result = Ok(());
    let allocations = allocation_counter::measure(|| {
        result = runtime.block_on(chat.send("question")).map(|_| ());
    });
    result?;
    println!(
        "chat/large_turn_allocations: {} / {} bytes",
        allocations.count_total, allocations.bytes_total
    );
    let iterations = iterations();
    let mut samples = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(runtime.block_on(chat.send("question"))?);
        }
        samples.push(started.elapsed().as_nanos() / iterations);
    }
    samples.sort();
    println!("chat/large_steady_turn: {:?} ns/turn", samples);
    let mut snapshot = None;
    let allocations = allocation_counter::measure(|| {
        snapshot = Some(chat.snapshot());
    });
    println!(
        "chat/large_snapshot: {} allocations / {} bytes",
        allocations.count_total, allocations.bytes_total
    );
    let snapshot =
        snapshot.ok_or_else(|| GraphError::Invalid("benchmark snapshot missing".into()))??;
    let restore = Instant::now();
    let _chat =
        Chat::<String, String, Vec<String>>::from_snapshot(agent, snapshot, Context::default())?;
    println!("chat/large_restore: {} ns", restore.elapsed().as_nanos());
    Ok(())
}

/// Allows longer sampling runs without changing the default comparison workload.
fn iterations() -> u128 {
    std::env::var("PRAVAH_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(if cfg!(debug_assertions) { 10 } else { 2000 })
}

fn main() -> Result<(), GraphError> {
    if std::env::var_os("PRAVAH_BENCH_CONSTRUCTION").is_some() {
        return cases::construction();
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|error| GraphError::Invalid(error.to_string()))?;
    if std::env::var_os("PRAVAH_BENCH_CASE").is_none() {
        run(&runtime)?;
    }
    cases::run(&runtime)?;
    tool_loop::run(&runtime)
}
