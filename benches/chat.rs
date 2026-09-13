use std::time::Instant;

use async_trait::async_trait;
use pravah::clients::{
    Client, ClientError, ClientFactory, ClientOptions, ClientOutput, ClientResponse, Message,
    ModelUrl, Provider,
};
use pravah::{
    Agent, AgentConfig, Chat, Context, GraphError, HistoryPreparation, HistoryPreparer,
    HistoryReplacement,
};

struct Factory;
struct Model {
    model: ModelUrl,
    options: ClientOptions,
}

impl ClientFactory for Factory {
    fn create(&self, model: &str, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
        Ok(Box::new(Model {
            model: ModelUrl::parse(model)?,
            options,
        }))
    }
}

#[async_trait]
impl Client for Model {
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
impl HistoryPreparer for BoundHistory {
    type Error = std::convert::Infallible;
    async fn prepare(
        &self,
        input: HistoryPreparation<'_>,
        _ctx: Context,
    ) -> Result<HistoryReplacement, Self::Error> {
        Ok(HistoryReplacement {
            evict_indices: (0..input.committed().len()).collect(),
            summary: None,
        })
    }
}

fn agent(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new("openai:///test", "Answer.", Message::user(input)).keep_alive())
}

/// Measures bounded-history steady-state sends, excluding initialization and warmup.
fn run(runtime: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let iterations = if cfg!(debug_assertions) { 10 } else { 2000 };
    let mut samples = Vec::new();
    for _ in 0..5 {
        let context = Context::default().with_client_factory(Factory);
        let construction = Instant::now();
        let mut chat = runtime
            .block_on(Chat::new(agent, context))?
            .with_history_preparer(BoundHistory);
        println!(
            "chat/construction: {} ns",
            construction.elapsed().as_nanos()
        );
        for _ in 0..3 {
            runtime.block_on(chat.send("question".into()))?;
        }
        let mut result = Ok(());
        let allocations = allocation_counter::measure(|| {
            result = runtime.block_on(chat.send("question".into())).map(|_| ());
        });
        result?;
        println!(
            "chat/turn_allocations: {} / {} bytes",
            allocations.count_total, allocations.bytes_total
        );
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(runtime.block_on(chat.send("question".into()))?);
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
    let context = Context::default().with_client_factory(Factory);
    let construction = Instant::now();
    let mut chat = runtime
        .block_on(Chat::with_state(agent, state, context))?
        .with_history_preparer(BoundHistory);
    println!(
        "chat/large_construction: {} ns",
        construction.elapsed().as_nanos()
    );
    for _ in 0..3 {
        runtime.block_on(chat.send("question".into()))?;
    }
    let mut result = Ok(());
    let allocations = allocation_counter::measure(|| {
        result = runtime.block_on(chat.send("question".into())).map(|_| ());
    });
    result?;
    println!(
        "chat/large_turn_allocations: {} / {} bytes",
        allocations.count_total, allocations.bytes_total
    );
    let iterations = if cfg!(debug_assertions) { 10 } else { 2000 };
    let mut samples = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(runtime.block_on(chat.send("question".into()))?);
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

fn main() -> Result<(), GraphError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|error| GraphError::Invalid(error.to_string()))?;
    run(&runtime)
}
