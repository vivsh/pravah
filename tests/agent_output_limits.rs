use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use pravah::clients::{Client, ClientError, ClientFactory, ClientOptions, Message};
use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::tools::ToolError;
use pravah::{
    Agent, AgentConfig, CompactionRequest, CompactionResult, Compactor, Context, Flow, GraphError,
    Runtime, Snapshot, Step, Toolset, compile,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[path = "agent_output_limits/failures.rs"]
mod failures;
#[path = "agent_output_limits/restore.rs"]
mod restore;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Encode(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    Decode(#[from] ciborium::de::Error<std::io::Error>),
    #[error("{0}")]
    Missing(&'static str),
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Request {
    cap: Option<u32>,
    duplicate: bool,
    turns: Option<u32>,
}

impl Request {
    fn capped() -> Self {
        Self {
            cap: Some(2048),
            duplicate: false,
            turns: None,
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Lookup {
    query: String,
}

async fn lookup(input: Lookup, _ctx: Context) -> Result<String, ToolError> {
    Ok(input.query)
}

fn tools(root: Toolset) -> Toolset {
    root.tool(lookup)
}

fn agent(root: Agent<Request>) -> Agent<String> {
    root.tools(tools).configure(configure)
}

fn workflow(root: Flow<Request>) -> Flow<String> {
    root.agent(agent)
}

/// Allows tests to select invalid declarations and independent turn budgets.
async fn configure(input: Request, _ctx: Context) -> Result<AgentConfig, GraphError> {
    let mut config = AgentConfig::new(
        "openai:///test",
        "Answer briefly.",
        Message::user("question"),
    );
    if let Some(cap) = input.cap {
        config = config.max_output_tokens(cap);
        if input.duplicate {
            config = config.max_output_tokens(cap);
        }
    }
    if let Some(turns) = input.turns {
        config = config.turn_budget(turns);
    }
    Ok(config)
}

/// Checks the options at the actual factory boundary on every dispatch.
struct CheckFactory {
    script: ScriptedFactory,
    cap: Option<u32>,
}

impl ClientFactory for CheckFactory {
    fn create(&self, model: &str, options: ClientOptions) -> Result<Box<dyn Client>, ClientError> {
        assert_eq!(options.max_output_tokens, self.cap);
        assert_eq!(options.turn_budget, None);
        self.script.create(model, options)
    }
}

fn context(script: ScriptedFactory, cap: Option<u32>) -> Context {
    Context::default().with_client_factory(CheckFactory { script, cap })
}

#[derive(Clone)]
struct ObserveCap {
    calls: Arc<AtomicUsize>,
    cap: Option<u32>,
}

impl Compactor for ObserveCap {
    type Error = std::convert::Infallible;

    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        assert_eq!(request.options().max_output_tokens, self.cap);
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CompactionResult::default())
    }
}

/// Bounds deterministic test execution without hiding unexpected suspension.
async fn finish(runtime: &mut Runtime) -> Result<(), TestError> {
    for _ in 0..50 {
        match runtime.next().await? {
            Step::Continue => {}
            Step::Done(_) => return Ok(()),
            Step::Suspend(_) => return Err(TestError::Missing("unexpected suspension")),
        }
    }
    Err(TestError::Missing("execution did not complete"))
}

/// Finds the serialized agent checkpoint without depending on compiled frame indices.
fn checkpoint_mut(snapshot: &mut Value) -> Result<&mut Value, TestError> {
    let frames = snapshot
        .pointer_mut("/state/frames")
        .and_then(Value::as_array_mut)
        .ok_or(TestError::Missing("snapshot frames"))?;
    for frame in frames {
        if let Some(checkpoints) = frame.get_mut("checkpoints").and_then(Value::as_array_mut) {
            for checkpoint in checkpoints {
                if let Some(value) = checkpoint.get_mut("value")
                    && value.get("resolved").is_some()
                {
                    return Ok(value);
                }
            }
        }
    }
    Err(TestError::Missing("agent checkpoint"))
}

/// Setting a valid cap adds no allocation to an already constructed configuration.
#[test]
fn setting_a_cap_allocates_nothing() {
    let config = AgentConfig::new("openai:///test", "Answer.", Message::user("question"));
    let mut config = Some(config);
    let allocations = allocation_counter::measure(|| {
        config = config.take().map(|config| config.max_output_tokens(2048));
        std::hint::black_box(&config);
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}

/// Captures committed activation before a model dispatch, without relying on exact step counts.
async fn activated(runtime: &mut Runtime) -> Result<Snapshot, TestError> {
    for _ in 0..20 {
        let snapshot = runtime.snapshot()?;
        if checkpoint_mut(&mut serde_json::to_value(&snapshot)?).is_ok() {
            return Ok(snapshot);
        }
        assert!(matches!(runtime.next().await?, Step::Continue));
    }
    Err(TestError::Missing("activation did not finish"))
}

/// Both absent and configured caps reach clients and the borrowed preparation view unchanged.
#[tokio::test]
async fn request_cap_is_optional_and_visible_to_preparation() -> Result<(), TestError> {
    let flow = compile(workflow)?;
    for cap in [None, Some(2048), Some(u32::MAX)] {
        let script = ScriptedFactory::new().then_output(json!("answer"));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = flow
            .start(
                Request {
                    cap,
                    ..Request::capped()
                },
                context(script.clone(), cap),
            )?
            .with_compactor(ObserveCap {
                calls: calls.clone(),
                cap,
            });
        finish(&mut runtime).await?;
        assert_eq!(script.calls().len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

/// Tool redispatch and the extra forced conclusion both retain the same per-request cap.
#[tokio::test]
async fn tool_loops_and_forced_conclusion_keep_the_cap() -> Result<(), TestError> {
    let flow = compile(workflow)?;
    for turns in [None, Some(1)] {
        let script = ScriptedFactory::new()
            .then_tool_calls(vec![mock_tool_call(
                "lookup-1",
                "lookup",
                json!({"query":"evidence"}),
            )])
            .then_output(json!("answer"));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = flow
            .start(
                Request {
                    turns,
                    ..Request::capped()
                },
                context(script.clone(), Some(2048)),
            )?
            .with_compactor(ObserveCap {
                calls: calls.clone(),
                cap: Some(2048),
            });
        finish(&mut runtime).await?;
        assert_eq!(script.calls().len(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
    Ok(())
}
