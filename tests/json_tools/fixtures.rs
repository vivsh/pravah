use super::*;
use pravah::clients::{
    Client, ClientError, ClientOptions, ModelUrl, ProviderFactory, ToolDefinition,
};
use pravah::deps::Deps;
use pravah::tools::ToolError;
use pravah::{Chat, ChatBuilder, Context};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicUsize};

#[derive(Debug, thiserror::Error)]
/// Structured failures used by the catalogue execution and transport fixtures.
pub(super) enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Value(#[from] pravah::graph::ValueError),
    #[error(transparent)]
    CborEncode(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    CborDecode(#[from] ciborium::de::Error<std::io::Error>),
    #[error("missing {0}")]
    Missing(&'static str),
}

#[derive(Default)]
/// Application-owned counters and a current permission flag shared through Context.
pub(super) struct Proxy {
    pub calls: AtomicUsize,
    pub denied: AtomicBool,
}

/// Creates generic JSON operation records without endpoint-specific Rust types.
pub(super) fn catalogue() -> Value {
    json!([
        {"alias":"find_staff_member", "operation":"staff.retrieve", "description":"Find staff",
         "input_schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object",
            "properties":{"staff_id":{"type":"integer","minimum":1}},
            "required":["staff_id"],"additionalProperties":false},
         "output_schema":{"type":"object","required":["staff_id","email"],"additionalProperties":false,
            "properties":{"staff_id":{"type":"integer"},"email":{"type":["string","null"],"format":"email"}}}},
        {"alias":"list_staff_visits", "operation":"visits.list", "description":"List visits",
         "input_schema":{"type":"object","properties":{"staff_id":{"type":"integer","minimum":1},
            "on":{"type":"string","format":"date"}},"required":["staff_id"],"additionalProperties":false},
         "output_schema":{"type":"array","items":{"type":"object","properties":{"status":{"enum":["arrived","left"]}},
            "required":["status"],"additionalProperties":false}}}
    ])
}

/// Consumes a fetched-catalogue-shaped JSON value in an immediate capturing toolset builder.
pub(super) fn builder(catalogue: Value) -> ChatBuilder<String, String> {
    Chat::builder()
        .model("test:///test")
        .instructions("Assist staff")
        .tools(move |mut tools| {
            for record in catalogue.as_array().expect("catalogue") {
                let operation = record["operation"].as_str().expect("operation").to_owned();
                let definition = ToolDefinition::new(
                    record["alias"].as_str().expect("alias").into(),
                    record["description"].as_str().expect("description").into(),
                    record["input_schema"].clone(),
                );
                tools = tools.json(
                    definition,
                    Some(record["output_schema"].clone()),
                    move |args, ctx| proxy(operation.clone(), args, ctx),
                );
            }
            tools
        })
}

/// Shares one generic proxy, with the trusted operation outside model-controlled arguments.
async fn proxy(operation: String, args: Value, ctx: Context) -> Result<Value, ToolError> {
    let proxy = ctx.require::<Proxy>()?;
    if proxy.denied.load(Ordering::SeqCst) {
        return Err(ToolError::Security("permission denied".into()));
    }
    proxy.calls.fetch_add(1, Ordering::SeqCst);
    match operation.as_str() {
        "staff.retrieve" => Ok(json!({"staff_id":args["staff_id"],"email":null})),
        "visits.list" => Ok(json!([{"status":"arrived"}])),
        _ => Err(ToolError::Fatal("operation not allowlisted".into())),
    }
}

/// Installs the shared proxy and scripted provider without production services.
pub(super) fn context(
    factory: &ScriptedFactory,
    proxy: &Arc<Proxy>,
) -> Result<Context, GraphError> {
    let mut deps = Deps::default();
    deps.insert(Arc::clone(proxy));
    Ok(Context::default()
        .with_deps(deps)
        .with_providers(pravah::testing::providers(factory.clone())?))
}

struct InspectFactory {
    script: ScriptedFactory,
    catalogue: Value,
}

impl ProviderFactory for InspectFactory {
    async fn llm(&self, url: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        for tool in &options.tools {
            let record = self
                .catalogue
                .as_array()
                .expect("catalogue")
                .iter()
                .find(|entry| entry["alias"] == tool.name)
                .expect("registered alias");
            assert_eq!(tool.parameters, record["input_schema"]);
            assert_eq!(
                tool.description,
                record["description"].as_str().expect("description")
            );
        }
        self.script.llm(url, options).await
    }
}

/// Checks model-visible declarations while executing with the same shared proxy.
pub(super) fn inspected_context(
    factory: &ScriptedFactory,
    catalogue: &Value,
    proxy: &Arc<Proxy>,
) -> Result<Context, GraphError> {
    let mut deps = Deps::default();
    deps.insert(Arc::clone(proxy));
    Ok(Context::default()
        .with_deps(deps)
        .with_providers(pravah::testing::providers(InspectFactory {
            script: factory.clone(),
            catalogue: catalogue.clone(),
        })?))
}

/// Proposes one valid call using the catalogue's default staff alias.
pub(super) fn staff_call() -> ToolCall {
    ToolCall::new(
        "staff".into(),
        "find_staff_member".into(),
        json!({"staff_id":42}),
    )
}

/// Reads accepted tool history as JSON without consuming any pending execution.
pub(super) fn tool_outputs(snapshot: &Snapshot) -> Vec<Value> {
    snapshot
        .history()
        .entries()
        .iter()
        .filter(|row| matches!(row.message.role, Role::Tool { .. }))
        .map(|row| serde_json::from_str(&row.message.content).expect("tool JSON"))
        .collect()
}

/// Executes configuration/generation until a tool request is waiting, without executing that tool.
pub(super) async fn advance_to_tool(
    chat: &mut Chat<String, String>,
) -> Result<pravah::AgentRequest, TestError> {
    for _ in 0..100 {
        match chat.next()? {
            ChatStep::Agent(request) if request.kind() == "tool" => return Ok(request),
            ChatStep::Agent(request) => {
                chat.resume_agent(chat.executor().execute(&request).await)?
            }
            ChatStep::Continue => {}
            _ => return Err(TestError::Missing("tool request")),
        }
    }
    Err(TestError::Missing("bounded tool dispatch"))
}

/// Completes existing manual stepping without invoking another send on an unfinished turn.
pub(super) async fn finish(chat: &mut Chat<String, String>) -> Result<String, TestError> {
    if let Some(request) = chat.pending_agent().cloned() {
        chat.resume_agent(chat.executor().execute(&request).await)?;
    }
    for _ in 0..100 {
        match chat.next()? {
            ChatStep::Agent(request) => {
                chat.resume_agent(chat.executor().execute(&request).await)?
            }
            ChatStep::Continue => {}
            ChatStep::Done(turn) => return Ok(turn.output),
            _ => return Err(TestError::Missing("response")),
        }
    }
    Err(TestError::Missing("bounded completion"))
}
