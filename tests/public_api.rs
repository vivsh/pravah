use pravah::clients::Message;
use pravah::{Agent, AgentConfig, Chat, Context, Flow, GraphError, Step, compile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Request {
    value: i64,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Response {
    value: i64,
}

fn workflow(root: Flow<Request>) -> Flow<Response> {
    root.map(|request| Response {
        value: request.value + 1,
    })
}

fn assistant(root: Agent<Request>) -> Agent<Response> {
    root.configure(configure_assistant)
}

/// Configures the compile-only chat agent used to verify crate-root exports.
async fn configure_assistant(request: Request, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///test",
        "Return the incremented value.",
        Message::user(request.value.to_string()),
    ))
}

/// Verifies the modern typed workflow and chat API is available at the crate root.
#[tokio::test]
async fn modern_typed_api_is_available_at_crate_root() -> Result<(), GraphError> {
    let compiled = compile(workflow)?;
    let mut execution = compiled.start(Request { value: 1 }, Context::default())?;
    let output = loop {
        match execution.next().await? {
            Step::Continue => {}
            Step::Done(value) => break compiled.decode_output(value)?,
            Step::Suspend(_) => {
                return Err(GraphError::Invalid("root export test suspended".into()));
            }
        }
    };
    assert_eq!(output, Response { value: 2 });

    let _chat = Chat::new(assistant, Context::default());
    Ok(())
}
