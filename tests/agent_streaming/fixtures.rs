use super::*;
use futures::{StreamExt, stream};
use pravah::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, ErrorKind, LlmBackend,
    Message, ModelUrl, ProviderFactory, ProviderRegistry, TokenUsage, ToolCall,
};
use rath::llm::LlmStream;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Complete,
    ToolRound,
    StartupFailure,
    FailureAfterProgress,
    OutputLimit,
    Eof,
    EmptyTools,
    WrongOutput,
    CreationFailure,
}

#[derive(Default)]
pub(super) struct Stats {
    pub starts: AtomicUsize,
    pub ordinary: AtomicUsize,
    pub polls: AtomicUsize,
    pub records: AtomicUsize,
    pub compactions: AtomicUsize,
    pub summaries_seen: AtomicUsize,
    pub tool_results_seen: AtomicUsize,
}

struct Factory {
    mode: Mode,
    stats: Arc<Stats>,
}

struct Model {
    mode: Mode,
    stats: Arc<Stats>,
    url: ModelUrl,
    options: ClientOptions,
}

/// Installs one deterministic provider without network access or global fixture state.
pub(super) fn context(mode: Mode, stats: Arc<Stats>) -> Result<Context, GraphError> {
    let providers = ProviderRegistry::new()
        .register("test", Factory { mode, stats })
        .map_err(|source| GraphError::AgentClient {
            operation: pravah::AgentClientOperation::Create,
            source,
        })?;
    Ok(Context::default().with_providers(providers))
}

impl ProviderFactory for Factory {
    async fn llm(&self, url: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        if matches!(self.mode, Mode::CreationFailure) {
            return Err(failure());
        }
        Ok(Client::from_backend(Model {
            mode: self.mode,
            stats: self.stats.clone(),
            url: url.clone(),
            options,
        }))
    }
}

impl LlmBackend for Model {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }
    fn options(&self) -> &ClientOptions {
        &self.options
    }
    async fn execute(&self, _: &[Message]) -> Result<ClientResponse, ClientError> {
        self.stats.ordinary.fetch_add(1, Ordering::SeqCst);
        Ok(answer())
    }
    /// Records startup and poll boundaries while validating the subsequent tool surface.
    async fn execute_stream<'a>(
        &'a self,
        messages: &[Message],
    ) -> Result<LlmStream<'a>, ClientError> {
        let turn = self.stats.starts.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::StartupFailure) {
            return Err(failure());
        }
        if messages
            .iter()
            .any(|m| m.content.contains("consolidated memory"))
        {
            self.stats.summaries_seen.fetch_add(1, Ordering::SeqCst);
        }
        if messages
            .iter()
            .any(|m| matches!(m.role, pravah::clients::Role::Tool { .. }))
        {
            self.stats.tool_results_seen.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, Mode::ToolRound) {
                assert!(
                    self.options.tools.is_empty(),
                    "exhausted tool must remain hidden"
                );
            }
        }
        let events = events(self.mode, turn);
        Ok(Box::pin(stream::iter(events).inspect(|_| {
            self.stats.polls.fetch_add(1, Ordering::SeqCst);
        })))
    }
}

fn answer() -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::Output(serde_json::json!("authoritative answer")),
    )
    .with_usage(Some(TokenUsage::new().with_input(10).with_output(3)))
}

/// Emits deliberately different previews and terminal output to detect accidental history writes.
fn events(mode: Mode, turn: usize) -> Vec<Result<LlmEvent, ClientError>> {
    let mut events = vec![Ok(LlmEvent::TextDelta {
        text: "provisional ".into(),
    })];
    if matches!(mode, Mode::ToolRound) && turn == 0 {
        events.push(Ok(LlmEvent::ToolCallDelta {
            index: 7,
            id: Some("call-1".into()),
            name_delta: Some("lookup".into()),
            arguments_delta: "{".into(),
        }));
    } else if matches!(mode, Mode::Complete | Mode::ToolRound) {
        events.push(Ok(LlmEvent::TextDelta {
            text: "text".into(),
        }));
    }
    events.extend(terminal(mode, turn));
    events
}

/// Builds complete proposals separately from deliberately malformed provisional argument fragments.
fn terminal(mode: Mode, turn: usize) -> Option<Result<LlmEvent, ClientError>> {
    let response = match mode {
        Mode::FailureAfterProgress => return Some(Err(failure())),
        Mode::OutputLimit => {
            return Some(Err(ClientError::new(
                ErrorKind::OutputLimitReached,
                "partial output",
            )));
        }
        Mode::Eof => return None,
        Mode::ToolRound if turn == 0 => ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::ToolCalls {
                text: None,
                calls: vec![ToolCall::new(
                    "call-1".into(),
                    "lookup".into(),
                    serde_json::json!({"query":"evidence"}),
                )],
            },
        ),
        Mode::EmptyTools => ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::ToolCalls {
                text: None,
                calls: Vec::new(),
            },
        ),
        Mode::WrongOutput => ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::Output(serde_json::json!(42)),
        ),
        _ => answer(),
    };
    Some(Ok(LlmEvent::Completed { response }))
}

pub(super) fn failure() -> ClientError {
    ClientError::new(ErrorKind::Http, "private diagnostic")
        .with_context(Provider::OpenAi, "stream")
        .with_http_status(429)
        .with_request_id("request-stream", &[])
        .with_provider_code("overloaded", &[])
        .with_retry_after("5", &[])
        .with_source(ClientError::new(ErrorKind::Timeout, "private cause"))
}
