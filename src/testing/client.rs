use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::clients::{
    Client, ClientError, ClientFactory, ClientOptions, ClientOutput, ClientResponse, Message,
    Provider, ToolCall,
};

struct ScriptedInner {
    responses: VecDeque<Result<ClientResponse, ClientError>>,
    calls: Vec<(String, Vec<Message>)>,
}

impl ScriptedInner {
    fn new() -> Self {
        Self {
            responses: VecDeque::new(),
            calls: Vec::new(),
        }
    }
}

/// Scripted LLM client — created per-dispatch by [`ScriptedFactory`].
struct ScriptedClient {
    inner: Arc<Mutex<ScriptedInner>>,
    model_url: String,
}

impl ScriptedClient {
    fn new(inner: Arc<Mutex<ScriptedInner>>, model_url: String) -> Self {
        Self { inner, model_url }
    }
}

#[async_trait]
impl Client for ScriptedClient {
    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.calls.push((self.model_url.clone(), messages.to_vec()));
        guard
            .responses
            .pop_front()
            .unwrap_or_else(|| Err(ClientError::Llm("ScriptedClient: response queue exhausted".into())))
    }
}

/// A [`ClientFactory`] that replays a pre-programmed sequence of responses.
///
/// Each call to [`ClientFactory::create`] produces a [`Client`] that shares the
/// same response queue and call log, so responses are consumed in order regardless
/// of which model URL triggered the dispatch.
///
/// Clone the factory before injecting it to keep an inspection handle:
///
/// ```rust,ignore
/// let factory = ScriptedFactory::new()
///     .then_output(serde_json::json!({ "answer": "yes" }));
/// let spy = factory.clone();
/// let mut runtime = FlowRuntime::new(input)?.with_factory(factory);
/// // drive the flow …
/// assert_eq!(spy.calls().len(), 1);
/// ```
#[derive(Clone)]
pub struct ScriptedFactory {
    inner: Arc<Mutex<ScriptedInner>>,
}

impl ScriptedFactory {
    /// Creates an empty factory with no queued responses.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ScriptedInner::new())),
        }
    }

    /// Enqueues a pre-built response.
    pub fn then(self, response: ClientResponse) -> Self {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .responses
            .push_back(Ok(response));
        self
    }

    /// Enqueues a client error as the next response.
    pub fn then_err(self, err: ClientError) -> Self {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .responses
            .push_back(Err(err));
        self
    }

    /// Enqueues a structured-output response with the given JSON `value`.
    pub fn then_output(self, value: Value) -> Self {
        self.then(output_response(value))
    }

    /// Enqueues a tool-call response.
    pub fn then_tool_calls(self, calls: Vec<ToolCall>) -> Self {
        self.then(tool_call_response(calls))
    }

    /// Enqueues a tool-call response accompanied by a model thought string.
    pub fn then_tool_calls_with_thought(
        self,
        thought: impl Into<String>,
        calls: Vec<ToolCall>,
    ) -> Self {
        self.then(tool_call_response_with_thought(thought.into(), calls))
    }

    /// Snapshot of all `(model_url, messages)` pairs sent to the LLM so far.
    pub fn calls(&self) -> Vec<(String, Vec<Message>)> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .calls
            .clone()
    }

    /// Number of responses still waiting in the queue.
    pub fn remaining(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .responses
            .len()
    }
}

impl Default for ScriptedFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientFactory for ScriptedFactory {
    fn create(
        &self,
        model_url: &str,
        _options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError> {
        Ok(Box::new(ScriptedClient::new(
            Arc::clone(&self.inner),
            model_url.to_owned(),
        )))
    }
}

/// Builds a [`ClientResponse`] wrapping a structured JSON output value.
pub fn output_response(value: Value) -> ClientResponse {
    ClientResponse::new(Provider::OpenAi, ClientOutput::Output(value))
}

/// Builds a [`ClientResponse`] wrapping tool calls with no thought text.
pub fn tool_call_response(calls: Vec<ToolCall>) -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::ToolCalls {
            thought: None,
            calls,
        },
    )
}

/// Builds a [`ClientResponse`] wrapping tool calls with a model thought string.
pub fn tool_call_response_with_thought(thought: String, calls: Vec<ToolCall>) -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::ToolCalls {
            thought: Some(thought),
            calls,
        },
    )
}

/// Constructs a [`ToolCall`] for use in scripted test responses.
///
/// `args` must be a [`serde_json::Value`]; use `serde_json::json!({…})` inline.
pub fn mock_tool_call(
    id: impl Into<String>,
    name: impl Into<String>,
    args: Value,
) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        args,
        thought_signatures: None,
    }
}
