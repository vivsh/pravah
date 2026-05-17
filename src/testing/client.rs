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

/// Scripted client used by tests.
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
    fn provider(&self) -> Provider {
        Provider::OpenAi
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.calls.push((self.model_url.clone(), messages.to_vec()));
        guard
            .responses
            .pop_front()
            .unwrap_or_else(|| Err(ClientError::Llm("ScriptedClient: response queue exhausted".into())))
    }
}

/// [`ClientFactory`] that replays a programmed sequence of responses.
/// All created clients share the same response queue and call log.
#[derive(Clone)]
pub struct ScriptedFactory {
    inner: Arc<Mutex<ScriptedInner>>,
}

impl ScriptedFactory {
    /// Creates an empty factory.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ScriptedInner::new())),
        }
    }

    /// Queues a pre-built response.
    pub fn then(self, response: ClientResponse) -> Self {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .responses
            .push_back(Ok(response));
        self
    }

    /// Queues a client error.
    pub fn then_err(self, err: ClientError) -> Self {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .responses
            .push_back(Err(err));
        self
    }

    /// Queues a structured-output response.
    pub fn then_output(self, value: Value) -> Self {
        self.then(output_response(value))
    }

    /// Queues a tool-call response.
    pub fn then_tool_calls(self, calls: Vec<ToolCall>) -> Self {
        self.then(tool_call_response(calls))
    }

    /// Queues a tool-call response with model thought text.
    pub fn then_tool_calls_with_thought(
        self,
        thought: impl Into<String>,
        calls: Vec<ToolCall>,
    ) -> Self {
        self.then(tool_call_response_with_thought(thought.into(), calls))
    }

    /// Returns all `(model_url, messages)` pairs sent so far.
    pub fn calls(&self) -> Vec<(String, Vec<Message>)> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .calls
            .clone()
    }

    /// Returns the number of queued responses left.
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

/// Builds a structured-output test response.
pub fn output_response(value: Value) -> ClientResponse {
    ClientResponse::new(Provider::OpenAi, ClientOutput::Output(value))
}

/// Builds a tool-call test response without thought text.
pub fn tool_call_response(calls: Vec<ToolCall>) -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::ToolCalls {
            thought: None,
            calls,
        },
    )
}

/// Builds a tool-call test response with thought text.
pub fn tool_call_response_with_thought(thought: String, calls: Vec<ToolCall>) -> ClientResponse {
    ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::ToolCalls {
            thought: Some(thought),
            calls,
        },
    )
}

/// Builds a [`ToolCall`] for scripted test responses.
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
