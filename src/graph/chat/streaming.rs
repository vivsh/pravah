//! Async progress delivery over the same synchronous Chat stepping and completion boundaries.

use super::{Chat, ChatRequest, ChatStep, ChatTurn, GraphError};
use crate::clients::LlmEvent;
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use std::future::Future;
use uuid::Uuid;

impl<I, O, S> Chat<I, O, S>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Sends typed input and awaits provisional generation events before each next stream poll.
    ///
    /// The callback receives a request UUID and Rath text/tool progress, never `Completed`.
    /// Tool rounds may create several streams; correlate progress using the UUID.
    /// Only terminal output enters history and produces the typed turn. Unsupported streaming
    /// and interrupted streams fail without fallback or retries. Cancelled unfinished turns
    /// retain the existing snapshot/manual-execution recovery boundary, not the live stream.
    /// The callback handles its own delivery errors. Internal suspension returns `ChatSuspended`.
    pub async fn send_stream<F, Fut>(
        &mut self,
        input: impl Into<ChatRequest<I>>,
        on_event: F,
    ) -> Result<ChatTurn<O>, GraphError>
    where
        F: FnMut(Uuid, LlmEvent) -> Fut + Send,
        Fut: Future<Output = ()> + Send,
    {
        self.accept_submission(input, None, "send_stream")?;
        self.finish_send_stream(on_event).await
    }

    /// Sends streaming typed input with a durable key overriding the configured user-message key.
    /// Readiness and conversion failures accept no input. Progress and failure semantics
    /// match `send_stream`; the key does not become a stream identity.
    pub async fn send_stream_with_key<F, Fut>(
        &mut self,
        input: impl Into<ChatRequest<I>>,
        key: impl Into<String>,
        on_event: F,
    ) -> Result<ChatTurn<O>, GraphError>
    where
        F: FnMut(Uuid, LlmEvent) -> Fut + Send,
        Fut: Future<Output = ()> + Send,
    {
        self.accept_submission(input, Some(key.into()), "send_stream_with_key")?;
        self.finish_send_stream(on_event).await
    }

    /// Drives the existing turn boundaries; only worker generation uses streaming transport.
    async fn finish_send_stream<F, Fut>(
        &mut self,
        mut on_event: F,
    ) -> Result<ChatTurn<O>, GraphError>
    where
        F: FnMut(Uuid, LlmEvent) -> Fut + Send,
        Fut: Future<Output = ()> + Send,
    {
        loop {
            match self.next()? {
                ChatStep::Continue => {}
                ChatStep::Agent(request) => {
                    let response = self.executor.execute_stream(&request, &mut on_event).await;
                    self.resume_agent(response)?;
                }
                ChatStep::Suspend(_) => return Err(GraphError::ChatSuspended),
                ChatStep::Done(turn) => return Ok(turn),
            }
        }
    }
}
