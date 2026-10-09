//! Operation-local Rath streaming; provisional events never enter VM state or history.

use super::{AgentError, AgentExecutor, AgentOperation, AgentRequest, AgentResponse};
use crate::clients::{ClientError, ErrorKind, LlmEvent};
use crate::graph::{AgentClientOperation, GraphError, Value};
use futures::StreamExt;
use std::future::Future;
use uuid::Uuid;

impl AgentExecutor {
    /// Completes one operation, streaming provisional generation events to an async callback.
    ///
    /// Each event carries the request UUID. The callback is awaited before polling again,
    /// providing backpressure without a queue. Only progress events are forwarded; terminal
    /// output becomes the returned completion, with existing history acknowledgements.
    /// Other operations execute normally without events. Unsupported streaming, startup
    /// errors, mid-stream failures and premature EOF return portable failures without fallback.
    /// Dropping this future stops local consumption, not necessarily remote generation/billing.
    /// The callback handles its own delivery errors; it cannot modify the runtime through this API.
    pub async fn execute_stream<F, Fut>(
        &self,
        request: &AgentRequest,
        mut on_event: F,
    ) -> AgentResponse
    where
        F: FnMut(Uuid, LlmEvent) -> Fut + Send,
        Fut: Future<Output = ()> + Send,
    {
        if !matches!(request.operation.as_ref(), AgentOperation::Generate { .. }) {
            return self.execute(request).await;
        }
        let mut response = AgentResponse::new(request.id(), Ok(Value::NULL));
        let result = self
            .generate_stream(request, &mut response, &mut on_event)
            .await;
        finish_response(response, result)
    }

    /// Runs the same ordered stages as ordinary execution, then consumes one generation.
    async fn generate_stream<F, Fut>(
        &self,
        request: &AgentRequest,
        completion: &mut AgentResponse,
        on_event: &mut F,
    ) -> Result<Value, GraphError>
    where
        F: FnMut(Uuid, LlmEvent) -> Fut + Send,
        Fut: Future<Output = ()> + Send,
    {
        self.persist(request.persist.as_deref().unwrap_or_default(), completion)
            .await?;
        let (client, messages) = self.prepare_generation(request, completion).await?;
        let mut stream = client
            .execute_stream(&messages)
            .await
            .map_err(|source| execution_error(&client, source))?;
        while let Some(event) = stream.next().await {
            match event.map_err(|source| execution_error(&client, source))? {
                LlmEvent::Completed { response } => {
                    return super::super::client_response::serialize_value(&response);
                }
                progress => on_event(request.id(), progress).await,
            }
        }
        Err(execution_error(
            &client,
            ClientError::new(
                ErrorKind::InvalidResponse,
                "stream ended without completion",
            ),
        ))
    }

    /// Reuses the ordinary history/client helpers without changing its non-streaming future layout.
    async fn prepare_generation(
        &self,
        request: &AgentRequest,
        completion: &mut AgentResponse,
    ) -> Result<(crate::clients::Client, Vec<crate::clients::Message>), GraphError> {
        let AgentOperation::Generate { model, options, .. } = request.operation.as_ref() else {
            return Err(GraphError::AgentRequestValidation(
                "expected generation".into(),
            ));
        };
        let options = super::super::options::deserialize(options)
            .map_err(|error| GraphError::AgentRequestValidation(error.to_string()))?;
        let messages = self
            .prepare_messages(request.operation.as_ref(), &options, completion)
            .await?;
        let client = self.client(model, options).await?;
        Ok((client, messages))
    }
}

/// Preserves completed stages even when the final operation returns a portable failure.
fn finish_response(
    mut response: AgentResponse,
    result: Result<Value, GraphError>,
) -> AgentResponse {
    response.outcome = result.map_err(|error| {
        AgentError::from_execution_error(&error).unwrap_or_else(|_| {
            AgentError::new("diagnostics", "could not encode failure diagnostics")
        })
    });
    response
}

/// Normalizes execution failures without flattening Rath's typed diagnostics.
fn execution_error(client: &crate::clients::Client, source: ClientError) -> GraphError {
    let source = if source.provider().is_none() {
        source.with_context(client.provider(), "generation")
    } else {
        source
    };
    GraphError::AgentClient {
        operation: AgentClientOperation::Execute,
        source,
    }
}
