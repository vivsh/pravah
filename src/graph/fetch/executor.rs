//! Runtime-only external execution. This owner has no pending or completed-operation state.

use std::{collections::BTreeMap, sync::Arc};

use futures::future::BoxFuture;

use crate::graph::{AgentClientOperation, GraphError, HandlerRegistry, RuntimeServices};
use crate::{
    Context,
    history::{Compactor, HistoryStore},
};

use super::{
    Fetch, FetchBody, FetchResponse,
    rath::{RathRequest, RathResponse},
};

/// An explicitly registered external scheme implementation, never run by the VM.
pub trait DynFetchHandler: Send + Sync {
    /// Executes one logical request; the caller controls retries and completion storage.
    fn execute<'a>(
        &'a self,
        fetch: &'a Fetch,
        context: Context,
    ) -> BoxFuture<'a, Result<FetchResponse, GraphError>>;
}

/// External request dispatcher. Clone-free borrowed requests have no lifecycle state here.
/// Reinstall runtime dependencies when restoring; this value is never serialized.
pub struct FetchExecutor {
    context: Context,
    registry: Arc<HandlerRegistry>,
    services: RuntimeServices,
    schemes: BTreeMap<String, Arc<dyn DynFetchHandler>>,
}

impl FetchExecutor {
    pub(crate) fn with_services(mut self, services: RuntimeServices) -> Self {
        self.services = services;
        self
    }
    /// Creates a dispatcher using the same immutable handlers as the prepared graph.
    pub fn new(context: Context, registry: Arc<HandlerRegistry>) -> Self {
        Self {
            context,
            registry,
            services: RuntimeServices::default(),
            schemes: BTreeMap::new(),
        }
    }

    /// Borrows the runtime dependencies installed by the host.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Borrows the registered graph handlers, without copying the registry.
    pub fn registry(&self) -> &HandlerRegistry {
        &self.registry
    }

    /// Borrows the external history-service bundle.
    pub fn services(&self) -> &RuntimeServices {
        &self.services
    }

    /// Installs pre-dispatch working-memory preparation outside the VM.
    pub fn with_compactor(mut self, compactor: impl Compactor + 'static) -> Self {
        self.services = self.services.with_compactor(compactor);
        self
    }

    /// Installs the store used for acknowledged history batches.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.services = self.services.with_store(store);
        self
    }

    /// Registers a unique application scheme. Built-in schemes cannot be shadowed.
    pub fn register(
        &mut self,
        scheme: &str,
        handler: impl DynFetchHandler + 'static,
    ) -> Result<(), GraphError> {
        if !valid_scheme(scheme)
            || matches!(scheme, "http" | "https" | "rath" | "pravah")
            || self.schemes.contains_key(scheme)
        {
            return Err(GraphError::FetchValidation(
                "invalid, reserved or duplicate scheme".into(),
            ));
        }
        self.schemes.insert(scheme.to_owned(), Arc::new(handler));
        Ok(())
    }

    /// Executes exactly one selected operation, with no retry or protocol fallback.
    /// Errors retain local sources; explicitly convert them before durable failure delivery.
    pub async fn execute(&self, fetch: &Fetch) -> Result<FetchResponse, GraphError> {
        let scheme = fetch
            .request()
            .url()
            .split_once(':')
            .map(|(scheme, _)| scheme)
            .filter(|scheme| valid_scheme(scheme))
            .ok_or_else(|| GraphError::FetchValidation("request has no valid scheme".into()))?;
        match scheme {
            "http" | "https" => self.http(fetch).await,
            "rath" => self.rath(fetch).await,
            "pravah" => {
                crate::graph::agent::execute_hook(
                    fetch,
                    &self.context,
                    &self.registry,
                    &self.services,
                )
                .await
            }
            _ => match self.schemes.get(scheme) {
                Some(handler) => handler.execute(fetch, self.context.clone()).await,
                None => Err(GraphError::FetchValidation(
                    "unsupported request scheme".into(),
                )),
            },
        }
    }

    /// Constructs and executes an ordinary Rath client; original typed errors remain intact.
    async fn rath(&self, fetch: &Fetch) -> Result<FetchResponse, GraphError> {
        let request = RathRequest::from_fetch_request(fetch.request())?;
        let (model, options, messages) = request.into_parts();
        let client = self
            .context
            .providers()
            .llm(&model, options)
            .await
            .map_err(|source| GraphError::AgentClient {
                operation: AgentClientOperation::Create,
                source,
            })?;
        let response = client.execute(&messages).await.map_err(|source| {
            let source = if source.provider().is_none() {
                source.with_context(client.provider(), "generation")
            } else {
                source
            };
            GraphError::AgentClient {
                operation: AgentClientOperation::Execute,
                source,
            }
        })?;
        RathResponse::new(response).into_fetch_response()
    }

    /// Buffers one HTTP response, leaving status interpretation and retry policy to the caller.
    async fn http(&self, fetch: &Fetch) -> Result<FetchResponse, GraphError> {
        let request = fetch.request();
        let method = reqwest::Method::from_bytes(request.method().as_bytes())
            .map_err(|_| GraphError::FetchValidation("invalid HTTP method".into()))?;
        let mut builder = self.context.http_client().request(method, request.url());
        for (name, value) in request.headers() {
            builder = builder.header(name, value.as_slice());
        }
        match request.body_ref() {
            Some(FetchBody::Bytes(bytes)) => builder = builder.body(bytes.to_vec()),
            Some(FetchBody::Value(value)) => builder = builder.json(value),
            None => {}
        }
        let response = builder.send().await.map_err(http_error)?;
        let mut result = FetchResponse::new(response.status().as_u16());
        for (name, value) in response.headers() {
            result = result.header(name.as_str(), value.as_bytes());
        }
        let bytes = response.bytes().await.map_err(http_error)?;
        Ok(result.body(FetchBody::Bytes(Arc::from(bytes.as_ref()))))
    }
}

fn valid_scheme(scheme: &str) -> bool {
    let mut bytes = scheme.bytes();
    bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')
        })
}

fn http_error(source: reqwest::Error) -> GraphError {
    GraphError::FetchTransport {
        source: source.without_url(),
    }
}

#[cfg(test)]
#[path = "tests/executor.rs"]
mod tests;
