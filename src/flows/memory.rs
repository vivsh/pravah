use std::future::Future;

use futures::future::BoxFuture;
use serde_json::Value;

use crate::context::Context;

/// Convenience alias for the return type of [`MemoryFactory::retrieve`].
///
/// `Ok(Some(string))` — inject this string into the system prompt.
/// `Ok(None)` — no context to inject; proceed without it.
/// `Err(e)` — retrieval failed; the flow will halt with [`FlowError::MemoryError`](crate::flows::FlowError).
pub type MemoryResult = Result<Option<String>, Box<dyn std::error::Error + Send + Sync>>;

/// Query passed to [`MemoryFactory::retrieve`].
///
/// Carries all per-invocation context needed to retrieve relevant memories or
/// dynamic context for an agent's system prompt.
pub struct MemoryQuery<'a> {
    /// Registered node id of the agent being dispatched.
    ///
    /// Use this to route different retrieval strategies per agent.
    pub agent_name: &'a str,
    /// Raw agent input as a JSON value.
    ///
    /// Extract query text, user id, or any other field your schema exposes:
    /// `query.input["user_id"].as_str()`.
    pub input: &'a Value,
    /// Shared execution context.
    ///
    /// Provides an HTTP client (`ctx.http_client()`), working directory, and
    /// registered external dependencies — enough to call any retrieval service.
    pub ctx: &'a Context,
}

/// Injects dynamic context into an agent's system prompt before each invocation.
///
/// Implement this trait to retrieve memories, RAG results, or any other
/// per-request context from an external source. The returned string is inserted
/// between the static preamble and the input-schema hint in the system prompt.
///
/// The result is cached in [`AgentState`](crate::flows::FlowRuntime) for the
/// lifetime of a single agent invocation, so `retrieve` is called at most once
/// per agent dispatch regardless of how many LLM turns occur.
///
/// # Example
///
/// ```rust,no_run
/// use pravah::flows::memory::{MemoryFactory, MemoryQuery, MemoryResult};
///
/// struct MyRetriever;
///
/// impl MemoryFactory for MyRetriever {
///     async fn retrieve(&self, query: &MemoryQuery<'_>) -> MemoryResult {
///         // Call your vector DB, REST API, local cache, etc.
///         let text = query.input["query"].as_str().ok_or("missing query field")?;
///         Ok(Some(format!("Relevant context: {text}")))
///     }
/// }
/// ```
pub trait MemoryFactory {
    fn retrieve(&self, query: &MemoryQuery<'_>) -> impl Future<Output = MemoryResult> + Send;
}

/// A [`MemoryFactory`] that always returns `None`. Default for [`FlowRuntime`](crate::flows::FlowRuntime).
pub struct NoopMemoryFactory;

impl MemoryFactory for NoopMemoryFactory {
    async fn retrieve(&self, _query: &MemoryQuery<'_>) -> MemoryResult {
        Ok(None)
    }
}

/// Object-safe version of [`MemoryFactory`] used internally for type erasure.
pub(crate) trait DynMemoryFactory: Send + Sync + 'static {
    fn retrieve_dyn<'a>(&'a self, query: &'a MemoryQuery<'a>) -> BoxFuture<'a, MemoryResult>;
}

impl<T> DynMemoryFactory for T
where
    T: MemoryFactory + Send + Sync + 'static,
{
    fn retrieve_dyn<'a>(&'a self, query: &'a MemoryQuery<'a>) -> BoxFuture<'a, MemoryResult> {
        Box::pin(self.retrieve(query))
    }
}
